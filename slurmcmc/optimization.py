from __future__ import annotations

import dataclasses
import logging
import signal
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Literal, Optional, Union

import nevergrad as ng
import numpy as np
import submitit

from slurmcmc.general_utils import set_logging, save_restart_file, load_restart_file, combine_args, point_to_tuple, \
    signal_handler
from slurmcmc.import_utils import deferred_import_function_wrapper
from slurmcmc.slurm_utils import Cluster, SlurmPool, submit_remote_run


@dataclass
class MinimizeConfig:
    """
    Complete, picklable description of a slurm_minimize run.

    This is the single source of truth for a run's parameters: `Minimizer`
    consumes it, and remote mode pickles it into the driver Slurm job (instead
    of the old, fragile `locals()` re-submission trick).
    """
    loss_fun: Union[Callable, Dict]
    param_bounds: List
    num_workers: int
    num_iters: int
    optimizer_package: Literal['nevergrad', 'botorch'] = 'nevergrad'
    optimizer_class: Optional[Any] = None
    botorch_kwargs: Optional[Dict] = None
    init_points: Optional[List] = None
    constraint_fun: Optional[Union[Callable, Dict]] = None
    num_asks_max: int = int(1e3)
    verbosity: int = 1
    slurm_verbosity: int = 0
    log_file: Optional[str] = None
    extra_arg: Any = None
    save_restart: bool = False
    load_restart: bool = False
    restart_file: str = 'opt_restart.pkl'
    work_dir: str = 'minimize'
    job_name: str = 'minimize'
    cluster: Cluster = 'slurm'
    submitit_kwargs: Optional[Dict] = None
    budget: int = int(1e6)
    job_fail_value: float = np.nan
    submit_retry_max_attempts: int = 5
    submit_retry_wait_seconds: float = 10
    submit_delay_seconds: float = 0
    check_output_interval_seconds: float = 1
    check_output_timeout_minutes: float = int(1e5)
    restart_save_interval: int = 1
    install_signal_handler: bool = True
    # remote run params:
    remote: bool = False
    remote_cluster: Literal['slurm', 'local'] = 'slurm'
    remote_submitit_kwargs: Optional[Dict] = None


class Minimizer:
    """
    Runner for parallel black-box optimization on a Slurm cluster
    (submitit + nevergrad / botorch).

    Keeps drawing points with optimizer.ask() until num_workers points are found
    that were not already evaluated and that pass constraint_fun, evaluates them
    in parallel through SlurmPool, and feeds the results back to the optimizer.

    Usage:
        result = Minimizer(MinimizeConfig(loss_fun=..., param_bounds=..., ...)).run()

    Most callers use the `slurm_minimize(...)` convenience wrapper instead.
    """

    def __init__(self, config: MinimizeConfig) -> None:
        self.cfg = config
        self.constraint_fun: Optional[Callable] = None
        self.optimizer: Any = None
        self.optimizer_package: str = config.optimizer_package
        self.instrum = None  # nevergrad parametrization (fresh nevergrad runs only)
        self.slurm_pool: Optional[SlurmPool] = None
        # running state (restored from the restart file when load_restart=True)
        self.ini_iter: int = 0
        self.x_min = None
        self.loss_min: float = np.inf
        self.loss_min_per_iter: List = []
        self.loss_min_all_iter: List = []
        self.num_workers_per_iter: List[int] = []
        self.loc_point_min_per_iter: List = []
        self.loc_point_min_all_iter = None
        self.num_loss_fun_calls_total: int = 0
        self.num_constraint_fun_calls_total: int = 0
        self.num_asks_total: int = 0
        self.candidates_ask_time_per_iter: List[float] = []
        self._status: Optional[Dict] = None

    # ------------------------------------------------------------------
    # Setup / state
    # ------------------------------------------------------------------

    def _create_optimizer(self) -> None:
        cfg = self.cfg
        lower_bounds = [b[0] for b in cfg.param_bounds]
        upper_bounds = [b[1] for b in cfg.param_bounds]

        if self.optimizer_package == 'nevergrad':
            self.instrum = ng.p.Instrumentation(
                ng.p.Array(init=[(l + u) / 2 for l, u in zip(lower_bounds, upper_bounds)])
                .set_bounds(lower=lower_bounds, upper=upper_bounds))
            optimizer_class = cfg.optimizer_class
            if optimizer_class is None:
                optimizer_class = ng.optimizers.DifferentialEvolution(crossover="twopoints",
                                                                      popsize=cfg.num_workers)
            self.optimizer = optimizer_class(parametrization=self.instrum, num_workers=cfg.num_workers)
        elif self.optimizer_package == 'botorch':
            from slurmcmc.botorch_optimizer import BoTorchOptimizer  # deferred: torch/botorch are heavy imports
            optimizer_class = cfg.optimizer_class
            if optimizer_class is None:
                optimizer_class = BoTorchOptimizer

            botorch_kwargs = dict(cfg.botorch_kwargs or {})  # copy — don't mutate caller's dict
            botorch_defaults = {'num_restarts': 10, 'raw_samples': 100, 'num_best_points': None,
                                'options': None, 'sequential': True}
            for key, value in botorch_defaults.items():
                if key not in botorch_kwargs:
                    botorch_kwargs[key] = value

            self.optimizer = optimizer_class(lower_bounds, upper_bounds, cfg.num_workers, **botorch_kwargs)
        else:
            err_msg = f'invalid optimizer_package: {self.optimizer_package}'
            logging.error(err_msg)
            raise ValueError(err_msg)

    def _create_slurm_pool(self) -> None:
        cfg = self.cfg
        self.slurm_pool = SlurmPool(cfg.work_dir, cfg.job_name, cfg.cluster, verbosity=cfg.slurm_verbosity,
                                    log_file=cfg.log_file, extra_arg=cfg.extra_arg,
                                    submitit_kwargs=dict(cfg.submitit_kwargs or {}),
                                    dim_input=len(cfg.param_bounds), dim_output=1, budget=cfg.budget,
                                    job_fail_value=cfg.job_fail_value,
                                    submit_retry_max_attempts=cfg.submit_retry_max_attempts,
                                    submit_retry_wait_seconds=cfg.submit_retry_wait_seconds,
                                    submit_delay_seconds=cfg.submit_delay_seconds,
                                    check_output_interval_seconds=cfg.check_output_interval_seconds,
                                    check_output_timeout_minutes=cfg.check_output_timeout_minutes,
                                    record_history=True,  # required in this implementation
                                    )

    def _load_state(self) -> None:
        cfg = self.cfg
        if cfg.verbosity >= 1:
            logging.info('loading restart file: ' + cfg.work_dir + '/' + cfg.restart_file)
        status = load_restart_file(cfg.work_dir, cfg.restart_file)
        self.optimizer = status['optimizer']
        self.optimizer_package = status['optimizer_package']
        self.instrum = None  # only needed at iter 0 with init_points, which never recurs after a restart
        self.x_min = status['x_min']
        self.loss_min_per_iter = status['loss_min_per_iter']
        self.loss_min_all_iter = status['loss_min_all_iter']
        self.num_workers_per_iter = status['num_workers_per_iter']
        self.loss_min = status['loss_min']
        self.loc_point_min_per_iter = status['loc_point_min_per_iter']
        self.loc_point_min_all_iter = status['loc_point_min_all_iter']
        self.slurm_pool = status['slurm_pool']
        self.ini_iter = status['ini_iter']
        self.num_loss_fun_calls_total = status['num_loss_fun_calls_total']
        self.num_constraint_fun_calls_total = status['num_constraint_fun_calls_total']
        self.num_asks_total = status['num_asks_total']
        self.candidates_ask_time_per_iter = status['candidates_ask_time_per_iter']
        self._status = status

    def _build_status(self, curr_iter: int) -> Dict:
        return {
            'optimizer': self.optimizer,
            'optimizer_package': self.optimizer_package,
            'x_min': self.x_min,
            'loss_min_per_iter': self.loss_min_per_iter,
            'loss_min_all_iter': self.loss_min_all_iter,
            'num_workers_per_iter': self.num_workers_per_iter,
            'loss_min': self.loss_min,
            'loc_point_min_per_iter': self.loc_point_min_per_iter,
            'loc_point_min_all_iter': self.loc_point_min_all_iter,
            'slurm_pool': self.slurm_pool,
            'ini_iter': curr_iter + 1,
            'num_loss_fun_calls_total': self.num_loss_fun_calls_total,
            'num_constraint_fun_calls_total': self.num_constraint_fun_calls_total,
            'num_asks_total': self.num_asks_total,
            'candidates_ask_time_per_iter': self.candidates_ask_time_per_iter,
        }

    # ------------------------------------------------------------------
    # Candidate selection
    # ------------------------------------------------------------------

    def _constraint_passed(self, candidate) -> bool:
        passed = self.constraint_fun(*combine_args(candidate, self.cfg.extra_arg)) <= 0
        self.num_constraint_fun_calls_total += 1
        return passed

    def _init_points_candidates(self) -> tuple:
        """Iteration 0 with user-supplied init_points."""
        cfg = self.cfg
        candidates = cfg.init_points
        candidates_nevergrad = []
        if self.optimizer_package == 'nevergrad':
            # construct candidates in the nevergrad format so they can be told to the optimizer
            for init_point in candidates:
                candidate_nevergrad = self.instrum.spawn_child()
                candidate_nevergrad.value = ((init_point,), {})
                candidates_nevergrad.append(candidate_nevergrad)

        # check init_points satisfy the constraint_fun
        if self.constraint_fun is not None:
            for ind_candidate, candidate in enumerate(candidates):
                if not self._constraint_passed(candidate):
                    err_msg = f'init point index {ind_candidate} does not satisfy constraint.'
                    logging.error(err_msg)
                    raise ValueError(err_msg)

        return candidates, candidates_nevergrad

    def _ask_candidates(self) -> tuple:
        """Draw new, unevaluated, constraint-satisfying candidates from the optimizer."""
        cfg = self.cfg
        candidates: List = []
        candidates_nevergrad: List = []
        candidates_set: set = set()
        num_asks = 0
        while len(candidates) < cfg.num_workers:
            if self.optimizer_package == 'nevergrad':
                candidate_nevergrad = self.optimizer.ask()
                candidate = candidate_nevergrad.value[0][0]
                candidates_batch = [candidate]
            elif self.optimizer_package == 'botorch':
                x_pts = self.slurm_pool.points_history
                y_pts = self.slurm_pool.values_history
                t_start_ask = time.time()
                candidates_batch = self.optimizer.ask(x_pts, y_pts)
                if cfg.verbosity >= 3:
                    logging.info(f'    botorch ask run time: {(time.time() - t_start_ask):.1f}s.')

            num_asks += 1
            if num_asks > cfg.num_asks_max:
                err_msg = (f'num_asks exceeded num_asks_max= {cfg.num_asks_max}'
                           f', having trouble finding candidates that pass constraints.')
                logging.error(err_msg)
                raise ValueError(err_msg)

            for candidate in candidates_batch:
                candidate_tuple = point_to_tuple(candidate)
                proceed_with_candidate = ((candidate_tuple not in self.slurm_pool.evaluated_points_set)
                                          and (candidate_tuple not in candidates_set))

                if proceed_with_candidate:
                    if self.constraint_fun is not None and not self._constraint_passed(candidate):
                        continue
                    candidates_set.add(candidate_tuple)
                    candidates += [candidate]
                    if self.optimizer_package == 'nevergrad':
                        candidates_nevergrad += [candidate_nevergrad]
                    if len(candidates) == cfg.num_workers:
                        break

        if cfg.verbosity >= 3:
            logging.info('    optimizer.ask was called ' + str(num_asks) + ' times.')
        self.num_asks_total += num_asks

        return candidates, candidates_nevergrad

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self) -> Dict:
        cfg = self.cfg
        set_logging(cfg.work_dir, cfg.log_file)
        if cfg.install_signal_handler:
            signal.signal(signal.SIGTERM, signal_handler)  # force termination on scancel

        if cfg.constraint_fun is not None:
            self.constraint_fun = deferred_import_function_wrapper(cfg.constraint_fun)

        if cfg.load_restart:
            self._load_state()
        else:
            self._create_optimizer()
            self._create_slurm_pool()

        for curr_iter in range(self.ini_iter, self.ini_iter + cfg.num_iters):
            if cfg.verbosity >= 1:
                logging.info('### curr opt iter: ' + str(curr_iter))

            # ask for points for current iteration
            t_start_ask_curr_iter = time.time()
            if curr_iter == 0 and cfg.init_points is not None:
                candidates, candidates_nevergrad = self._init_points_candidates()
            else:
                candidates, candidates_nevergrad = self._ask_candidates()
            # appended for both branches so its length always equals the number of iterations
            self.candidates_ask_time_per_iter += [time.time() - t_start_ask_curr_iter]

            # calculate loss_fun on current iteration candidates
            results = self.slurm_pool.map(cfg.loss_fun, candidates)
            self.num_loss_fun_calls_total += len(candidates)

            # inform the optimizer with the new data
            if self.optimizer_package == 'nevergrad':
                for candidate_nevergrad, result in zip(candidates_nevergrad, results):
                    self.optimizer.tell(candidate_nevergrad, result)
            elif self.optimizer_package == 'botorch':
                # the data is already contained in slurm_pool
                pass

            # evaluate optimization metrics post current iteration
            results_arr = np.array([r if r is not None else np.nan for r in results], dtype=float)
            if np.all(np.isnan(results_arr)):
                err_msg = f'all {len(results)} evaluations failed in iteration {curr_iter}, cannot proceed.'
                logging.error(err_msg)
                raise RuntimeError(err_msg)
            ind_curr_iter_min = np.nanargmin(results_arr)
            curr_iter_x_min, curr_iter_loss_min = candidates[ind_curr_iter_min], results[ind_curr_iter_min]
            curr_iter_loc_point_min = (curr_iter, ind_curr_iter_min)
            if curr_iter_loss_min < self.loss_min:
                self.loss_min = curr_iter_loss_min
                self.x_min = curr_iter_x_min
                self.loc_point_min_all_iter = curr_iter_loc_point_min
            self.loss_min_all_iter += [self.loss_min]
            self.loss_min_per_iter += [curr_iter_loss_min]
            self.num_workers_per_iter += [len(candidates)]  # equals num_workers except possibly at iter 0
            self.loc_point_min_per_iter += [curr_iter_loc_point_min]

            if cfg.verbosity >= 2:
                logging.info(f'    curr loss_min: {self.loss_min}, curr x_min: {self.x_min}')
            elif cfg.verbosity >= 1:
                logging.info(f'    curr loss_min: {self.loss_min}')

            self._status = self._build_status(curr_iter)

            if cfg.save_restart and np.mod(curr_iter, cfg.restart_save_interval) == 0:
                if cfg.verbosity >= 3:
                    logging.info('    saving restart file: ' + cfg.work_dir + '/' + cfg.restart_file)
                save_restart_file(self._status, cfg.work_dir, cfg.restart_file)

        if cfg.verbosity >= 1:
            logging.info(f'### opt loop done. x_min: {self.x_min}, loss_min: {self.loss_min}')

        return self._status


def run_minimize(config: MinimizeConfig) -> Dict:
    """Run a Minimizer from a config object. Module-level so remote mode can pickle it by reference."""
    return Minimizer(config).run()


def slurm_minimize(
        loss_fun: Union[Callable, Dict],
        param_bounds: List,
        num_workers: int,
        num_iters: int,
        optimizer_package: Literal['nevergrad', 'botorch'] = 'nevergrad',
        optimizer_class: Optional[Any] = None,
        botorch_kwargs: Optional[Dict] = None,
        init_points: Optional[List] = None,
        constraint_fun: Optional[Union[Callable, Dict]] = None,
        num_asks_max: int = int(1e3),
        verbosity: int = 1,
        slurm_verbosity: int = 0,
        log_file: Optional[str] = None,
        extra_arg: Any = None,
        save_restart: bool = False,
        load_restart: bool = False,
        restart_file: str = 'opt_restart.pkl',
        work_dir: str = 'minimize',
        job_name: str = 'minimize',
        cluster: Cluster = 'slurm',
        submitit_kwargs: Optional[Dict] = None,
        budget: int = int(1e6),
        job_fail_value: float = np.nan,
        submit_retry_max_attempts: int = 5,
        submit_retry_wait_seconds: float = 10,
        submit_delay_seconds: float = 0,
        check_output_interval_seconds: float = 1,
        check_output_timeout_minutes: float = int(1e5),
        restart_save_interval: int = 1,
        install_signal_handler: bool = True,
        # remote run params:
        remote: bool = False,
        remote_cluster: Literal['slurm', 'local'] = 'slurm',
        remote_submitit_kwargs: Optional[Dict] = None,
) -> Union[Dict, submitit.Job]:
    """
    Combine submitit + nevergrad + botorch to allow parallel optimization on slurm.
    has capability to keep drawing points using optimizer.ask() until num_workers points are found, that were not
    already calculated previously, and that pass constraint_fun. This prevents wasting compute on irrelevant points.
    Default optimizer is nevergrad's implementation for DifferentialEvolution.

    This is a thin convenience wrapper around MinimizeConfig + Minimizer; use those
    directly for programmatic access to the run's configuration and state.

    Parameters
    ----------
    install_signal_handler : bool
        If True (default), install a SIGTERM handler that exits with code 1
        so Slurm marks the job as FAILED rather than COMPLETED on scancel.
        Set to False if you are embedding this function in a larger application
        that manages its own signal handling.
    remote : bool
        If True, submit the whole optimization loop as its own job on
        remote_cluster and return the submitit Job handle immediately
        (job.result() gives the status dict). If False (default), run the loop
        in this process and return the status dict.
    """
    config = MinimizeConfig(
        loss_fun=loss_fun, param_bounds=param_bounds, num_workers=num_workers, num_iters=num_iters,
        optimizer_package=optimizer_package, optimizer_class=optimizer_class, botorch_kwargs=botorch_kwargs,
        init_points=init_points, constraint_fun=constraint_fun, num_asks_max=num_asks_max,
        verbosity=verbosity, slurm_verbosity=slurm_verbosity, log_file=log_file, extra_arg=extra_arg,
        save_restart=save_restart, load_restart=load_restart, restart_file=restart_file,
        work_dir=work_dir, job_name=job_name, cluster=cluster, submitit_kwargs=submitit_kwargs,
        budget=budget, job_fail_value=job_fail_value,
        submit_retry_max_attempts=submit_retry_max_attempts,
        submit_retry_wait_seconds=submit_retry_wait_seconds,
        submit_delay_seconds=submit_delay_seconds,
        check_output_interval_seconds=check_output_interval_seconds,
        check_output_timeout_minutes=check_output_timeout_minutes,
        restart_save_interval=restart_save_interval,
        install_signal_handler=install_signal_handler,
        remote=remote, remote_cluster=remote_cluster, remote_submitit_kwargs=remote_submitit_kwargs,
    )

    if config.remote:
        set_logging(config.work_dir, config.log_file)
        if config.install_signal_handler:
            signal.signal(signal.SIGTERM, signal_handler)
        print('Running slurm_minimize remotely.')
        return submit_remote_run(run_minimize, dataclasses.replace(config, remote=False),
                                 config.work_dir, config.job_name,
                                 config.remote_cluster, config.remote_submitit_kwargs)

    return Minimizer(config).run()
