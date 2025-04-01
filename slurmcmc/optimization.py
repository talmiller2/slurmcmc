import functools
import logging
import time

import nevergrad as ng
import numpy as np
import submitit

from slurmcmc.botorch_optimizer import BoTorchOptimizer
from slurmcmc.general_utils import set_logging, save_restart_file, load_restart_file, combine_args, point_to_tuple
from slurmcmc.import_utils import deferred_import_function_wrapper
from slurmcmc.slurm_utils import SlurmPool


def slurm_minimize(loss_fun, param_bounds, num_workers=1, num_iters=10,
                   optimizer_package='nevergrad', optimizer_class=None, botorch_kwargs=None,
                   init_points=None, constraint_fun=None, num_asks_max=int(1e3),
                   verbosity=1, slurm_vebosity=0, log_file=None, extra_arg=None,
                   save_restart=False, load_restart=False, restart_file='opt_restart.pkl',
                   work_dir='minimize', job_name='minimize', cluster='slurm', submitit_kwargs=None,
                   # remote run params:
                   remote=False, remote_cluster='slurm', remote_submitit_kwargs=None,
                   ):
    """
    combine submitit + nevergrad + botorch to allow parallel optimization on slurm.
    has capability to keep drawing points using optimizer.ask() until num_workers points are found, that were not
    already calculated previously, and that pass constraint_fun. This prevents wasting compute on irrelevant points.
    """
    set_logging(work_dir, log_file)

    if remote == True:
        print('Running slurm_minimize remotely.')
        if remote_submitit_kwargs is None:
            remote_submitit_kwargs = {}
        if 'slurm_job_name' not in remote_submitit_kwargs:
            remote_submitit_kwargs['slurm_job_name'] = 'main_' + job_name
        if 'timeout_min' not in remote_submitit_kwargs:
            remote_submitit_kwargs['timeout_min'] = int(60 * 24 * 30) # 1 month
        kwargs = locals()
        kwargs['remote'] = False
        executor = submitit.AutoExecutor(folder=work_dir, cluster=remote_cluster)
        executor.update_parameters(**remote_submitit_kwargs)
        job = executor.submit(functools.partial(slurm_minimize, **kwargs))
        return job

    else:

        if submitit_kwargs is None:
            submitit_kwargs = {}
        if botorch_kwargs is None:
            botorch_kwargs = {}

        if constraint_fun is not None:
            constraint_fun = deferred_import_function_wrapper(constraint_fun)

        if load_restart:
            if verbosity >= 1:
                logging.info('loading restart file: ' + work_dir + '/' + restart_file)

                status = load_restart_file(work_dir, restart_file)
                optimizer = status['optimizer']
                optimizer_package = status['optimizer_package']
                x_min = status['x_min']
                loss_min_per_iter = status['loss_min_per_iter']
                loss_min_all_iter = status['loss_min_all_iter']
                num_workers_per_iter = status['num_workers_per_iter']
                loss_min = status['loss_min']
                loc_point_min_per_iter = status['loc_point_min_per_iter']
                loc_point_min_all_iter = status['loc_point_min_all_iter']
                slurm_pool = status['slurm_pool']
                ini_iter = status['ini_iter']
                num_loss_fun_calls_total = status['num_loss_fun_calls_total']
                num_constraint_fun_calls_total = status['num_constraint_fun_calls_total']
                num_asks_total = status['num_asks_total']
                candidates_ask_time_per_iter = status['candidates_ask_time_per_iter']

        else:
            # param_bounds is a list (length num_params) that contains the lower and upper bounds per parameter
            lower_bounds = [b[0] for b in param_bounds]
            upper_bounds = [b[1] for b in param_bounds]
            budget = num_iters * num_workers

            if optimizer_package == 'nevergrad':
                instrum = ng.p.Instrumentation(
                    ng.p.Array(init=[(l + u) / 2 for l, u in zip(lower_bounds, upper_bounds)])
                    .set_bounds(lower=lower_bounds, upper=upper_bounds))
                if optimizer_class is None:
                    optimizer_class = ng.optimizers.DifferentialEvolution(crossover="twopoints", popsize=num_workers)
                optimizer = optimizer_class(parametrization=instrum, budget=budget, num_workers=num_workers)
            elif optimizer_package == 'botorch':
                if optimizer_class is None:
                    optimizer_class = BoTorchOptimizer

                botorch_defaults = {'num_restarts': 10, 'raw_samples': 100, 'num_best_points': None}
                for key, value in botorch_defaults.items():
                    if key not in botorch_kwargs:
                        botorch_kwargs[key] = value

                optimizer = optimizer_class(lower_bounds, upper_bounds, num_workers, **botorch_kwargs)

            else:
                raise ValueError('invalid optimizer_package:', optimizer_package)

            slurm_pool = SlurmPool(work_dir, job_name, cluster, verbosity=slurm_vebosity, log_file=log_file,
                                   extra_arg=extra_arg, submitit_kwargs=submitit_kwargs)
            ini_iter = 0
            num_loss_fun_calls_total = 0
            num_constraint_fun_calls_total = 0
            num_asks_total = 0
            loss_min_per_iter = []
            loss_min_all_iter = []
            num_workers_per_iter = []
            loss_min = np.inf
            loc_point_min_per_iter = []
            candidates_ask_time_per_iter = []

        ## start optimization iterations
        for curr_iter in range(ini_iter, ini_iter + num_iters):
            if verbosity >= 1:
                logging.info('### curr opt iter: ' + str(curr_iter))

            # ask for points for current iteration
            t_start_ask_curr_iter = time.time()
            if curr_iter == 0 and init_points is not None:
                candidates = init_points
                candidates_nevergrad = []
                # construct candidates in the nevergrad format
                for init_point in candidates_nevergrad:
                    candidate_nevergrad = instrum.spawn_child()
                    candidate_nevergrad.value = ((init_point,), {})
                    candidates_nevergrad += [candidate_nevergrad]

                # check init_points satisfy the constraint_fun
                if constraint_fun is not None:
                    for ind_candidate, candidate in enumerate(candidates):
                        constraint_passed = constraint_fun(*combine_args(candidate, extra_arg)) <= 0
                        num_constraint_fun_calls_total += 1
                        if not constraint_passed:
                            raise ValueError('init point index ' + str(ind_candidate) + ' does not satisfy constraint.')

            else:
                candidates = []
                candidates_nevergrad = []
                candidates_set = set()
                num_asks = 0
                while len(candidates) < num_workers:
                    if optimizer_package == 'nevergrad':
                        candidate_nevergrad = optimizer.ask()
                        candidate = candidate_nevergrad.value[0][0]
                        candidates_batch = [candidate]
                    elif optimizer_package == 'botorch':
                        x_pts = slurm_pool.points_history
                        y_pts = slurm_pool.values_history
                        t_start_ask = time.time()
                        candidates_batch = optimizer.ask(x_pts, y_pts)
                        t_end_ask = time.time()
                        if verbosity >= 3:
                            logging.info('    botorch ask run time: ' + '{:.1f}'.format(t_end_ask - t_start_ask) + 's.')

                    num_asks += 1
                    if num_asks > num_asks_max:
                        raise ValueError('num_asks exceeded num_asks_max=', num_asks_max,
                                         ', having trouble finding candidates that pass constraints.')

                    for candidate in candidates_batch:
                        candidate_tuple = point_to_tuple(candidate)
                        if (candidate_tuple not in slurm_pool.evaluated_points_set
                                and candidate_tuple not in candidates_set):
                            if constraint_fun is not None:
                                constraint_passed = constraint_fun(*combine_args(candidate, extra_arg)) <= 0
                                num_constraint_fun_calls_total += 1

                            if constraint_fun is not None and not constraint_passed:
                                pass
                            else:
                                candidates_set.add(candidate_tuple)
                                candidates += [candidate]
                                if optimizer_package == 'nevergrad':
                                    candidates_nevergrad += [candidate_nevergrad]
                                if len(candidates) == num_workers:
                                    break

                if verbosity >= 3:
                    logging.info('    optimizer.ask was called ' + str(num_asks) + ' times.')
                num_asks_total += num_asks

                t_end_ask_curr_iter = time.time()
                candidates_ask_time_per_iter += [t_end_ask_curr_iter - t_start_ask_curr_iter]

            # calculate loss_fun on current iteration candidates
            results = slurm_pool.map(loss_fun, candidates)
            num_loss_fun_calls_total += len(candidates)

            # inform the optimizer with the new data
            if optimizer_package == 'nevergrad':
                for candidate_nevergrad, result in zip(candidates_nevergrad, results):
                    optimizer.tell(candidate_nevergrad, result)
            elif optimizer_package == 'botorch':
                # the data is already contained in slurm_pool
                pass

            # evaluate optimization metrics post current iteration
            ind_curr_iter_min = np.nanargmin(results)
            curr_iter_x_min, curr_iter_loss_min = candidates[ind_curr_iter_min], results[ind_curr_iter_min]
            curr_iter_loc_point_min = (curr_iter, ind_curr_iter_min)
            if curr_iter_loss_min < loss_min:
                loss_min = curr_iter_loss_min
                x_min = curr_iter_x_min
                loc_point_min_all_iter = curr_iter_loc_point_min
            loss_min_all_iter += [loss_min]
            loss_min_per_iter += [curr_iter_loss_min]
            num_workers_per_iter += [num_workers]
            loc_point_min_per_iter += [curr_iter_loc_point_min]

            if verbosity >= 2:
                logging.info('    curr best: x_min: ' + str(x_min) + ', loss_min: ' + str(loss_min))

            # optimization status
            status = {}
            status['optimizer'] = optimizer
            status['optimizer_package'] = optimizer_package
            status['x_min'] = x_min
            status['loss_min_per_iter'] = loss_min_per_iter
            status['loss_min_all_iter'] = loss_min_all_iter
            status['num_workers_per_iter'] = num_workers_per_iter
            status['loss_min'] = loss_min
            status['loc_point_min_per_iter'] = loc_point_min_per_iter
            status['loc_point_min_all_iter'] = loc_point_min_all_iter
            status['slurm_pool'] = slurm_pool
            status['ini_iter'] = curr_iter + 1
            status['num_loss_fun_calls_total'] = num_loss_fun_calls_total
            status['num_constraint_fun_calls_total'] = num_constraint_fun_calls_total
            status['num_asks_total'] = num_asks_total
            status['candidates_ask_time_per_iter'] = candidates_ask_time_per_iter

            if save_restart:
                if verbosity >= 3:
                    logging.info('    saving restart file: ' + work_dir + '/' + restart_file)

                save_restart_file(status, work_dir, restart_file)

        if verbosity >= 1:
            logging.info('### opt loop done. x_min: ' + str(x_min) + ', loss_min: ' + str(loss_min))

        return status
