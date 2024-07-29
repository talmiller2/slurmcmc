import logging
import time

import nevergrad as ng
import numpy as np

from slurmcmc.botorch_optimizer import BoTorchOptimizer
from slurmcmc.general_utils import set_logging, save_restart_file, load_restart_file, combine_args
from slurmcmc.slurm_utils import SlurmPool


def slurm_minimize(loss_fun, param_bounds, num_workers=1, num_iters=10,
                   optimizer_package='nevergrad', optimizer_class=None, botorch_kwargs={},
                   init_points=None, constraint_fun=None, num_asks_max=int(1e3),
                   verbosity=1, slurm_vebosity=0, log_file=None, extra_arg=None,
                   save_restart=False, load_restart=False, restart_file='opt_restart.pkl',
                   work_dir='tmp', job_name='minimize', cluster='slurm', slurm_dict={}):
    """
    combine submitit + nevergrad + botorch to allow parallel optimization on slurm.
    has capability to keep drawing points using optimizer.ask() until num_workers points are found, that were not
    already calculated previously, and that pass constraint_fun. This prevents wasting compute on irrelevant points.
    """
    set_logging(work_dir, log_file)

    if load_restart:
        if verbosity >= 1:
            logging.info('loading restart file: ' + work_dir + '/' + restart_file)

            status = load_restart_file(work_dir, restart_file)
            optimizer = status['optimizer']
            x_min = status['x_min']
            loss_min = status['loss_min']
            loss_min_iters = status['loss_min_iters']
            loss_per_iters = status['loss_per_iters']
            slurm_pool = status['slurm_pool']
            ini_iter = status['ini_iter']
            num_loss_fun_calls_total = status['num_loss_fun_calls_total']
            num_constraint_fun_calls_total = status['num_constraint_fun_calls_total']
            num_asks_total = status['num_asks_total']
            evaluated_points = status['evaluated_points']

    else:
        # param_bounds is a list (length num_params) that contains the lower and upper bounds per parameter
        lower_bounds = [b[0] for b in param_bounds]
        upper_bounds = [b[1] for b in param_bounds]
        num_params = len(lower_bounds)
        budget = num_iters * num_workers

        if optimizer_package == 'nevergrad':
            instrum = ng.p.Instrumentation(ng.p.Array(shape=(num_params,))
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
                               extra_arg=extra_arg, **slurm_dict)
        ini_iter = 0
        num_loss_fun_calls_total = 0
        num_constraint_fun_calls_total = 0
        num_asks_total = 0
        evaluated_points = set()
        loss_min = np.inf
        loss_min_iters = []
        loss_per_iters = []

    ## start optimization iterations
    for curr_iter in range(ini_iter, ini_iter + num_iters):
        if verbosity >= 1:
            logging.info('### curr opt iter: ' + str(curr_iter))

        # ask for points for current iteration
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
                    candidate_tuple = tuple(candidate)
                    if candidate_tuple not in evaluated_points:
                        if constraint_fun is not None:
                            constraint_passed = constraint_fun(*combine_args(candidate, extra_arg)) <= 0
                            num_constraint_fun_calls_total += 1

                        if constraint_fun is not None and not constraint_passed:
                            pass
                        else:
                            evaluated_points.add(candidate_tuple)
                            candidates += [candidate]
                            if optimizer_package == 'nevergrad':
                                candidates_nevergrad += [candidate_nevergrad]
                            if len(candidates) == num_workers:
                                break

            if verbosity >= 3:
                logging.info('    optimizer.ask was called ' + str(num_asks) + ' times.')
            num_asks_total += num_asks

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
        ind_curr_min = np.nanargmin(results)
        curr_x_min, curr_loss_min = candidates[ind_curr_min], results[ind_curr_min]
        if curr_loss_min < loss_min:
            loss_min = curr_loss_min
            x_min = curr_x_min
        loss_min_iters += [loss_min]
        loss_per_iters += [curr_loss_min]

        if verbosity >= 2:
            logging.info('    curr best: x_min: ' + str(x_min) + ', loss_min: ' + str(loss_min))

        # optimization status
        status = {}
        status['optimizer'] = optimizer
        status['x_min'] = x_min
        status['loss_min'] = loss_min
        status['loss_min_iters'] = loss_min_iters
        status['loss_per_iters'] = loss_per_iters
        status['slurm_pool'] = slurm_pool
        status['ini_iter'] = curr_iter + 1
        status['num_loss_fun_calls_total'] = num_loss_fun_calls_total
        status['num_constraint_fun_calls_total'] = num_constraint_fun_calls_total
        status['num_asks_total'] = num_asks_total
        status['evaluated_points'] = evaluated_points

        if save_restart:
            if verbosity >= 3:
                logging.info('    saving restart file: ' + work_dir + '/' + restart_file)

            save_restart_file(status, work_dir, restart_file)

    if verbosity >= 1:
        logging.info('### opt loop done. x_min: ' + str(x_min) + ', loss_min: ' + str(loss_min))

    return status
