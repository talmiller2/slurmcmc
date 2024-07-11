import numpy as np
import nevergrad as ng
from slurmcmc.general_utils import print_log, save_restart_file, load_restart_file
from slurmcmc.slurm_utils import SlurmPool


def slurm_minimize(loss_fun, param_bounds, optimizer_class=None, num_workers=1, num_iters=10,
                   init_points=None, constraint_fun=None, num_asks_max=int(1e3),
                   verbosity=1, slurm_vebosity=0, log_file=None,
                   save_restart=False, load_restart=False, restart_file='opt_restart.pkl',
                   work_dir='tmp', job_name='minimize', cluster='slurm', slurm_dict={}):
    """
    combine submitit + nevergrad to allow parallel optimization on slurm.
    has capability to keep drawing points from optimizer.ask() until num_workers points are found that were not
    already calculated previously, and that pass constraint_fun.
    """

    if load_restart:
        if verbosity >= 1:
            print_log('loading restart file.', work_dir, log_file)

            status = load_restart_file(work_dir, restart_file)
            optimizer = status['optimizer']
            loss_min_iters = status['loss_min_iters']
            loss_per_iters = status['loss_per_iters']
            slurm_pool = status['slurm_pool']
            num_loss_fun_calls_total = status['num_loss_fun_calls_total']
            num_constraint_fun_calls_total = status['num_constraint_fun_calls_total']
            num_asks_total = status['num_asks_total']
            evaluated_points = status['evaluated_points']

    else:
        # param_bounds is a list (length num_params) that contains the lower and upper bounds per parameter
        lower_bounds = [b[0] for b in param_bounds]
        upper_bounds = [b[1] for b in param_bounds]
        num_params = len(lower_bounds)
        instrum = ng.p.Instrumentation(ng.p.Array(shape=(num_params,))
                                       .set_bounds(lower=lower_bounds, upper=upper_bounds))

        if optimizer_class is None:
            optimizer_class = ng.optimizers.DifferentialEvolution(crossover="twopoints", popsize=num_workers)

        budget = num_iters * num_workers
        optimizer = optimizer_class(parametrization=instrum, budget=budget, num_workers=num_workers)

        slurm_pool = SlurmPool(work_dir, job_name, cluster, verbosity=slurm_vebosity, log_file=log_file, **slurm_dict)

        num_loss_fun_calls_total = 0
        num_constraint_fun_calls_total = 0
        num_asks_total = 0
        evaluated_points = set()
        loss_min_iters = []
        loss_per_iters = []

    ## start optimization iterations
    for curr_iter in range(num_iters):
        if verbosity >= 2:
            print_log('### curr opt iter: ' + str(curr_iter), work_dir, log_file)

        if curr_iter == 0 and init_points is not None:
            candidates = []
            for init_point in init_points:
                candidate = instrum.spawn_child()
                candidate.value = ((init_point,), {})
                candidates += [candidate]

            if constraint_fun is not None:
                for ind_candidate, candidate in enumerate(candidates):
                    constraint_passed = constraint_fun(*candidate.args, **candidate.kwargs) <= 0
                    num_constraint_fun_calls_total += 1
                    if not constraint_passed:
                        raise ValueError('init point index ' + str(ind_candidate) + ' does not satisfy constraint.')

        else:
            candidates = []
            num_asks = 0
            while len(candidates) < num_workers:
                candidate = optimizer.ask()
                num_asks += 1
                if num_asks > num_asks_max:
                    raise ValueError('num_asks exceeded num_asks_max, having trouble finding satisfactory candidates.')
                candidate_tuple = tuple(candidate.value[0][0])
                if candidate_tuple not in evaluated_points:
                    if constraint_fun is not None:
                        constraint_passed = constraint_fun(*candidate.args, **candidate.kwargs) <= 0
                        num_constraint_fun_calls_total += 1

                    if constraint_fun is not None and not constraint_passed:
                        pass
                    else:
                        evaluated_points.add(candidate_tuple)
                        candidates += [candidate]

            if verbosity >= 3:
                print_log('optimizer.ask was called ' + str(num_asks) + ' times', work_dir, log_file)

            num_asks_total += num_asks

        points = [c.value[0][0] for c in candidates]
        results = slurm_pool.map(loss_fun, points)
        num_loss_fun_calls_total += len(points)

        for candidate, result in zip(candidates, results):
            optimizer.tell(candidate, result)

        x_min = optimizer.current_bests['minimum'].parameter.value[0][0]
        loss_min = optimizer.current_bests['minimum'].mean
        loss_min_iters += [loss_min]
        loss_per_iters += [np.nanmin(results)]

        if verbosity >= 2:
            print_log('curr best: x_min: ' + str(x_min) + ', loss_min: ' + str(loss_min), work_dir, log_file)

        # optimization status
        status = {}
        status['optimizer'] = optimizer
        status['x_min'] = x_min
        status['loss_min'] = loss_min
        status['loss_min_iters'] = loss_min_iters
        status['loss_per_iters'] = loss_per_iters
        status['slurm_pool'] = slurm_pool
        status['num_loss_fun_calls_total'] = num_loss_fun_calls_total
        status['num_constraint_fun_calls_total'] = num_constraint_fun_calls_total
        status['num_asks_total'] = num_asks_total
        status['evaluated_points'] = evaluated_points

        if save_restart:
            save_restart_file(status, work_dir, restart_file)

    x_min = optimizer.current_bests['minimum'].parameter.value[0][0]
    loss_min = optimizer.current_bests['minimum'].mean
    if verbosity >= 1:
        print_log('### opt loop done. x_min: ' + str(x_min) + ', loss_min: ' + str(loss_min), work_dir, log_file)

    return status
