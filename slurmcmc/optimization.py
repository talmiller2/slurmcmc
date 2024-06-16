import nevergrad as ng

from slurmcmc.slurm_utils import SlurmPool


def slurm_minimize(loss_fun, param_bounds, optimizer_class=None, num_workers=1, num_iters=10,
                   init_points=None, constraint_fun=None, verbosity=1, num_asks_max=int(1e3),
                   work_dir='tmp', job_name='minimize', cluster='local', **job_params):
    """
    combine submitit + nevergrad to allow parallel optimization on slurm.
    """

    # param_bounds is a list (length num_params) that contains the lower and upper bounds per parameter
    lower_bounds = [b[0] for b in param_bounds]
    upper_bounds = [b[1] for b in param_bounds]
    num_params = len(lower_bounds)
    instrum = ng.p.Instrumentation(ng.p.Array(shape=(num_params,))
                                   .set_bounds(lower=lower_bounds, upper=upper_bounds))

    if optimizer_class is None:
        optimizer_class = ng.optimizers.DifferentialEvolution(crossover="twopoints", popsize=num_workers)
        # optimizer_class = ng.optimizers.ConfPSO(popsize=num_workers)

    budget = num_iters * num_workers
    optimizer = optimizer_class(parametrization=instrum, budget=budget, num_workers=num_workers)

    slurm_pool = SlurmPool(work_dir, job_name, cluster, **job_params)

    num_loss_fun_calls_total = 0
    num_constraint_fun_calls_total = 0
    num_asks_total = 0
    evaluated_points = set()

    # TODO: allow to restart from file

    ## start optimization iterations
    for curr_iter in range(num_iters):
        if verbosity >= 1:
            print('### curr opt iter:', curr_iter)

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
            num_constraint_fun_calls_curr = 0
            while len(candidates) < num_workers:
                candidate = optimizer.ask()
                num_asks += 1
                if num_asks > num_asks_max:
                    raise ValueError('num_asks exceeded num_asks_max, having trouble finding satisfactory candidates.')
                candidate_tuple = tuple(candidate.value[0][0])
                if candidate_tuple not in evaluated_points:
                    if constraint_fun is not None:
                        constraint_passed = constraint_fun(*candidate.args, **candidate.kwargs) <= 0
                        num_constraint_fun_calls_curr += 1
                        num_constraint_fun_calls_total += 1
                    if constraint_fun is not None and not constraint_passed:
                        pass
                    else:
                        evaluated_points.add(candidate_tuple)
                        candidates += [candidate]

            if verbosity >= 3:
                print('    optimizer.ask() was called:', num_asks, 'times')
            num_asks_total += num_asks

        points = [c.value[0][0] for c in candidates]
        results = slurm_pool.map(loss_fun, points)
        num_loss_fun_calls_total += len(points)

        for candidate, result in zip(candidates, results):
            optimizer.tell(candidate, result)

        if verbosity >= 2:
            x_min = optimizer.current_bests['minimum'].parameter.value[0][0]
            loss_min = optimizer.current_bests['minimum'].mean
            print('    curr best: x_min:', x_min, ', loss_min:', loss_min)

    x_min = optimizer.current_bests['minimum'].parameter.value[0][0]
    loss_min = optimizer.current_bests['minimum'].mean
    if verbosity >= 1:
        print('### opt loop done. x_min:', x_min, ', loss_min:', loss_min)

    history = [(point, info.mean) for point, info in optimizer.archive.items_as_arrays()]

    # return optimization results
    result_dict = {}
    result_dict['x_min'] = x_min
    result_dict['loss_min'] = loss_min
    result_dict['history'] = history
    result_dict['slurm_pool'] = slurm_pool
    result_dict['num_loss_fun_calls_total'] = num_loss_fun_calls_total
    result_dict['num_constraint_fun_calls_total'] = num_constraint_fun_calls_total
    result_dict['num_asks_total'] = num_asks_total
    result_dict['evaluated_points'] = evaluated_points

    return result_dict
