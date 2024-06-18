import os
import numpy as np
import submitit

def add_to_history(x_history, x):
    """
    Add a new value to the history array
    """
    if len(x_history) == 0:
        x_history = np.array(x)
    else:
        if len(x_history.shape) == 1 and len(x.shape) == 1:
            x_history = np.hstack([x_history, np.array(x)])
        else:
            x_history = np.vstack([x_history, np.array(x)])
    return x_history

class SlurmPool():
    """
    A class defined for the sole purpose of consistency with the pool.map syntax for parallelization in the emcee
    package. Instead of using multiprocessing, we integrate with the submitit package to perform parallel calculations
    on a cluster of multiple processors.

    cluster: 'slurm' or 'local' (run locally with submitit) or 'local-map' (run locally with map)

    TODO: its possible to save progress in nevergrad and emcee, so no real need to save progress manually
     test it with cases of NaNs etc and then kill this.
    """
    def __init__(self, work_dir, job_name='tmp', cluster='slurm', **job_params):
        self.num_calls = 0
        self.points_history = []
        self.values_history = []
        self.failed_points_history = []
        self.work_dir = work_dir
        self.job_name = job_name
        self.job_params = job_params
        self.cluster = cluster
        return

    def map(self, fun, points):
        if self.cluster == 'local-map':
            res = [fun(point) for point in points]
        else:
            res = self.send_and_receive_jobs(fun, points)

        # update history arrays
        self.num_calls += 1
        inds_failed = [i for i, r in enumerate(res) if r == None or np.isnan(r)]
        inds_success = [i for i, r in enumerate(res) if i not in inds_failed]
        failed_points = np.array([p for i, p in enumerate(points) if i in inds_failed])
        success_points = np.array([p for i, p in enumerate(points) if i in inds_success])
        success_values = np.array([v for i, v in enumerate(res) if i in inds_success])
        if len(inds_failed) > 0:
            self.failed_points_history = add_to_history(self.failed_points_history, failed_points)
        # print('success_points:', success_points)
        # print('success_values:', success_values)
        if len(inds_success) > 0:
            self.points_history = add_to_history(self.points_history, success_points)
            self.values_history = add_to_history(self.values_history, success_values)

        return res

    def send_and_receive_jobs(self, fun, points):
        """
        In case of a remote function evaluation, the function would be re-spawned in a new process
        so there is no need to directly take it as an argument.
        """

        # prepare directories and input files for the jobs and send them
        iteration_dir = self.work_dir + '/' + str(self.num_calls)
        os.makedirs(iteration_dir, exist_ok=True)
        point_dirs = []
        for ind_point, point in enumerate(points):
            point_dir = iteration_dir + '/' + str(ind_point)
            point_dirs += [point_dir]
            os.makedirs(point_dir, exist_ok=True)
            # print('$$$$$$$$$', 'point:', point)
            np.savetxt(point_dir + '/input.txt', [point])

        # send the jobs
        jobs = []
        for ind_point, (point, point_dir) in enumerate(zip(points, point_dirs)):
            job_name = self.job_name + '_' + str(self.num_calls) + '_' + str(ind_point)
            self.executor = submitit.AutoExecutor(folder=point_dir, cluster=self.cluster)
            self.executor.update_parameters(slurm_job_name=job_name, **self.job_params)
            job = self.executor.submit(fun, point)
            jobs += [job]

        # collect the results
        outputs = []
        for ind_point, job in enumerate(jobs):
            output = job.result()
            point_dir = iteration_dir + '/' + str(ind_point)
            np.savetxt(point_dir + '/output.txt', [output])
            outputs += [output]

        # save current iteration points and results
        np.savetxt(iteration_dir + '/inputs.txt', np.array(points))
        np.savetxt(iteration_dir + '/outputs.txt', np.array(outputs))

        return outputs
