import json
import os

import numpy as np
import submitit

from slurmcmc.general_utils import print_log


class SlurmPool():
    """
    A class defined for the sole purpose of consistency with the pool.map syntax for parallelization in the emcee
    package. Instead of using multiprocessing, we integrate with the submitit package to perform parallel calculations
    on a cluster of multiple processors.

    cluster: 'slurm' or 'local' (run locally with submitit) or 'local-map' (run locally with map)
    """

    def __init__(self, work_dir, job_name='tmp', cluster='slurm', budget=int(1e6), verbosity=1, log_file=None,
                 extra_arg=None, **job_params):
        self.num_calls = 0
        self.points_history = []
        self.values_history = []
        self.failed_points_history = []
        self.work_dir = work_dir
        self.job_name = job_name
        self.job_params = job_params
        self.cluster = cluster
        self.verbosity = verbosity
        self.budget = budget
        self.log_file = log_file

        if cluster in ['local', 'slurm'] and os.path.isdir(work_dir + '/0'):
            error_msg = 'work_dir appears to already contain runs, move or delete it first.'
            error_msg += '\n' + 'work_dir:' + work_dir
            raise ValueError(error_msg)
            # in case of continuing from restart, SlurmPool is loaded and not initialized so will not error.

        # capability to call the function with additional argument that is constant during the map
        self.extra_arg = extra_arg

        return

    def map(self, fun, points):
        # split points into chunks if exceeding budget
        chunks = self.split_points(points, self.budget)
        chunk_sizes = [len(chunk) for chunk in chunks]
        if self.verbosity >= 1 and len(chunks) > 1:
            print_log('split points into ' + str(len(chunks)) + ' chunks of sizes ' + str(chunk_sizes) + '.',
                      self.work_dir, self.log_file)

        res = []
        for chunk in chunks:
            res += self.map_chunk(fun, chunk)
        return res

    def split_points(self, points, budget):
        # calculate the number of chunks
        num_chunks = len(points) // budget
        if len(points) % budget != 0:
            num_chunks += 1

        # split points into chunks
        chunks = [points[i * budget:(i + 1) * budget] for i in range(num_chunks)]
        return chunks

    def get_fun_args(self, point):
        args = [point]
        if self.extra_arg is not None:
            args += [self.extra_arg]
        return args

    def map_chunk(self, fun, points):
        if self.verbosity >= 1:
            print_log('slurm_pool.map called with ' + str(len(points)) + ' points.', self.work_dir, self.log_file)

        if self.cluster == 'local-map':
            res = [fun(*self.get_fun_args(point)) for point in points]
        else:
            res = self.send_and_receive_jobs(fun, points)

        self.num_calls += 1

        # number of parameters (dimension of each point)
        dim_input = self.calc_dimension(points)
        dim_output = self.calc_dimension(res)

        # update history arrays
        inds_failed = [i for i, r in enumerate(res) if self.check_failed(r)]
        inds_success = [i for i, r in enumerate(res) if i not in inds_failed]
        failed_points = np.array([p for i, p in enumerate(points) if i in inds_failed])
        success_points = np.array([p for i, p in enumerate(points) if i in inds_success])
        success_values = np.array([v for i, v in enumerate(res) if i in inds_success])
        success_values = success_values.reshape(-1, 1)  # switch to column array
        if len(inds_failed) > 0:
            self.failed_points_history = self.add_to_history(self.failed_points_history, failed_points, dim=dim_input)
        if len(inds_success) > 0:
            self.points_history = self.add_to_history(self.points_history, success_points, dim=dim_input)
            self.values_history = self.add_to_history(self.values_history, success_values, dim=dim_output)

        return res

    def calc_dimension(self, points):
        if np.array(points[0]).shape == ():
            dim = 1
        else:
            dim = np.array(points[0]).shape[0]
        return dim

    def check_failed(self, r):
        if r == None:
            return True
        elif type(r) in [np.ndarray, float]:
            return np.all(np.isnan(r))
        elif type(r) == list:
            for e in r:
                if e == None or np.isnan(e):
                    return True
        return False

    def add_to_history(self, x_history, x, dim):
        """
        Add a new value to the history array
        """
        x = np.array(x)
        if dim == 1:
            x = x.reshape(-1, 1)
        if len(x_history) == 0:
            x_history = x
        else:
            x_history = np.vstack([x_history, x])
        return x_history

    def send_and_receive_jobs(self, fun, points):
        ini_dir = os.getcwd()

        # prepare directories and input files for the jobs and send them
        iteration_dir = self.work_dir + '/' + str(self.num_calls)
        os.makedirs(iteration_dir, exist_ok=True)
        point_dirs = []
        for ind_point, point in enumerate(points):
            point_dir = iteration_dir + '/' + str(ind_point)
            point_dirs += [point_dir]
            os.makedirs(point_dir, exist_ok=True)
            np.savetxt(point_dir + '/input.txt', [point])

        # save current iteration points in the main iteration_dir
        np.savetxt(iteration_dir + '/inputs.txt', np.array(points))
        if self.extra_arg is not None:
            # save the extra_arg in the run folder to document the full input used for this point
            extra_arg_file = iteration_dir + '/extra_arg.txt'
            with open(extra_arg_file, 'w') as json_file:
                json.dump(self.extra_arg, json_file)

        # send the jobs
        jobs = []
        for ind_point, (point, point_dir) in enumerate(zip(points, point_dirs)):
            job_name = self.job_name + '_' + str(self.num_calls) + '_' + str(ind_point)
            self.executor = submitit.AutoExecutor(folder=point_dir, cluster=self.cluster)
            self.executor.update_parameters(slurm_job_name=job_name, **self.job_params)
            os.chdir(point_dir) # each point evaluation (query) is born in its own dir
            job = self.executor.submit(fun, *self.get_fun_args(point))
            jobs += [job]

        # collect the results
        outputs = []
        for ind_point, job in enumerate(jobs):
            output = job.result()
            point_dir = iteration_dir + '/' + str(ind_point)
            np.savetxt(point_dir + '/output.txt', [output])
            outputs += [output]

        # save current iteration results
        np.savetxt(iteration_dir + '/outputs.txt', np.array(outputs))

        os.chdir(ini_dir)  # return to initial dir
        return outputs
