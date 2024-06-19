import os
import numpy as np
import submitit

class SlurmPool():
    """
    A class defined for the sole purpose of consistency with the pool.map syntax for parallelization in the emcee
    package. Instead of using multiprocessing, we integrate with the submitit package to perform parallel calculations
    on a cluster of multiple processors.

    cluster: 'slurm' or 'local' (run locally with submitit) or 'local-map' (run locally with map)

    TODO: print to log_file

    """
    def __init__(self, work_dir, job_name='tmp', cluster='slurm', budget=int(1e6), verbosity=1, **job_params):
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
        return

    # TODO: add a budget option, if more points are asked than budget, need to split to several
    def map(self, fun, points):
        # split points into chunks if exceeding budget
        chunks = self.split_points(points, self.budget)
        chunk_sizes = [len(chunk) for chunk in chunks]
        if self.verbosity >= 1 and len(chunks) > 1:
            print('split points into ' + str(len(chunks)) + ' chunks of sizes ' + str(chunk_sizes) + '.')

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

    # # Example usage
    # # points = [[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 9]]
    # points = [[1, 2], [2, 3], [3, 4]]
    # budget = 3
    # chunks = split_points(points, budget)
    # for idx, chunk in enumerate(chunks):
    #     print(f"Chunk {idx + 1}: {chunk}")

    def map_chunk(self, fun, points):
        if self.verbosity >= 1:
            print('slurm_pool.map() called with ' + str(len(points)) + ' points.')

        if self.cluster == 'local-map':
            res = [fun(point) for point in points]
        else:
            res = self.send_and_receive_jobs(fun, points)

        self.num_calls += 1

        # number of parameters (dimension of each point)
        if np.array(points[0]).shape == ():
            num_params =  1
        else:
            num_params = np.array(points[0]).shape[0]

        # update history arrays
        inds_failed = [i for i, r in enumerate(res) if r == None or np.isnan(r)]
        inds_success = [i for i, r in enumerate(res) if i not in inds_failed]
        failed_points = np.array([p for i, p in enumerate(points) if i in inds_failed])
        success_points = np.array([p for i, p in enumerate(points) if i in inds_success])
        success_values = np.array([v for i, v in enumerate(res) if i in inds_success])
        success_values = success_values.reshape(-1, 1) # switch to column array
        if len(inds_failed) > 0:
            self.failed_points_history = self.add_to_history(self.failed_points_history, failed_points, dim=num_params)
        if len(inds_success) > 0:
            self.points_history = self.add_to_history(self.points_history, success_points, dim=num_params)
            self.values_history = self.add_to_history(self.values_history, success_values, dim=1)

        return res


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
