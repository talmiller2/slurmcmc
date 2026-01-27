import logging
import os
import shutil
import subprocess
import time
import numpy as np
import submitit

from slurmcmc.general_utils import (combine_args, set_logging, save_extra_arg_to_file, point_to_tuple, list_directories,
                                    calc_dimension)
from slurmcmc.import_utils import deferred_import_function_wrapper


class SlurmPool():
    """
    A class defined for the sole purpose of consistency with the pool.map syntax for parallelization in the emcee
    package. Instead of using multiprocessing, we integrate with the submitit package to perform parallel calculations
    on a cluster of multiple processors.

    cluster: 'slurm' or 'local' (run locally with submitit) or 'local-map' (run locally with map)
    """

    def __init__(self, work_dir='slurmpool', job_name='slurmpool', cluster='slurm',
                 verbosity=1, log_file=None, extra_arg=None, submitit_kwargs=None,
                 dim_input=None, dim_output=None,
                 budget=int(1e6), job_fail_value=np.nan,
                 submit_retry_max_attempts=5, submit_retry_wait_seconds=10, submit_delay_seconds=0,
                 check_output_interval_seconds=1, check_output_timeout_minutes=int(1e5),
                 record_history=True,
                 ):
        if not isinstance(dim_input, int) and not dim_input > 0:
            err_msg = f'dim_input must be a positive integer. dim_input={dim_input}'
            logging.error(err_msg)
            raise ValueError(err_msg)
        self.dim_input = dim_input
        if not isinstance(dim_output, int) and not dim_output > 0:
            err_msg = f'dim_output must be a positive integer. dim_output={dim_output}'
            logging.error(err_msg)
            raise ValueError(err_msg)
        self.dim_output = dim_output
        self.num_calls = 0  # initialize call counter
        self.run_time_minutes_per_call = []
        self.record_history = record_history
        if self.record_history:
            self.num_evaluated_points = 0  # initialize counter
            self.points_history = []  # all points
            self.values_history = []  # function values of all points
            self.inds_success_points = []  # indices of points that were successfully calculated
            self.inds_failed_points = []  # indices of points where the calculation failed
            self.evaluated_points_set = set()  # for fast checking of points that appear in points_history
            self.point_loc_dict = {}
        self.work_dir = work_dir
        self.job_name = job_name
        self.cluster = cluster
        self.verbosity = verbosity
        self.log_file = log_file
        if submitit_kwargs is None:
            submitit_kwargs = {}
        if 'slurm_job_name' not in submitit_kwargs:
            submitit_kwargs['slurm_job_name'] = job_name
        if 'timeout_min' not in submitit_kwargs:
            submitit_kwargs['timeout_min'] = int(60 * 24 * 30)  # 1 month
        self.submitit_kwargs = submitit_kwargs
        self.budget = budget
        self.job_fail_value = job_fail_value
        self.submit_retry_max_attempts = submit_retry_max_attempts
        self.submit_retry_wait_seconds = submit_retry_wait_seconds
        self.submit_delay_seconds = submit_delay_seconds
        self.check_output_interval_seconds = check_output_interval_seconds
        self.check_output_timeout_minutes = check_output_timeout_minutes

        set_logging(self.work_dir, self.log_file)

        if cluster in ['local', 'slurm']:
            os.makedirs(work_dir, exist_ok=True)
            if len(list_directories(work_dir)) > 0:
                err_msg = 'work_dir appears to already contain runs, move or delete it first.'
                err_msg += '\n' + f'work_dir: {work_dir}'
                raise ValueError(err_msg)
                # note: in case of continuing from restart, SlurmPool is loaded and not initialized so will not error.

        # capability to call the function with additional argument that is constant during the map
        self.extra_arg = extra_arg

        return

    def map(self, fun, points):
        fun = deferred_import_function_wrapper(fun)

        # split points into chunks if exceeding budget
        chunks = self.split_points(points, self.budget)
        chunk_sizes = [len(chunk) for chunk in chunks]
        if self.verbosity >= 1 and len(chunks) > 1:
            logging.info('split points into ' + str(len(chunks)) + ' chunks of sizes ' + str(chunk_sizes) + '.')

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

    def _combine_args(self, point):
        return combine_args(point, self.extra_arg)

    def submit_with_retry(self, fun, point):
        attempts = 0
        while attempts < self.submit_retry_max_attempts:
            try:
                job = self.executor.submit(fun, *self._combine_args(point))
                return job
            except Exception as e:
                attempts += 1
                logging.info(f"Submission failed: {e}. Retrying {attempts}/{self.submit_retry_max_attempts}")
                time.sleep(self.submit_retry_wait_seconds)  # wait before retrying
            if attempts == self.submit_retry_max_attempts:
                err_msg = "max submit retry attempts reached."
                logging.error(err_msg)
                raise Exception(err_msg)

    def map_chunk(self, fun, points):
        map_start_time = time.time()

        if self.verbosity >= 1:
            logging.info('slurm_pool.map_chunk called with ' + str(len(points)) + ' points.')

        # check if points have correct dimensions
        for point in points:
            dim_curr_input = calc_dimension(point)
            if dim_curr_input != self.dim_input:
                err_msg = (f'inconsistent dimensions. expecting dim_input={self.dim_input} '
                           f'but dim_curr_input={dim_curr_input}')
                logging.error(err_msg)
                raise ValueError(err_msg)

        # calculate fun on the points
        if self.cluster == 'local-map':
            res = [fun(*self._combine_args(point)) for point in points]
        else:
            res = self.send_and_receive_jobs(fun, points)

        # check if outputs have correct dimensions
        for output in res:
            dim_curr_output = calc_dimension(output)
            if dim_curr_output != self.dim_output:
                err_msg = (f'inconsistent dimensions. expecting dim_output={self.dim_output} '
                           f'but dim_curr_output={dim_curr_output}')
                logging.error(err_msg)
                raise ValueError(err_msg)

        if self.record_history:
            # track if point was previously evaluated
            for point in points:
                point_tuple = point_to_tuple(point)
                if point_tuple not in self.evaluated_points_set:
                    self.evaluated_points_set.add(point_tuple)

            # update history arrays
            inds_failed = [self.num_evaluated_points + i for i, v in enumerate(res) if self.check_failed(v)]
            inds_success = [self.num_evaluated_points + i for i, v in enumerate(res) if not self.check_failed(v)]
            self.inds_failed_points += inds_failed
            self.inds_success_points += inds_success
            self.points_history = self.add_to_history(self.points_history, np.array(points), dim=self.dim_input)
            self.values_history = self.add_to_history(self.values_history, np.array(res), dim=self.dim_output)
            self.num_evaluated_points += len(res)

        # update call counters
        self.num_calls += 1
        map_run_time_minutes = (time.time() - map_start_time) / 60.0
        self.run_time_minutes_per_call += [map_run_time_minutes]

        return res

    def check_failed(self, r):
        if type(r) == list or (type(r) == np.ndarray and r.ndim > 0):
            pass
        else:
            r = [r]
        for e in r:
            if e == None or np.isnan(e) or e == self.job_fail_value:
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

            if self.record_history:
                # track location of point calculation
                point_tuple = point_to_tuple(point)
                if point_tuple not in self.evaluated_points_set:
                    self.point_loc_dict[point_tuple] = (self.num_calls, ind_point)

        # save current iteration points in the main iteration_dir
        np.savetxt(iteration_dir + '/inputs.txt', np.array(points))

        # save the extra_arg in the run folder to document the full input used for this point
        save_extra_arg_to_file(iteration_dir, self.extra_arg)

        # send the jobs
        jobs = []
        for ind_point, (point, point_dir) in enumerate(zip(points, point_dirs)):
            job_name = self.job_name + '_' + str(self.num_calls) + '_' + str(ind_point)
            self.submitit_kwargs['slurm_job_name'] = job_name
            self.executor = submitit.AutoExecutor(folder=point_dir, cluster=self.cluster)
            self.executor.update_parameters(**self.submitit_kwargs)
            os.chdir(point_dir)  # each point evaluation (query) is born in its own dir
            job = self.submit_with_retry(fun, point)
            jobs += [job]
            if self.submit_delay_seconds > 0:
                time.sleep(self.submit_delay_seconds)

        # collect the results
        outputs = []
        for ind_point, job in enumerate(jobs):
            try:
                # initialize variables for tracking
                check_output_timeout_seconds = self.check_output_timeout_minutes * 60
                running_started = False
                job_running_start_time = None
                job_failed = False

                # job state monitoring loop, will escape if job is no longer running or pending
                state = check_job_state(job, self.cluster)
                while state in ['RUNNING', 'PENDING']:
                    if state == 'RUNNING':
                        if not running_started:
                            running_started = True
                            job_running_start_time = time.time()
                        else:
                            # Check elapsed time since job started running
                            job_running_time = time.time() - job_running_start_time
                            if job_running_time > check_output_timeout_seconds:
                                if self.verbosity >= 1:
                                    logging.info(f"ind_point {ind_point} job {job.job_id} exceeded running time of "
                                                 f"{self.check_output_timeout_minutes:.2f} minutes. Cancelling job.")
                                job.cancel()  # Works for both local (terminates subprocess) and Slurm (sends scancel)
                                job_failed = True
                                break

                    time.sleep(self.check_output_interval_seconds)
                    state = check_job_state(job, self.cluster)

                if not job_failed:
                    # load the result as done within submitit's job.result(), but avoid the faulty job.wait()
                    outcome, output = job._get_outcome_and_result()
                    if outcome == "error":
                        if self.verbosity >= 1:
                            logging.info('job._get_outcome_and_result() failed. The exception message:\n'
                                         + str(job.exception()))
                        job_failed = True

            except Exception as e:
                if self.verbosity >= 1:
                    logging.info('Failed obtaining job result. The exception message:\n' + str(e))
                job_failed = True

            if job_failed:
                output = self.job_fail_value

            point_dir = iteration_dir + '/' + str(ind_point)
            np.savetxt(point_dir + '/output.txt', [output])
            outputs += [output]

        # save current iteration results
        np.savetxt(iteration_dir + '/outputs.txt', np.array(outputs))

        os.chdir(ini_dir)  # return to initial dir
        return outputs


def is_slurm_cluster():
    """
    return True if running in a computer connected to Slurm cluster
    """
    return shutil.which('srun') != None


def check_job_state(job, cluster):
    """
    Check the state of a job, depending on the cluster type used.
    """
    if cluster == 'local':
        return job.state
    elif cluster == 'slurm':
        job_id = job.job_id
        return check_slurm_job_state(job_id)
    else:
        err_msg = f"invalid cluster type: {cluster}."
        logging.error(err_msg)
        raise ValueError(err_msg)


def check_slurm_job_state(job_id):
    """
    Check the state of a SLURM job by its job ID.
    """
    try:
        # Run squeue command to check job status for the specific job ID
        result = subprocess.run(
            ['squeue', '-j', str(job_id), '-h', '-o', '%T'],
            capture_output=True,
            text=True,
            check=True
        )

        # Get the output (job state)
        output = result.stdout.strip()

        # If output is empty, the job is not in the queue
        if not output:
            return "NOT_FOUND"

        # Check the job state
        if output in ["RUNNING", "PENDING"]:
            return output
        else:
            return "OTHER"  # For other states like COMPLETING, FAILED, etc.

    except subprocess.CalledProcessError as e:
        # Check if the error is due to an invalid job ID
        if "Invalid job id" in e.stderr:
            return "NOT_FOUND"
        # Handle other errors
        err_msg = f"Error checking job status: {e}"
        logging.error(err_msg)
        raise ValueError(err_msg)
