import logging
import os
import pickle
import shutil
import time


def set_logging(work_dir=None, log_file=None):
    if log_file is not None:
        # create save directory and log file
        os.makedirs(work_dir, exist_ok=True)
        log_file_path = work_dir + '/' + log_file

        # basic logging definition
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)

        # remove any previously defined loggers
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)

        # print log messages to a log file
        fh = logging.FileHandler(log_file_path)
        fh_formatter = logging.Formatter(fmt='%(asctime)s %(message)s', datefmt='%d-%m-%Y %H:%M:%S')
        fh.setFormatter(fh_formatter)
        fh.setLevel(logging.INFO)
        logger.addHandler(fh)

        # print log messages to the console/termnal
        ch = logging.StreamHandler()
        ch_formatter = logging.Formatter('%(message)s')
        ch.setFormatter(ch_formatter)
        ch.setLevel(logging.INFO)
        logger.addHandler(ch)

    else:
        # print log messages to the console/terminal only
        logging.basicConfig(format='%(message)s', level=logging.DEBUG)

        # disable the warnings generated by matplotlib when in interactive mode
        logging.getLogger('matplotlib').setLevel(logging.ERROR)

    return


def print_log(string, work_dir, log_file):
    if log_file is None:
        print(string)
    else:
        os.makedirs(work_dir, exist_ok=True)
        with open(work_dir + '/' + log_file, 'a') as log_file:
            print(string, file=log_file)


def save_restart_file(status, work_dir, restart_file):
    os.makedirs(work_dir, exist_ok=True)
    with open(work_dir + '/' + restart_file, 'wb') as f:
        pickle.dump(status, f)


def load_restart_file(work_dir, restart_file):
    with open(work_dir + '/' + restart_file, 'rb') as f:
        status = pickle.load(f)
    return status


def combine_args(arg, extra_arg=None):
    args = [arg]
    if extra_arg is not None:
        args += [extra_arg]
    return args


def delete_directory_with_retries(path, delay=1, retries=5):
    """
    Delete a directory with multiple retries.
    """
    for ind_retry in range(retries):

        # If directory does not exist, then either did not exist or was previously deleted
        if not os.path.isdir(path):
            return True

        # Attempt to delete the directory
        try:
            shutil.rmtree(path)

        except OSError as e:
            if e.errno == 16:  # Errno 16 is 'Device or resource busy'
                print(f"Retrying ({ind_retry + 1}/{retries})... {e.strerror}")
                time.sleep(delay)

    return True
