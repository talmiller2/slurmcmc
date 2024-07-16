import os
import pickle


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
