import importlib
import os
import sys


def import_function_from_module(module_dir, module_name, function_name):
    """
    Import a function defined in a python file that is not in the same directory of the running script.
    """
    if os.path.isdir(module_dir):
        sys.path.insert(0, module_dir)
    else:
        raise ValueError('directory ' + module_dir + ' does not exist.')
    module = importlib.import_module(module_name)
    importlib.reload(module)  # refreshes the definitions in case the module was already loaded and was changed
    imported_function = getattr(module, function_name)
    return imported_function


def imported_fun(x, module_dict):
    '''
    Define a function that is imported from a different dir than the main dir.
    Allows the function to be passed through the submitit pipeline without error.
    To use, pass module_dict as an extra_arg when defining slurm_utils.SlurmPool.
    '''
    fun = import_function_from_module(module_dir=module_dict['module_dir'],
                                      module_name=module_dict['module_name'],
                                      function_name=module_dict['function_name'])
    return fun(x, module_dict)
