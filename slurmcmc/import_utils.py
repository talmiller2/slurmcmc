import importlib
import logging
import os
import sys


def import_function_from_module(module_dir, module_name, function_name):
    """
    import a function defined in a python file in a remote directory.
    """
    if not os.path.isdir(module_dir):
        err_msg = f'Directory {module_dir} does not exist.'
        logging.error(err_msg)
        raise ValueError(err_msg)
    sys.path.insert(0, module_dir)

    try:
        module = importlib.import_module(module_name)
        importlib.reload(module)  # Refresh definitions in case the module changed
        imported_function = getattr(module, function_name)
        return imported_function
    finally:
        # clean up sys.path to avoid side effects
        sys.path.pop(0)


class DeferredImportFunction:
    """
    A callable that defers importing a function until it's actually called.
    """

    def __init__(self, module_dir, module_name, function_name):
        self.module_dir = module_dir
        self.module_name = module_name
        self.function_name = function_name

    def __call__(self, *args, **kwargs):
        # Import the function only when called
        imported_fun = import_function_from_module(self.module_dir, self.module_name, self.function_name)
        return imported_fun(*args, **kwargs)

    def __getstate__(self):
        # Ensure only the attributes are pickled, not the imported function
        return {'module_dir': self.module_dir, 'module_name': self.module_name, 'function_name': self.function_name}

    def __setstate__(self, state):
        # Reconstruct the object from the pickled state
        self.module_dir = state['module_dir']
        self.module_name = state['module_name']
        self.function_name = state['function_name']


def deferred_import_function_wrapper(fun):
    if callable(fun) == True:
        return fun
    elif type(fun) == dict:
        if 'module_dir' in fun.keys():
            module_dir = fun['module_dir']
        else:
            err_msg = 'input is a dict, but does not contain module_dir key.'
            logging.error(err_msg)
            raise ValueError(err_msg)
        if 'module_name' in fun.keys():
            module_name = fun['module_name']
        else:
            err_msg = 'input is a dict, but does not contain module_name key.'
            logging.error(err_msg)
            raise ValueError(err_msg)
        if 'function_name' in fun.keys():
            function_name = fun['function_name']
        else:
            err_msg = 'input is a dict, but does not contain function_name key.'
            logging.error(err_msg)
            raise ValueError(err_msg)

        # Return a deferred function object instead of importing immediately
        deferred_import_function = DeferredImportFunction(module_dir, module_name, function_name)
        return deferred_import_function
    else:
        err_msg = f'input should be function or dictionary, type(fun)={type(fun)}'
        logging.error(err_msg)
        raise ValueError(err_msg)
