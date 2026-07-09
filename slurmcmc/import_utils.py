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

    The import is performed at most once per object instance and then cached.
    Since pickling drops the cache (only the module/function names survive),
    each unpickled copy — e.g. inside a remote Slurm job — re-imports fresh
    definitions, while repeated local calls avoid re-import overhead.
    """

    def __init__(self, module_dir, module_name, function_name):
        self.module_dir = module_dir
        self.module_name = module_name
        self.function_name = function_name
        self._cached_fun = None

    def __call__(self, *args, **kwargs):
        if self._cached_fun is None:
            self._cached_fun = import_function_from_module(self.module_dir, self.module_name, self.function_name)
        return self._cached_fun(*args, **kwargs)

    def __getstate__(self):
        # Ensure only the attributes are pickled, not the imported function
        return {'module_dir': self.module_dir, 'module_name': self.module_name, 'function_name': self.function_name}

    def __setstate__(self, state):
        # Reconstruct the object from the pickled state
        self.module_dir = state['module_dir']
        self.module_name = state['module_name']
        self.function_name = state['function_name']
        self._cached_fun = None


def deferred_import_function_wrapper(fun):
    if callable(fun):
        return fun
    elif isinstance(fun, dict):
        for key in ('module_dir', 'module_name', 'function_name'):
            if key not in fun:
                err_msg = f'input is a dict, but does not contain {key} key.'
                logging.error(err_msg)
                raise ValueError(err_msg)

        # Return a deferred function object instead of importing immediately
        return DeferredImportFunction(fun['module_dir'], fun['module_name'], fun['function_name'])
    else:
        err_msg = f'input should be function or dictionary, type(fun)={type(fun)}'
        logging.error(err_msg)
        raise ValueError(err_msg)
