import os
import shutil
import unittest

from slurmcmc.import_utils import import_function_from_module, imported_fun
from slurmcmc.slurm_utils import SlurmPool


class test_map_local_imported_fun(unittest.TestCase):
    def setUp(self):
        os.chdir(os.path.dirname(__file__)) # needed to make tests not crash on a cluster

        self.work_dir = os.path.dirname(__file__) + '/test_work_dir'
        self.verbosity = 1

        module_dict = {}
        module_dict['module_dir'] = os.path.dirname(__file__) + '/test_module_dir/'
        module_dict['module_name'] = 'test_module'
        module_dict['function_name'] = 'fun_with_extra_arg'
        self.module_dict = module_dict
        self.fun_with_extra_arg = import_function_from_module(self.module_dict['module_dir'],
                                                              self.module_dict['module_name'],
                                                              self.module_dict['function_name'])

    def tearDown(self):
        if os.path.isdir(self.work_dir):
            shutil.rmtree(self.work_dir)
        pass

    def test_slurmpool_local_imported_fun_fail(self):
        """
        using a function that is imported from a different directory, should fail when running with
        cluster='local' or 'slurm' because does not (cloud)pickle properly.
        """
        setup_dict = {'weather': 'sunny'}
        points = [2, 3, 4]
        slurm_pool = SlurmPool(self.work_dir, cluster='local', verbosity=self.verbosity, extra_arg=setup_dict)
        with self.assertRaises(Exception):
            slurm_pool.map(self.fun_with_extra_arg, points)

    def test_slurmpool_local_using_imported_fun(self):
        """
        use imported_fun to allow the function to pass the submitit pipeline,
        with the module_dict supplied as an extra_arg.
        """
        setup_dict = {'weather': 'sunny'}
        setup_dict.update(self.module_dict)
        points = [2, 3, 4]
        res_expected = [self.fun_with_extra_arg(point, setup_dict) for point in points]
        slurm_pool = SlurmPool(self.work_dir, cluster='local', verbosity=self.verbosity, extra_arg=setup_dict)
        res = slurm_pool.map(imported_fun, points)
        self.assertEqual(res, res_expected)


if __name__ == '__main__':
    unittest.main()
