# slurmcmc

Perform model calibration with uncertainty quantification (also known as Bayesian model calibration) for models that require computationally expensive black-box queries, using parallel computing on a Slurm-managed cluster.
Implemented by stitching together [``submitit``](https://github.com/facebookincubator/submitit) + [``nevergrad``](https://github.com/facebookresearch/nevergrad) + [``botorch``](https://github.com/pytorch/botorch) + [``emcee``](https://github.com/dfm/emcee).


Install locally using
```
pip install -e .
```

Run tests from root project folder using:
```
cd tests; python run_all_tests.py
```

Run a specific test
```
python -m unittest <test_module>.<TestClass>.<test_method>
```

Quick how to use in the examples folder.
