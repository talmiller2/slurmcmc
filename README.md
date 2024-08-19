# slurmcmc

Perform model calibration with uncertainty quantification (also known as Bayesian model calibration) for models that require computationally expensive black-box queries, using parallel computing on a Slurm-managed cluster.
Implemented by stitching together [``submitit``](https://github.com/facebookincubator/submitit) + [``nevergrad``](https://github.com/facebookresearch/nevergrad) + [``botorch``](https://github.com/pytorch/botorch) + [``emcee``](https://github.com/dfm/emcee).

<div align="center">
    <img src="examples/docs/pics/logo.jpeg" alt="slurmcmc logo" width="700" height="auto">
</div>


## Install

Install prerequisites:

```
pip install submitit nevergrad botorch emcee corner
```

Install the package locally using
```
pip install -e .
```

Run tests from root project folder using:
```
pytest -vv tests
```

To run a specific test, for example
```
pytest -vv tests/test_map_local.py::test_slurmpool_localmap
```

## Parallelization

The algorithms used are based on parallel evaluations of functions, which we want to perform on Slurm-managed cluster.
The run options determined by the  `cluster` argument whose options are:
* `'slurm'`: parallel evaluations using jobs submitted to Slurm via [``submitit``](https://github.com/facebookincubator/submitit).
* `'local'`: parallel evaluations using local processes, also via [``submitit``](https://github.com/facebookincubator/submitit) (on the same directories structure as in `'slurm'`, useful for debugging).
* `'local-map'`: evaluate function in-line sequentially on multiple points. Useful for analytic functions where 
parallelization is less important, for demonstrations and for debugging. 
This mode is used in the examples below for quick evaluations.


## Examples

* [Optimization](examples/docs/optimization.md)
* [Bayesian model calibration (MCMC)](examples/docs/mcmc.md)
