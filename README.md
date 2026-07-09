# slurmcmc

Perform model calibration with uncertainty quantification (Bayesian model calibration) for computationally expensive black-box models, using parallel computing on a Slurm-managed cluster.

Implemented by wrapping and stitching together [`submitit`](https://github.com/facebookincubator/submitit) + [`nevergrad`](https://github.com/facebookresearch/nevergrad) + [`botorch`](https://github.com/pytorch/botorch) + [`emcee`](https://github.com/dfm/emcee).

<div align="center">
    <img src="examples/docs/pics/logo.jpeg" alt="slurmcmc logo">
</div>

---

## Features

- **Parallel black-box optimization** via [`nevergrad`](https://github.com/facebookresearch/nevergrad) (Differential Evolution, PSO, …) or Bayesian optimization via [`botorch`](https://github.com/pytorch/botorch) (Gaussian Process with Expected Improvement).
- **Ensemble MCMC** via [`emcee`](https://github.com/dfm/emcee), with walkers evaluated in parallel on a cluster.
- **MCMC with a surrogate model**: evaluate an expensive model to build a surrogate, then sample the surrogate cheaply.
- **Full audit trail**: each function evaluation gets its own directory with `input.txt`, `output.txt`, and `inputs.txt`/`outputs.txt` per iteration.
- **Restart/checkpoint support**: save and resume long runs from pickle files.
- **Constraint handling**: skip infeasible points before evaluating the expensive function.
- **Deferred function import**: pass a `{module_dir, module_name, function_name}` dict to avoid pickling issues with remotely-defined functions.

---

## Parallelization modes

Set via the `cluster` argument:

| `cluster` | Description |
|---|---|
| `'slurm'` | Submit jobs to a Slurm cluster via `submitit`. |
| `'local'` | Run locally using `submitit`'s local executor (same directory layout as `'slurm'` — useful for debugging). |
| `'local-map'` | Evaluate sequentially in-process. Fastest for analytic functions and CI tests. |

---

## Install

Install the package (core dependencies are pulled in automatically):

```bash
pip install -e .
```

To also use the Bayesian optimization backend (botorch):

```bash
pip install -e ".[botorch]"
```

Or install everything needed for the examples or the tests:

```bash
pip install -e ".[examples]"
pip install -e ".[test]"
```

Requires Python >= 3.10.

> **On a shared Slurm cluster without admin/root access**, add `--user` to install into your
> home directory instead of the system site-packages, e.g. `pip install -e . --user` or
> `pip install -e ".[test]" --user`.

---

## Run tests

```bash
pytest -vv tests
```

Tests that require a real Slurm cluster are automatically skipped locally (they are marked with `@pytest.mark.skipif(not is_slurm_cluster(), ...)`). The mock-Slurm tests (`tests/test_mock_slurm.py`) exercise the Slurm code path without a cluster.

To run a specific test:

```bash
pytest -vv tests/test_map_local.py::test_slurmpool_localmap
```

---

## Pedagogical Examples

1. [Optimization](examples/docs/optimization.md)
1. [Comparing Optimization Algorithms](examples/docs/optimization_algorithms_comparison.md)
1. [MCMC](examples/docs/mcmc.md)
1. [Comparing MCMC and MC](examples/docs/mcmc_and_mc_comparison.md)
1. [MCMC with surrogate](examples/docs/mcmc_surrogate.md)
