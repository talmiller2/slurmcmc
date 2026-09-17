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
- **Automated hybrid surrogate-MCMC** (`slurm_mcmc_hybrid`): iteratively refine a Gaussian-process (or polynomial) surrogate with importance-weight validation until the MCMC distribution converges — typically at a fraction of the expensive-evaluation cost.
- **Full audit trail**: each function evaluation gets its own directory with `input.txt`, `output.txt`, and `inputs.txt`/`outputs.txt` per iteration, and every point and result is also collected in `points_history.txt`/`values_history.txt`, the in-memory history written to disk. When only the results matter, `keep_run_dirs='none'` (or `'failed'`) removes the per-iteration directories once their results are in.
- **Restart/checkpoint support**: `save_restart`/`load_restart` on `slurm_minimize`, `slurm_mcmc` and `slurm_mcmc_hybrid` — resume a run that was interrupted, without repeating any expensive evaluation. Writes are atomic, so a job killed mid-save cannot corrupt the file. A `SlurmPool` used on its own takes `load_restart=True`, which rebuilds its call counter and evaluation history from those history files.
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

## API

The two entry points are `slurm_minimize(...)` and `slurm_mcmc(...)` (keyword-argument
convenience wrappers). Under the hood each is a thin layer over a config dataclass and
a runner class, which can also be used directly:

```python
from slurmcmc.optimization import MinimizeConfig, Minimizer
from slurmcmc.mcmc import MCMCConfig, MCMCRunner

result = Minimizer(MinimizeConfig(loss_fun=..., param_bounds=..., num_workers=..., num_iters=...)).run()
status = MCMCRunner(MCMCConfig(log_prob_fun=..., init_points=..., num_iters=...)).run()
```

With `remote=True`, the whole optimization/MCMC loop is submitted as its own Slurm job
(so it survives login-node limits) by pickling the config object into the driver job;
the call returns a `submitit.Job` whose `.result()` is the status dict.

Each iteration's points are submitted as a single Slurm **job array**
(one scheduler transaction per iteration), and every point gets its own working
directory with an `input.txt`/`output.txt` audit trail.

---

## Install

Install the package (core dependencies are pulled in automatically):

```bash
pip install -e .
```

To also use the Bayesian optimization backend (botorch), or the hybrid
surrogate-MCMC pipeline (scikit-learn):

```bash
pip install -e ".[botorch]"
pip install -e ".[hybrid]"
```

Or install everything needed for the examples or the tests:

```bash
pip install -e ".[examples]"
pip install -e ".[test]"
```

To install everything at once — the package with every optional dependency:

```bash
pip install -e ".[botorch,hybrid,examples,test]"
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
1. [MCMC with surrogate (automated hybrid pipeline)](examples/docs/mcmc_surrogate_hybrid.md)
