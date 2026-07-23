# pypolar

Python implementation of POLAR (Preference Optimization and Learning Algorithm for Robotics) for preference-based learning. Uses JAX for automatic differentiation and JIT compilation of likelihood functions.

## Project structure

```
pypolar/
├── pyproject.toml              # Package config, dependencies, pytest settings
├── src/pypolar/                # Installable package
│   ├── __init__.py             # Public API, enables JAX float64
│   ├── pbl.py                  # PreferenceBasedLearning - core preference logic (JAX likelihoods)
│   ├── gp/                     # GP models, split by posterior representation
│   │   ├── kernels.py          # SquaredExponential
│   │   ├── base.py             # GPModel ABC - kernel algebra, shared sampling
│   │   ├── conjugate.py        # ConjugateGP - exact posterior for regression feedback
│   │   ├── laplace.py          # LaplaceGP - JAX autodiff + scipy for other likelihoods
│   │   └── multi.py            # MultiObjectiveGP
│   ├── feedback.py             # SimulatedFeedback, SimulatedObjective - feedback simulation
│   └── sampler.py              # RandomSampler - action sampling
├── examples/                   # Runnable example scripts
│   ├── example_1d.py           # 1D preference learning (verification script)
│   ├── example_2d.py           # 2D preference learning with 3D plots
│   └── example_3d.py           # 3D (RGB color) preference learning
├── scripts/                    # Scratch/experimentation files (not part of package)
└── tests/                      # pytest test suite
    ├── test_pbl.py             # Tests for PreferenceBasedLearning + JAX compatibility
    ├── test_gp.py              # Tests for the GP models
    ├── test_feedback.py        # Tests for SimulatedFeedback/SimulatedObjective
    └── test_sampler.py         # Tests for RandomSampler
```

## Environment setup

Use the `pypolar` conda environment for all operations:

```bash
conda activate pypolar
pip install -e ".[dev]"    # editable install with test/plot deps
```

## How to run tests

```bash
conda activate pypolar
python -m pytest tests/ -v
```

47 tests covering action space generation, index mapping, sigmoid functions, feedback collection, likelihood computation, JAX grad/jit compatibility, GP kernel/fitting, and sampling.

## How to run examples

```bash
conda activate pypolar
python examples/example_1d.py    # primary verification script
python examples/example_2d.py
python examples/example_3d.py
```

`example_1d.py` is the go-to script for verifying the package works end-to-end. It collects 20 pairwise comparisons, fits a GP, evaluates prediction accuracy (~85%, varies with the random query trajectory), and shows a matplotlib plot.

## Package API

All public classes are importable directly from `pypolar`:

```python
from pypolar import PreferenceBasedLearning, ConjugateGP, LaplaceGP, MultiObjectiveGP, RandomSampler
```

Importing `pypolar` enables JAX float64 mode (`jax_enable_x64`), required for numerical stability in the likelihood functions.

### PreferenceBasedLearning (`pbl.py`)

Core class that manages the discretized action space and feedback likelihoods. Likelihood functions use `jax.numpy` and are compatible with `jax.grad`, `jax.jit`, and `jax.hessian`.

**Constructor args:**
- `low`, `high` (array): bounds of the action space per dimension
- `action_dims` (array of int): number of discretization points per dimension
- `preference_noise`, `coactive_noise`, `ordinal_noise` (float): noise parameters c_p, c_c, c_o

**Key methods:**
- `add_feedback(preference, coactive, ordinal)` - add human/simulated feedback tuples (pass `None` for unused types)
  - preference: `(curr_action, prev_action, curr_is_preferred: bool)`
  - coactive: `(suggested_action, curr_action, suggested_is_better: bool)`
  - ordinal: `(action, lower_bound, upper_bound)`
- `compile()` - **must be called after adding feedback and before using `overall_likelihood` (legacy path).** Converts feedback lists into JAX arrays. Clears compiled state when new feedback is added. Not needed for the `feedback_data` / `likelihood_from_data` fast path below.
- `overall_likelihood(r)` - compute total negative log-likelihood given reward vector `r`. Compatible with `jax.grad` and `jax.jit`. Feedback is baked in as a constant, so a JIT'd version recompiles whenever feedback changes.
- `feedback_data()` - build a padded JAX feedback pytree for the recompilation-free fast path. Array shapes grow geometrically (capacity doubling), so they change only O(log n) times over a run. Reads the raw feedback lists directly (no `compile()` required). Pass to `likelihood_from_data` to build a single-argument likelihood for `LaplaceGP.set_data`.
- `likelihood_from_data(r, data)` - same negative log-likelihood as `overall_likelihood`, but as a pure function of `(r, data)` where `data` comes from `feedback_data()`. Because feedback is a runtime argument, a JIT'd version is reused across iterations and recompiles only when an array shape grows. Numerically identical to `overall_likelihood`.
- `predict(r, a1, a2)` - predict whether a1 is preferred over a2 (numpy)
- `optimal_action(r)` - return action with highest predicted reward (numpy)
- `get_idx(action)` - map continuous action to nearest index in discretized space (numpy)

**Key attributes:**
- `action_space` - numpy array of shape `(N, d)` containing all discretized actions

**JAX usage:**
```python
pbl.compile()
grad_fn = jax.grad(pbl.overall_likelihood)
gradient = grad_fn(jnp.array(r))

jitted_likelihood = jax.jit(pbl.overall_likelihood)
value = jitted_likelihood(jnp.array(r))
```

### GP models (`optimization/gp/`)

The GP is split by **posterior representation**. Both backends implement the same
contract, so samplers and plotting are backend-agnostic:
`set_data(...)` -> `fit()` / `std()` / `posterior_cov()` / `prepare_sampling()` /
`sample_posterior(rng)`.

- `gp/kernels.py` - `SquaredExponential`, `make_kernel`
- `gp/base.py` - `GPModel` ABC: kernel algebra, cached prior Cholesky, `functionalize`
- `gp/conjugate.py` - `ConjugateGP`
- `gp/laplace.py` - `LaplaceGP`
- `gp/multi.py` - `MultiObjectiveGP`

**Shared constructor args** (`ConjugateGP`, `LaplaceGP`):
- `kernel` (str or kernel object): currently only `'squared_exp'`
- `signal_variance` (float): controls reward magnitude
- `length_scale` (float): controls smoothness
- `rng`: numpy random Generator

#### ConjugateGP

Exact GP posterior for Gaussian (regression) feedback. Solves only the M x M
system `G = K_XX + sigma^2 I`, where M is the number of feedback points, so it
never forms an N x N matrix: O(N M^2) time, O(N M) memory. `fit()` is a single
triangular solve, with no optimizer and no autodiff. Required for large action
spaces.

- `set_data(action_space, idx, y, precision)` - `(idx, y, precision)` is exactly
  what `MultiObjectiveRegression.get_regression_data()` returns per objective.
- `fit(**kwargs)` - closed-form posterior mean. kwargs accepted and ignored.
- `std()` - variance diagonal directly, in O(N M^2). Prefer over
  `posterior_cov()`, which is O(N^2) memory.
- `sample_posterior(rng)` - Matheron's rule; factorizes nothing beyond the cached
  (data-independent) prior Cholesky and the M x M Gram factor.

#### LaplaceGP

Laplace approximation for non-Gaussian likelihoods (e.g. the sigmoid preference
likelihood). Minimizes `likelihood(r) + 0.5 r^T K^-1 r` with scipy, using exact
gradients and Hessians from JAX autodiff, and approximates the posterior
covariance by the inverse Hessian at the mode.

- Extra constructor arg: `x0_init_method` (str), currently only `'random'`
- `set_data(action_space, likelihood, quadratic=None)` - `likelihood` is called
  as `likelihood(r)` and must be JAX-compatible. Builds JIT-compiled objective,
  jacobian and hessian. The prior covariance and its inverse are cached and
  recomputed only when the action space or kernel hyperparameters change.
  Feedback is baked in as a constant, so `set_data()` must be re-run when
  feedback changes -- and each re-run recompiles.
- `quadratic` - whether the objective has a constant Hessian. `None` (default)
  auto-detects. This is a *solver shortcut*, not a different model: when it
  holds, `fit()` does one `cho_solve` (`mu = -H^-1 g0`) instead of calling
  scipy, and the Laplace posterior is exact. Regression likelihoods are
  quadratic, so `LaplaceGP` and `ConjugateGP` agree to ~1e-12 on them --
  prefer `ConjugateGP` there, it is far cheaper.
- `fit(**kwargs)` - `method` and `options` are passed to
  `scipy.optimize.minimize`. Good methods: `'trust-constr'` (1D),
  `'trust-krylov'` (higher dims). Ignored on the closed-form path.

**Key attributes:** `mu` (current reward mean, length N), `actions`,
`objective`/`jacobian`/`hessian` (scipy-compatible numpy wrappers),
`_jax_objective`/`_jax_jacobian`/`_jax_hessian` (JIT-compiled).

#### MultiObjectiveGP

Independent per-objective GPs over a shared action space, with per-objective
kernel hyperparameters passed as lists.

- `setup(action_space, likelihoods=None, regressions=None)` - pass exactly one.
  `regressions` selects `ConjugateGP`, `likelihoods` selects `LaplaceGP`; the
  backend is switched to match. `gps` is built eagerly in `__init__` and mutated
  in place, so `len(optimizer.gps)` and samplers constructed with
  `gps=optimizer.gps` are valid before the first `setup()`.
- `fit(**kwargs)` - fits every objective, returns the list of means
- `std(r=None)` - per-objective standard deviations
- `gps` - the list of per-objective models

### SimulatedFeedback (`feedback.py`)

Simulates feedback from a perfect decider with a known objective.

**Constructor args:**
- `objective` - a callable (e.g. `SimulatedObjective`) that maps actions to scalar rewards
- `action_space` - the discretized action space (required for coactive feedback)

**Key method:**
- `evaluate(curr, prev=None, get_pairwise=False, get_coactive=False, get_ordinal=False)` - returns `(preference, coactive, ordinal)` tuple. Each is `None` unless the corresponding `get_*` flag is `True`.

### SimulatedObjective (`feedback.py`)

Callable objective function for testing. Supports multiple built-in functions.

**Constructor args:**
- `function` (str): one of `'1D'`, `'2D'`, `'3D'`, `'2D-circle'`, `'blue'`, `'dark'`, `'vibrant'`
- `amplitude`, `freq`, `smooth` - parameters for the 1D objective
- `ord_lbls` (int or None): number of ordinal label bins
- `action_space` - required when `ord_lbls` is set

**Usage:** call the instance directly: `objective(action)` or `objective(batch_of_actions)`

### RandomSampler (`sampler.py`)

Samples actions uniformly at random from the action space.

**Key method:**
- `sample(actions)` - returns a single randomly chosen action from the array

## Typical workflow

**Regression / HILO (preferred — used by `hilo/mo.py` and the `hipexo_sim` sweeps):**

1. Create a `Regression` / `MultiObjectiveRegression` to define the action space
2. Create a `MultiObjectiveGP` and a sampler (`DSTSampler(gps=optimizer.gps, ...)`)
3. Loop: sample an action, query the oracle, `add_feedback()`, then
   `optimizer.setup(action_space, regressions=regression.get_regression_data())`,
   `optimizer.fit()`, `sampler.update_posterior()`
4. Read `optimizer.gps[i].mu` / `.std()`

`ConjugateGP.fit()` is a closed-form solve, so there is no warm start to preserve
and no JAX compilation involved.

**Preference-based (`LaplaceGP`):**

1. Create `PreferenceBasedLearning` to define the action space
2. Create a feedback source (an oracle for testing, or collect real feedback)
3. Create `LaplaceGP`; bind the current feedback into a single-argument
   likelihood, e.g. `lambda r: pbl.likelihood_from_data(r, pbl.feedback_data())`,
   and pass it to `gp.set_data(action_space, likelihood)`
4. Loop: sample actions, collect feedback, `add_feedback()`, re-run `set_data()`
   with the rebound likelihood, then `gp.fit()`
5. Use `pbl.predict()` or `pbl.optimal_action()` with `gp.mu`

`mu` persists across `fit()` calls, giving a natural warm start. Note that
`set_data()` rebuilds the JIT-compiled objective, so this path recompiles once
per iteration.

## Dependencies

- **Runtime:** numpy, scipy, scikit-learn, jax[cpu]
- **Dev:** pytest, matplotlib

## JAX notes

- `pypolar` enables `jax_enable_x64` on import for numerical stability (float32 causes overflow in sigmoid/log computations)
- All likelihood functions are vectorized (no Python loops) for JIT compatibility
- The GP's objective, gradient, and Hessian are JIT-compiled on `setup()`
- Feedback data is treated as static constants during JAX tracing — if feedback changes, re-compile and re-setup
