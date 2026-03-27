# pypolar

Python implementation of POLAR (Preference Optimization and Learning Algorithm for Robotics) for preference-based learning. Uses JAX for automatic differentiation and JIT compilation of likelihood functions.

## Project structure

```
pypolar/
├── pyproject.toml              # Package config, dependencies, pytest settings
├── src/pypolar/                # Installable package
│   ├── __init__.py             # Public API, enables JAX float64
│   ├── pbl.py                  # PreferenceBasedLearning - core preference logic (JAX likelihoods)
│   ├── gp.py                   # BasicGP - Gaussian process reward model (JAX autodiff)
│   ├── feedback.py             # SimulatedFeedback, SimulatedObjective - feedback simulation
│   └── sampler.py              # RandomSampler - action sampling
├── examples/                   # Runnable example scripts
│   ├── example_1d.py           # 1D preference learning (verification script)
│   ├── example_2d.py           # 2D preference learning with 3D plots
│   └── example_3d.py           # 3D (RGB color) preference learning
├── scripts/                    # Scratch/experimentation files (not part of package)
└── tests/                      # pytest test suite
    ├── test_pbl.py             # Tests for PreferenceBasedLearning + JAX compatibility
    ├── test_gp.py              # Tests for BasicGP
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

`example_1d.py` is the go-to script for verifying the package works end-to-end. It collects 20 pairwise comparisons, fits a GP, evaluates prediction accuracy (~90%), and shows a matplotlib plot.

## Package API

All public classes are importable directly from `pypolar`:

```python
from pypolar import PreferenceBasedLearning, BasicGP, SimulatedFeedback, SimulatedObjective, RandomSampler
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
- `compile()` - **must be called after adding feedback and before fitting/differentiating.** Converts feedback lists into JAX arrays. Clears compiled state when new feedback is added.
- `overall_likelihood(r)` - compute total negative log-likelihood given reward vector `r`. Compatible with `jax.grad` and `jax.jit`.
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

### BasicGP (`gp.py`)

Gaussian process model that learns the latent reward function. Uses JAX autodiff to automatically derive exact gradients and Hessians for scipy optimization — no manual derivative implementation needed.

**Constructor args:**
- `kernel` (str): kernel type, currently only `'squared_exp'`
- `signal_var` (float): signal variance (controls reward magnitude)
- `lengthscale` (float): lengthscale (controls smoothness)
- `mu_init_method` (str): mean initialization, currently only `'random'`

**Key methods:**
- `setup(action_space, likelihood)` - initialize GP with action space and compiled PBL likelihood. Automatically creates JIT-compiled objective, jacobian, and hessian via `jax.grad` and `jax.hessian`. The scipy-compatible wrappers handle numpy/jax conversion.
- `fit(**kwargs)` - fit the GP via `scipy.optimize.minimize`. Always provides exact gradient and Hessian. Pass `method` and `options` as kwargs. Good methods: `'trust-constr'` (1D), `'trust-krylov'` (higher dims).
- `functionalize(degree=2)` - fit a polynomial to the GP mean for continuous evaluation

**Key attributes:**
- `mu` - the current reward mean vector (numpy array of length N)
- `actions` - the action space passed during setup
- `objective` / `jacobian` / `hessian` - scipy-compatible wrappers (numpy in/out)
- `_jax_objective` / `_jax_jacobian` / `_jax_hessian` - JIT-compiled JAX functions

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

1. Create `PreferenceBasedLearning` to define the action space
2. Create a feedback source (`SimulatedFeedback` for testing, or collect real feedback)
3. Loop: sample actions, collect feedback via `evaluate()`, add via `add_feedback()`
4. Call `pbl.compile()` to convert feedback into JAX arrays
5. Create `BasicGP` and call `setup()` with the PBL's action space and `overall_likelihood`
6. Call `gp.fit()` to learn the reward function (uses JAX autodiff for exact gradients)
7. Use `pbl.predict()` or `pbl.optimal_action()` with `gp.mu`

**Important:** `compile()` must be called after all `add_feedback()` calls and before `gp.setup()`. If new feedback is added after compilation, call `compile()` again and re-run `gp.setup()`.

## Dependencies

- **Runtime:** numpy, scipy, scikit-learn, jax[cpu]
- **Dev:** pytest, matplotlib

## JAX notes

- `pypolar` enables `jax_enable_x64` on import for numerical stability (float32 causes overflow in sigmoid/log computations)
- All likelihood functions are vectorized (no Python loops) for JIT compatibility
- The GP's objective, gradient, and Hessian are JIT-compiled on `setup()`
- Feedback data is treated as static constants during JAX tracing — if feedback changes, re-compile and re-setup
