# Overall rules
Comment code as it is, not why it was put there or what it replaced. This applies to variable names as well. Keep comments very brief (max two sentences of description. You may allot more space for denoting variables/return types). If a section of code took 200 lines, but could be written in 50, replace it. Keep the style of the rest of the repository.

When possible, source code from highly used, verified libraries (like scipy, scikit-learn, etc) instead of writing your own code. 

When explaining things, do not assume knowledge. Explain via deduction and don't gloss over details. It should be clear from an outside observer, who may not be completely familiar with this repository, what you changed and why it is scientifically/mathematically justified.

# pypolar

Preference- and regression-based Bayesian optimization for human-in-the-loop
robotics, built around POLAR (Preference Optimization and Learning Algorithm for
Robotics). Two learning paths share one interface: a **regression** path (exact
Gaussian posteriors, used by every HILO experiment in the tree) and a
**preference** path (non-Gaussian likelihoods written in JAX, approximated by
Laplace).

## Project structure

```
pyPolar/
├── pyproject.toml                  # Package config, dependencies, pytest settings
├── src/pypolar/                    # Installable package
│   ├── __init__.py                 # Public API, enables JAX float64
│   ├── optimization/
│   │   ├── problem.py              # Likelihood, PreferenceBasedLearning, Regression,
│   │   │                           #   MultiObjectiveRegression - action space + likelihoods
│   │   └── gp/                     # GP models, split by posterior representation
│   │       ├── kernels.py          # SquaredExponential, make_kernel
│   │       ├── base.py             # GPModel ABC - kernel algebra, shared sampling
│   │       ├── conjugate.py        # ConjugateGP  - exact posterior, numpy/scipy
│   │       ├── botorch_gp.py       # BoTorchGP    - exact posterior, BoTorch/gpytorch
│   │       ├── laplace.py          # LaplaceGP    - JAX autodiff + scipy
│   │       └── multi.py            # MultiObjectiveGP
│   ├── feedback/
│   │   ├── rewards.py              # InternalReward hierarchy (groundtruth objectives)
│   │   └── oracles.py              # Simulated humans: Bradley-Terry, noisy regression
│   ├── performance/mo.py           # groundtruth_hypervolume, pareto_overlay
│   ├── utils/                      # pareto.py, plotting.py, gp.py (hyperparam heuristics)
│   └── sampler.py                  # Action sampling and acquisition functions
├── config/                         # Pydantic experiment configs (NOT part of the package)
│   ├── base.py                     # Config base + every problem/optimizer/sampler config
│   ├── aq_test.py                  # sim_1d, sim_2d, sim_3d  (single-objective)
│   └── hipexo.py                   # hipexo_sim_idealized, hipexo_sim_idealized_2d
├── hilo/                           # Runnable experiments (NOT part of the package)
│   ├── create.py                   # Config -> objects. The wiring layer.
│   ├── mo.py                       # Single multi-objective run, seed 95
│   ├── aq_1d.py                    # Single-objective acquisition comparison
│   ├── aq_mo.py                    # Multi-objective acquisition comparison
│   └── hipexo_sim/                 # Multi-trial experiment harness + plotting
├── examples/example_1d.py          # 1D preference-learning demo (currently broken, see below)
├── scripts/                        # Scratch/experimentation files (not maintained)
└── tests/                          # pytest test suite
    ├── test_public_api.py          # Every __all__ name imports
    ├── test_no_stale_references.py # No dangling references in examples/scripts/hilo
    ├── test_acquisition.py         # Acquisition functions and their closed forms
    ├── test_botorch_gp.py          # BoTorchGP, incl. parity against ConjugateGP
    └── test_config_roundtrip.py    # Configs survive JSON serialization
```

`config/` and `hilo/` are top-level directories, not part of the installed
package. Scripts under them import as `config.base` / `hilo.create`, so run them
from the repo root as modules (`python -m hilo.mo`) — `python hilo/mo.py` puts
`hilo/` on `sys.path` instead of the repo root and fails to import `hilo.create`.

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

169 tests: public API surface (49), dangling references (25), acquisition
functions and the posterior cross-covariance (51, parametrized over both
regression backends), BoTorchGP (35, including off-grid `mu_at`/`std_at`), config
round-trips (9).

## How to run experiments

```bash
conda activate pypolar
python -m hilo.mo         # one multi-objective run at seed 95, prints hv + overlay
python -m hilo.aq_1d      # single-objective acquisition comparison across seeds
python -m hilo.aq_mo      # multi-objective acquisition comparison across seeds
python -m hilo.hipexo_sim.experiment --n-trials 5 --n-queries 50
```

`hilo/mo.py` is the quickest end-to-end check that the regression path works: 200
queries in well under a second on the ConjugateGP backend.

**`examples/example_1d.py` is broken** and was already broken before the BoTorch
work: `BradleyTerryOracle.query` calls `rng.choice([0, 1], p=[p_w, p_v])` where
`p_w`/`p_v` are arrays, raising `ValueError: p must be 1-dimensional`. It is the
only remaining example and the only exercise of the preference path outside
tests, so it is worth fixing.

## Package API

All public classes are importable directly from `pypolar`:

```python
from pypolar import (
    Regression, MultiObjectiveRegression, PreferenceBasedLearning,
    ConjugateGP, BoTorchGP, LaplaceGP, MultiObjectiveGP,
    DSTSampler, QNEHVISampler, KnowledgeGradientSampler,
)
```

Importing `pypolar` enables JAX float64 mode (`jax_enable_x64`), required for
numerical stability in the likelihood functions. It also imports torch (via
BoTorch), which costs a couple of seconds of startup.

### Problems (`optimization/problem.py`)

All four classes derive from `Likelihood`, which owns the discretized action
space: `low`, `high`, `action_dims` -> `action_space`, an `(N, d)` numpy array,
plus `get_idx(action)` mapping a continuous action to its nearest grid index.

#### Regression / MultiObjectiveRegression

The workhorses. Feedback is a scalar (or vector) measurement per action.

- `Regression(low, high, action_dims, precision=1.0)`
- `MultiObjectiveRegression(low, high, action_dims, num_objs, precisions)`
- `add_feedback(action, value)` / `add_feedback(action, values)` — appends a row
  `[action_idx, *values]` to `feedback_data`, shape `(n, num_objs + 1)`.
- `get_regression_data()` (multi-objective) — per-objective
  `(idx, y, precision)` tuples, exactly what `MultiObjectiveGP.setup(regressions=...)`
  and `ConjugateGP.set_data`/`BoTorchGP.set_data` consume.
- `get_feedback_idxs()` / `get_feedback_values(obj=0)` — for plotting.

The likelihood is `precision * ||S r - y||^2`, i.e. a Gaussian NLL with
`sigma^2 = 1/(2*precision)`. **This factor of two is load-bearing** — see
`utils/gp.py:derive_precision`, and both regression backends set
`sigma2 = 1/(2*precision)`.

#### PreferenceBasedLearning

Manages preference/coactive/ordinal feedback with JAX likelihoods compatible
with `jax.grad`, `jax.jit` and `jax.hessian`.

**Constructor:** `(low, high, action_dims, preference_noise=0.01,
coactive_noise=0.01, ordinal_noise=0.01)` — the noise terms are c_p, c_c, c_o.

**Key methods:**
- `add_feedback(preference, coactive, ordinal)` — pass `None` for unused types
  - preference: `(curr_action, prev_action, curr_is_preferred: bool)`
  - coactive: `(suggested_action, curr_action, suggested_is_better: bool)`
  - ordinal: `(action, lower_bound, upper_bound)`
- `feedback_data()` — padded JAX feedback pytree for the recompilation-free path.
  Array shapes grow geometrically (capacity doubling), so they change only
  O(log n) times over a run. Needs no `compile()`.
- `likelihood_from_data(r, data)` — negative log-likelihood as a pure function of
  `(r, data)`. Because feedback is a runtime argument, a JIT'd version is reused
  across iterations and recompiles only when an array shape grows. **This is the
  path to use.**
- `overall_likelihood(r)` — same value, but with feedback baked in as a
  constant, so a JIT'd version recompiles whenever feedback changes. Requires
  `compile()` first. Legacy.
- `predict(r, a1, a2)`, `optimal_action(r)` — numpy, for reading out results.

### GP models (`optimization/gp/`)

Split by **posterior representation**. Every backend implements the same
contract, so samplers and plotting are backend-agnostic:

```
set_data(...) -> fit() / std() / posterior_cov() / posterior_cov_cross(idx)
              -> prepare_sampling() / sample_posterior(rng)
```

`posterior_cov_cross(idx)` returns the `(N, C)` block of the posterior covariance
between every action and `actions[idx]`. The base implementation slices
`posterior_cov()`; both regression backends override it with an O(N M C)
data-space form that never builds the N x N matrix.
`KnowledgeGradientSampler` calls it on every query.

`mu_at(X)` (regression backends and `MultiObjectiveGP`, not `LaplaceGP`) is the
posterior mean at **arbitrary** points, on or off the grid: `mu(x) = k(x, X_obs)
(K_XX + sigma^2 I)^-1 y`, reusing the Gram factor from `set_data`, so it involves
no refit and costs O(n M d). The discretized action space is only where `fit()`
samples this function, so use `mu_at` for anything continuous — interpolating
between buckets, or handing the objective to a continuous optimizer. It
reproduces `mu` exactly on grid points, except that `ConjugateGP` differs at the
*fed-back* actions by `JITTER * alpha_m` (~1e-5), since `set_data` adds the
nugget where test and train indices coincide and the continuous form omits it.

`std_at(X)` (same backends) is the matching posterior standard deviation,
`sigma^2(x) = k(x, x) - k(x, X_obs) (K_XX + sigma^2 I)^-1 k(X_obs, x)`, off the
same Gram factor, in O(n M^2). It likewise omits `JITTER`, which `std()` carries
in the prior variance, so the two differ by `JITTER` in variance (~5e-5 in
standard deviation) at *every* action rather than only at fed-back ones.
`BoTorchGP` applies no jitter at all, so there it reproduces `std()` exactly.

Together `mu_at` and `std_at` make any posterior-based acquisition a continuous,
differentiable function of the action, so it can be maximized off grid — the
usual two-stage scheme is a quasi-random scan to locate the basins followed by
box-constrained L-BFGS-B, which is what `scratch/bo_continuous_2d.py` does.

**Shared constructor args:** `kernel` (str or kernel object, currently only
`'squared_exp'`), `signal_variance` (float), `length_scale` (float), `rng`.

Note `signal_variance` is an **amplitude**: `k = signal_variance**2 * exp(...)`.

`GPModel.JITTER = 1e-5` is added to the prior covariance diagonal by the numpy
kernel algebra. `BoTorchGP` does not apply it (gpytorch has its own convention),
which is the sole numerical difference between the two regression backends.

#### ConjugateGP — the default regression backend

Exact GP posterior for Gaussian feedback, in numpy/scipy. Solves only the M x M
system `G = K_XX + sigma^2 I` where M is the number of feedback points, so it
never forms an N x N matrix: O(N M^2) time, O(N M) memory. `fit()` is a single
triangular solve — no optimizer, no autodiff.

- `set_data(action_space, idx, y, precision)`
- `std()` — variance diagonal directly. Prefer over `posterior_cov()`, O(N^2) memory.
- `sample_posterior(rng)` — Matheron's rule against a cached, data-independent
  N x N prior Cholesky. Essentially free after the first call, but that first
  call is O(N^3) — the dominant cost on an 8000-point grid.

#### BoTorchGP — the BoTorch/gpytorch regression backend

Same exact posterior, wrapping `botorch.models.SingleTaskGP`, so ARD and
marginal-likelihood hyperparameter fitting are configuration rather than new
numerical code. Extra constructor args:

- `fit_hypers` (bool) — refit kernel hyperparameters by marginal likelihood on
  every `fit()`, starting from `signal_variance`/`length_scale`
- `ard` (bool) — one length scale per action dimension

Everything runs on CPU in float64; numpy is the boundary in both directions.

Load-bearing implementation details, if you touch this file:

- Observation noise is pinned via `train_Yvar`, which selects
  `FixedNoiseGaussianLikelihood`, so only the kernel is ever fitted.
- `outcome_transform=None` **must** be passed explicitly. Recent BoTorch defaults
  to `Standardize`, which would silently rescale `mu`.
- `mean_module=ZeroMean()`, matching the other backends and the zero-prior-mean
  assumption `MaxValueEntropySampler` depends on. There is a TODO about
  `ConstantMean` — see the MES note under Samplers.
- **`mu`, `std()` and `posterior_cov_cross()` are computed in data space** from
  `model.covar_module`, not via `model.posterior()`. gpytorch's exact-GP
  prediction path materializes the N x N test-test block even when only the mean
  is requested, which is quadratic in the action-space size: 259 ms per `fit()`
  at N=8000, versus 0.7 ms for the data-space form. They agree to ~1e-12.
  `posterior_cov()` still goes through the model, and is deliberately left that
  way as an independent check on the data-space algebra (see
  `tests/test_botorch_gp.py::TestAgainstTheModelPosterior`).
- `sample_posterior(rng)` draws a fresh `draw_matheron_paths` per call, inside
  `torch.random.fork_rng(devices=[])` + `torch.manual_seed`, so draws are
  reproducible from the numpy generator and torch's global state is untouched.
  A path is fixed once drawn and MES draws many times per fit, so the draw
  cannot be hoisted into `prepare_sampling()` (which is a no-op here).

**Performance, measured** — the two backends are not uniformly ranked:

| Workload | ConjugateGP | BoTorchGP |
|---|---|---|
| 20x20 grid, DST, 50 queries | 0.6 ms/q | 47.9 ms/q |
| 8000-pt grid, KG(20), 50 queries | 137 ms/q | 132 ms/q |
| 8000-pt grid, Thompson, 20 queries | 144 ms/q | 65 ms/q |
| `hilo/mo.py`, 200 queries | 0.15 s | 9.5 s |

The gap is entirely `sample_posterior`: `draw_matheron_paths` costs a flat ~21 ms
per call in BoTorch module-construction overhead, independent of N and of batch
size. So sampling-driven acquisitions (DST, Thompson, MES) on small grids pay
60-80x, while on large grids BoTorchGP is equal (KG never samples) or faster
(Thompson, where ConjugateGP pays for the 8000x8000 prior Cholesky). If the
small-grid cost ever matters, the fix is a hybrid: exact data-space Matheron with
the cached prior Cholesky when `fit_hypers=False`, RFF paths otherwise.

#### LaplaceGP — the preference backend

Laplace approximation for non-Gaussian likelihoods (e.g. the sigmoid preference
likelihood). Minimizes `likelihood(r) + 0.5 r^T K^-1 r` with scipy, using exact
gradients and Hessians from JAX autodiff, and approximates the posterior
covariance by the inverse Hessian at the mode.

- Extra constructor arg: `x0_init_method` (str), currently only `'random'`
- `set_data(action_space, likelihood, quadratic=None)` — `likelihood(r)` must be
  JAX-compatible. Builds JIT-compiled objective, jacobian and hessian. Feedback
  is baked in as a constant, so `set_data()` must be re-run when feedback
  changes — and each re-run recompiles.
- `quadratic` — whether the objective has a constant Hessian. `None` auto-detects.
  A *solver shortcut*, not a different model: when it holds, `fit()` does one
  `cho_solve` instead of calling scipy, and the Laplace posterior is exact.
  Regression likelihoods are quadratic, so `LaplaceGP` and `ConjugateGP` agree to
  ~1e-12 on them — prefer a regression backend there, it is far cheaper.
- `fit(**kwargs)` — `method` and `options` go to `scipy.optimize.minimize`. Good
  methods: `'trust-constr'` (1D), `'trust-krylov'` (higher dims).

`LaplaceGP` is why `gp/base.py` and `gp/kernels.py` must stay even if a
regression backend is ever removed.

#### MultiObjectiveGP

Independent per-objective GPs over a shared action space, with per-objective
hyperparameters passed as lists.

- `MultiObjectiveGP(num_objs, kernels, signal_variances, length_scales,
  x0_init_methods, rng, backend='conjugate', fit_hypers=None, ard=None)`
- `backend` — `'conjugate'`, `'botorch'` or `'laplace'`. Whichever of
  `'conjugate'`/`'botorch'` is named is the class the regression path uses;
  `regression_backend` records it.
- `setup(action_space, likelihoods=None, regressions=None)` — pass exactly one.
  `regressions` selects the regression backend, `likelihoods` selects `LaplaceGP`.
- `gps` is built eagerly in `__init__` and **mutated in place** by `_build`, never
  reassigned, so `len(optimizer.gps)` and samplers constructed with
  `gps=optimizer.gps` stay valid before the first `setup()` and across a backend
  switch. Do not replace it with `ModelListGP` or reassignment.
- `fit(**kwargs)` returns the list of means; `std(r=None)` the list of stds.

### Feedback (`feedback/`)

**`rewards.py`** — groundtruth objectives, all `InternalReward` subclasses,
called directly (`reward(x)`) or via `.compute(x)`:

- `IdealPoint(w, delta, gamma=0.0)` — reward peaks at ideal point `w`
- `BoundedIdealPoint(w, delta, gamma=0.0, lower_bound=-1.0, upper_bound=1.0)` —
  the one the HILO configs use. `w` may be `(d,)` or a `(m, d)` stack, one row
  per objective.
- `MultiObjectiveIdealPoint(ideal_point_rewards)`, `NonStationaryIdealPoint`

**`oracles.py`** — simulated humans:

- `NoisyRegressionOracle(reward_fn, noise_std, rng)` — `query(x)` returns the
  reward plus Gaussian noise. The regression/HILO path.
- `BradleyTerryOracle(beta_boltzmann, reward_fn, rng)` — `query(w, v)` returns 0
  or 1, a noisy preference. The preference path.
- `MultiObjectiveOracle(...)` — multi-objective preferences.

### Samplers (`sampler.py`)

Every sampler implements `sample(actions)` -> a single action, and
`update_posterior()`, called after each `gp.fit()`.

- `RandomSampler(rng)` — uniform random.
- `UniformSampler(n, rng, ...)` — Sobol sequence snapped to the discrete actions.
- `ThompsonSampler(gp, rng)` — argmax of a posterior draw.
- `DSTSampler(gps, rng, rho)` — dueling scalarized Thompson sampling,
  multi-objective. One posterior draw per objective, normalized, then scalarized
  against a Dirichlet weight vector.
- `QNEHVISampler(gps, rng, ref_point=None, num_samples=128, max_batch_size=1024)`
  — noisy expected hypervolume improvement, multi-objective. Wraps BoTorch's
  `qLogNoisyExpectedHypervolumeImprovement` over a `ModelListGP` of the
  per-objective models, enumerated across the discrete grid with
  `optimize_acqf_discrete`. **Requires the `BoTorchGP` backend** — the other
  backends expose no BoTorch model. `ref_point=None` infers it from the feedback
  collected so far.

`AcquisitionSampler` is the base for single-objective samplers that score every
action and take the argmax, with uniform tie-breaking. Until the GP is fit and
`update_posterior()` has been called — and for the first `n_warmup` queries —
they fall back to uniform random. Subclasses implement `acquisition(actions)`:

- `ExpectedImprovementSampler(gp, rng, xi=0.0)` — improvement over the best
  posterior mean. `xi` trades off exploration.
- `KnowledgeGradientSampler(gp, rng, noise_var=None, num_candidates=None)` —
  exact discrete KG, the expected increase in `max(mu)` from one observation.
  Costs O(C N log N); use `num_candidates` to subsample on large action spaces.
  `noise_var` defaults to `gp.sigma2`, which only the regression backends define.
- `MaxValueEntropySampler(gp, rng, num_maxima=32, noise_var=None)` — mutual
  information with the maximum reward value, using posterior-sample maxima as
  samples of `f*`.

EI and KG may return the same action more than once; repeat observations average
the noise down.

Module-level helpers: `expected_max_of_lines(a, b)` (exact
`E_Z[max_i(a_i + b_i Z)]`) and `knowledge_gradient(mu, sigma_tilde)`.

**Note:** every GP here has a zero prior mean. On an objective that is not
centered near zero, the posterior mean over unexplored actions sits far below the
sampled `f*`, which drives `MaxValueEntropySampler` to exploit the known peak and
never explore. Center the feedback values before fitting if you want MES to
behave. (A fitted `ConstantMean` on `BoTorchGP` would also fix this — see the
TODO in `botorch_gp.py`.)

### Metrics (`performance/mo.py`, `utils/pareto.py`)

- `groundtruth_hypervolume(estimated_objs, true_objs, tol=0.0)` — groundtruth
  hypervolume attained under the *predicted* Pareto set, normalized by the true
  optimum. 1.0 means the predicted front is as good as the real one.
- `pareto_overlay(estimated_objs, true_objs, tol=0.0)` — Jaccard overlap between
  the estimated and true fronts. Stricter: it penalizes getting the right
  hypervolume via the wrong actions.
- `tol` relaxes non-domination by a fraction of each objective's range, so
  near-ties stay on the front. The experiments use `tol=0.02`.

### Hyperparameter heuristics (`utils/gp.py`)

`derive_gp_hyperparams(domain_size, expected_range, noise_var)` ->
`(lengthscale, prior_variance, precision)` = `(0.2 * domain, 0.5 * range,
1/(2*noise_var))`. `hilo/aq_1d.py` and `hilo/aq_mo.py` use it to keep the GP
matched to the oracle. The `hipexo_sim` noise sweep deliberately does *not* —
it holds precision fixed as a misspecification study.

## Configs (`config/`)

Pydantic models with numpy support and JSON round-tripping
(`to_json_string` / `from_json_string`, used by `hipexo_sim/experiment.py` to
record what was run in each `run.json`).

- Problem: `Regression`, `MultiObjectiveRegression`
- Optimizer: `GaussianProcess`, `MultiObjectiveGaussianProcess` — both carry
  `gptype` (`'ConjugateGP'` | `'BoTorchGP'` | `'LaplaceGP'`, validated against
  `GPTYPES`) plus `fit_hypers`/`ard` for the BoTorch backend
- Sampler: `RandomSampling`, `ThompsonSampling`, `DSTS`, `QNEHVI`,
  `ExpectedImprovement`, `KnowledgeGradient`, `MaxValueEntropy`
- Oracle/objective: `BoundedIdealPoint`, `NoisyRegressionOracle`
- Top level: `HILO` (single-objective), `MOHILO` (multi-objective)

`HILO.sampler` and `MOHILO.sampler` are **unions**, and `Config._decode_field`
resolves a union by matching the encoded keys against each member's fields
(exact match first, then largest overlap). Members with identical field sets —
`RandomSampling` and `ThompsonSampling`, both empty — remain ambiguous on
decode; the first declared wins. Adding a field to one of them would fix it.
`tests/test_config_roundtrip.py` guards this.

`hilo/create.py` is the only place configs become objects:
`create_1d_sim(rng, cfg)` and `create_hipexo_sim(rng, cfg)`, with
`_make_sampler` / `_make_mo_sampler` dispatching on the sampler config type and
`BACKENDS` mapping `gptype` to the `MultiObjectiveGP` backend key.

## Typical workflow

**Regression / HILO — the path everything in `hilo/` uses:**

1. Create a `Regression` / `MultiObjectiveRegression` to define the action space
2. Create a `MultiObjectiveGP` and a sampler (`DSTSampler(gps=optimizer.gps, ...)`)
3. Loop: sample an action, query the oracle, `add_feedback()`, then
   `optimizer.setup(action_space, regressions=regression.get_regression_data())`,
   `optimizer.fit()`, `sampler.update_posterior()`
4. Read `optimizer.gps[i].mu` / `.std()`

Both regression backends are closed-form solves, so there is no warm start to
preserve and no JAX compilation involved. Calling `setup()` every iteration is
cheap and is the intended usage.

**Preference-based (`LaplaceGP`):**

1. Create `PreferenceBasedLearning` to define the action space
2. Create a feedback source (a `BradleyTerryOracle` for testing, or real feedback)
3. Create `LaplaceGP`; bind the current feedback into a single-argument
   likelihood, e.g. `lambda r: pbl.likelihood_from_data(r, pbl.feedback_data())`,
   and pass it to `gp.set_data(action_space, likelihood)`
4. Loop: sample actions, collect feedback, `add_feedback()`, re-run `set_data()`
   with the rebound likelihood, then `gp.fit()`
5. Use `pbl.predict()` or `pbl.optimal_action()` with `gp.mu`

`mu` persists across `fit()` calls, giving a natural warm start. `set_data()`
rebuilds the JIT-compiled objective, so this path recompiles once per iteration.

## Dependencies

- **Runtime:** numpy, scipy, scikit-learn, jax[cpu], pymoo, botorch
  (which pulls torch and gpytorch)
- **Dev:** pytest, matplotlib
- Experiments additionally use tqdm and pydantic

Verified against botorch 0.18.1 / gpytorch 1.15.2 / torch 2.13.0. Several details
in `botorch_gp.py` are version-sensitive: the `outcome_transform` default,
`GPyTorchPosterior.distribution` (an instance attribute, not a class one), and
the `draw_matheron_paths` import path.

## JAX notes

Only the preference path uses JAX; the regression path is numpy/scipy or torch.

- `pypolar` enables `jax_enable_x64` on import (float32 overflows in the
  sigmoid/log computations)
- All likelihood functions are vectorized (no Python loops) for JIT compatibility
- `LaplaceGP`'s objective, gradient and Hessian are JIT-compiled on `set_data()`
- Feedback is a static constant during tracing, which is why
  `likelihood_from_data` + `feedback_data()` exists: it moves feedback into a
  runtime argument so the JIT cache survives across iterations

## Known stale / broken code

Not worth trusting without checking first:

- `examples/example_1d.py` — raises, see above
- `scripts/kernel_test.py` — `from gp import BasicGP`, a module that no longer
  exists. `test_no_stale_references.py` only validates `pypolar.*` references, so
  a bare `from gp import ...` slips past it.
- `hilo/scalarized.py` and `scripts/gp_pareto.py` — the only remaining users of
  `LaplaceGP` outside tests, so they exercise the preference path
- `scripts/` generally — scratch work, not maintained

`hilo/hipexo_sim/frontier_length.py` and `hilo/hipexo_sim/_action_size.py` are
near-identical copies of each other that both drive the current
`create_hipexo_sim` path; `action_size.py` sits alongside `_action_size.py`. They
are not stale, but the duplication means a change to one silently leaves the
others behind.
