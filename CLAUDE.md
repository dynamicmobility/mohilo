# Overall rules
Comment code as it is, not why it was put there or what it replaced. This applies to variable names as well. Keep comments very brief (max two sentences of description. You may allot more space for denoting variables/return types). If a section of code took 200 lines, but could be written in 50, replace it. Keep the style of the rest of the repository.

When possible, source code from highly used, verified libraries (like scipy, scikit-learn, etc) instead of writing your own code.

When explaining things, do not assume knowledge. Explain via deduction and don't gloss over details. It should be clear from an outside observer, who may not be completely familiar with this repository, what you changed and why it is scientifically/mathematically justified.

Always ask how much code the user wants you to edit. Do not edit more than they ask you to without asking them first. Always pose this question before you implement something.

# pypolar

Multi-objective Bayesian optimization for human-in-the-loop robotics. The GP
layer is **BoTorch and nothing else** — there is no hand-written kernel algebra,
no custom posterior solver, and no custom acquisition code in the package. One
class, `DecoupledMOGP`, wraps a BoTorch `ModelListGP`.

## Project structure

```
pyPolar/
├── pyproject.toml                  # Package config, dependencies, pytest settings
├── src/pypolar/                    # Installable package
│   ├── __init__.py                 # Public API
│   ├── optimization/
│   │   ├── objectives.py           # AffineTransform, Objective, DecoupledObjectives
│   │   └── gp.py                   # DecoupledMOGP - the only GP in the repo
│   ├── feedback/
│   │   ├── rewards.py              # InternalReward hierarchy (groundtruth objectives)
│   │   └── oracles.py              # Simulated humans: Bradley-Terry, noisy regression
│   ├── performance/mo.py           # groundtruth_hypervolume, pareto_overlay
│   └── utils/                      # pareto.py, plotting.py
├── hilo/
│   ├── plot_pilot.py               # The one live experiment: pilot data -> fronts + optima
│   └── output/                     # Figures and recorded runs
├── scripts/                        # Scratch/experimentation files (not maintained)
├── human_data/                     # Pilot CSVs consumed by plot_pilot.py
└── tests/                          # pytest test suite
    ├── test_public_api.py          # Every __all__ name imports
    ├── test_no_stale_references.py # No dangling references in scripts/hilo
    ├── test_objectives.py          # AffineTransform, Objective, DecoupledObjectives
    └── test_decoupled_mogp.py      # DecoupledMOGP behaviour
```

`hilo/` is a top-level directory, not part of the installed package. Run its
scripts from the repo root as modules (`python -m hilo.plot_pilot`) — `python
hilo/plot_pilot.py` puts `hilo/` on `sys.path` instead of the repo root.

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

126 tests: public API surface (29), dangling references (8), objectives (67),
`DecoupledMOGP` (22).

## How to run experiments

```bash
conda activate pypolar
python -m hilo.plot_pilot   # fits the pilot data, writes hilo/output/test.png,
                            # prints the best action per objective
```

`plot_pilot.py` needs `pandas`, which is not a package dependency (it is an
experiment-only import). `optimize_acqf` seeds its restarts from torch's global
state, which the script does not set, so the reported optima wobble in the third
or fourth decimal between runs.

## Package API

```python
from pypolar import DecoupledMOGP, DecoupledObjectives, Objective, AffineTransform
```

Everything runs on CPU in float64 (`pypolar.optimization.gp.DTYPE`). numpy is the
boundary in both directions: every public method takes and returns numpy arrays,
and torch never escapes the module.

### Objectives (`optimization/objectives.py`)

The bookkeeping layer. It owns the two coordinate changes that the GP assumes
have already happened, so nothing downstream has to think about units or signs.

**`AffineTransform(scale, shift)`** — `x -> (x + shift) * scale`, with `inv()`.
Three constructors: `make_standardized` (zero mean, unit variance),
`make_centered` (zero mean), `make_normalized` (mapped onto `[0, 1]`). Each
guards against zero-variance/zero-range data by leaving it unscaled.

**`Objective(name, maximize, ydata, xdata, column=None)`** — one measured
objective: `ydata` is `(N,)`, `xdata` is `(N, K)`. On construction it builds

```
ytransform = AffineTransform(scale=sign, shift=-mean(ydata))
centered_y = ytransform(ydata)
```

where `sign` is `+1` for `maximize=True` and `-1` otherwise. So `centered_y` is
zero-mean **and larger-is-better regardless of the objective's direction**. This
is what lets the GP layer ignore minimization entirely: a minimized cost has
already been negated by the time a GP sees it, and `ytransform.inv()` puts a
prediction back into the objective's own units for reporting.

Built with `from_df(df, column, action_columns, maximize, name)` or
`from_data(actions, values, maximize, name)`. `add_points()` appends and re-runs
`__post_init__`, so the transform tracks the new mean.

**`DecoupledObjectives(objectives)`** — a list of `Objective`s that are
*decoupled*: each carries its own actions and its own number of measurements.
Nothing requires them to share a design. What they do share is one action frame:

```
xtransform = AffineTransform.make_normalized(concat(all xdata), axis=0)
```

a single per-dimension normalization spanning every objective's actions, so
actions from different objectives land in the same `[0, 1]^d` box and one set of
GP lengthscales is meaningful across all of them.

Selectors (`objs`) accept a name, an index, a list of either, a slice, or `None`
for all. `__getitem__` returns a bare `Objective` for a single selector and a new
`DecoupledObjectives` for a multi-selector, so `objectives[['Cost', 'Comfort']]`
is how you fit a GP to a subset. `feedback(objs)` returns `centered_y`,
`actions(objs)` returns normalized actions, and `max`/`min`/`range` report the
centered values.

### DecoupledMOGP (`optimization/gp.py`)

Independent per-objective GPs over that shared action frame — a BoTorch
`ModelListGP` of `SingleTaskGP`s, one per objective. "Decoupled" is the modelling
claim: the objectives are assumed independent, so there is no cross-objective
covariance and the joint marginal likelihood factorizes.

```python
mogp = DecoupledMOGP(objectives, fit_hyperparameters=True, noise_std=0.05)
mu, std     = mogp.posterior_at(X)          # (n, m) each
best, mu, s = mogp.best_actions()           # (m, d), (m,), (m,)
mogp.update_feedback(objectives)            # rebuild after new measurements
```

**Class attributes** `LENGTH_SCALE = 0.2` (20% of the unit action box) and
`SIGNAL_VAR = 1.0` are the *starting* hyperparameters. Each sub-model is an ARD
`ScaleKernel(RBFKernel)` — one lengthscale per action dimension — with
`ZeroMean()` and `Standardize(m=1)`.

`mean_module=ZeroMean()` is correct rather than assumed: `Standardize` centers
the training targets, so the prior mean of the standardized values genuinely is
zero.

**Noise.** `noise_std` is a fraction of each objective's *own* standard
deviation, not a raw magnitude. It is pinned via `train_Yvar` (which selects
`FixedNoiseGaussianLikelihood`, so only the kernel is ever fitted) as
`(noise_std * spread)^2`. The pre-multiplication by `spread` is load-bearing:
`Standardize` divides `train_Yvar` by the same variance it divides `train_Y` by,
so scaling first is what makes the parameter scale-free.

**`fit_hyperparameters`** fits lengthscales and signal variances by marginal
likelihood in one `fit_gpytorch_mll` call on a `SumMarginalLogLikelihood`. The
sum splits over the sub-models precisely because the objectives are independent,
so a single call fits all of them.

**`posterior_at(action, normalized=False, chunk=2048)`** — posterior mean and
standard deviation at *arbitrary* actions, on or off any grid. `normalized=False`
(the default) maps raw actions through `objectives.xtransform` first; pass
`normalized=True` if they are already in the `[0, 1]^d` box. Returns `(n, m)`
arrays in **maximization space** — larger-is-better and centered, matching
`objectives.feedback()`. Use `objectives[name].ytransform.inv()` to report in raw
units.

Actions are fed as `q=1` batch elements (`unsqueeze(1)`), so each posterior is
1x1 per objective and gpytorch never materializes the `n x n` test-test block —
that path is quadratic in the number of evaluation points and dominates the cost
on a large scan. `chunk` bounds the per-call batch.

**`best_actions(num_restarts=8, raw_samples=512)`** — the action maximizing each
objective's posterior mean, over the **continuous box**, not over the measured
actions. Two stages, which is what `optimize_acqf` does internally: a Sobol scan
of `raw_samples` points to locate the basins, then box-constrained L-BFGS-B from
the `num_restarts` best starting points. One single-output problem per objective.

No sign handling appears here, and none is needed — `feedback()` is already
flipped to larger-is-better, so the argmax of the posterior *is* the optimum for
a minimized objective too. Returns actions in raw units (via `xtransform.inv`),
plus the diagonal of the posterior evaluated at those actions.

**`update_feedback(objectives)`** rebuilds every sub-model from scratch and
refits. There is no warm start and no incremental conditioning; a closed-form
exact GP is cheap enough that rebuilding is the intended usage.

### Feedback (`feedback/`)

Simulated humans and groundtruth objectives, all numpy, no GP dependency. Kept
for simulation work; `plot_pilot.py` uses real data and touches none of it.

**`rewards.py`** — `InternalReward` subclasses, called as `reward(x)` or
`.compute(x)`: `IdealPoint(w, delta, gamma=0.0)`,
`BoundedIdealPoint(w, delta, gamma=0.0, lower_bound=-1.0, upper_bound=1.0)`,
`MultiObjectiveIdealPoint`, `NonStationaryIdealPoint`.

**`oracles.py`** — `NoisyRegressionOracle(reward_fn, noise_std, rng)` returns a
reward plus Gaussian noise; `BradleyTerryOracle(beta_boltzmann, reward_fn, rng)`
returns a noisy preference; `MultiObjectiveOracle` the multi-objective version.

### Metrics (`performance/mo.py`, `utils/pareto.py`)

- `get_nondominated(F)` — indices of the non-dominated front of `F` (higher is
  better), via pymoo.
- `get_nondominated_tol(F, tol=0.0)` — same, but a point is dropped only when
  another beats it by more than `tol` *in every objective*, with `tol` a fraction
  of each objective's range. Keeps near-ties on the front. `tol=0` defers to
  `get_nondominated`.
- `groundtruth_hypervolume(estimated_objs, true_objs, tol=0.0)` — groundtruth
  hypervolume attained under the *predicted* Pareto set, normalized by the true
  optimum. 1.0 means the predicted front is as good as the real one.
- `pareto_overlay(estimated_objs, true_objs, tol=0.0)` — Jaccard overlap between
  the estimated and true fronts. Stricter: it penalizes getting the right
  hypervolume via the wrong actions.

### Plotting (`utils/plotting.py`)

`plot_gp_1d(ax, mu, std, action_space, feedback_idxs, feedback_values, ...)` —
takes plain arrays, so it is independent of any GP class.

## Typical workflow

1. Build an `Objective` per measured quantity, each with `maximize=True/False`,
   and collect them in a `DecoupledObjectives`.
2. Select the subset you want to model: `objectives[['Cost', 'Comfort']]`.
3. `DecoupledMOGP(subset, fit_hyperparameters=True)`.
4. Read the posterior with `posterior_at`, the optima with `best_actions`, and
   the front with `get_nondominated` over a scan of `posterior_at`.
5. After new measurements: `objectives.add_point(...)` then
   `mogp.update_feedback(objectives)`.

## Dependencies

- **Runtime:** numpy, scipy, pymoo, botorch (which pulls torch and gpytorch)
- **Dev:** pytest, matplotlib
- `hilo/plot_pilot.py` additionally uses pandas

Verified against botorch 0.18.1 / gpytorch 1.15.2 / torch 2.13.0. Two details in
`gp.py` are version-sensitive: `outcome_transform` must be passed explicitly
(BoTorch's default changed to `Standardize`), and `train_Yvar` is what selects
`FixedNoiseGaussianLikelihood`.

## Known stale code

`scripts/` is scratch work and is not maintained. Nothing in it imports
`pypolar`, so `test_no_stale_references.py` has nothing to check there.
