# Overall rules
Comment code as it is, not why it was put there or what it replaced. This applies to variable names as well. Keep comments very brief (max two sentences of description. You may allot more space for denoting variables/return types). If a section of code took 200 lines, but could be written in 50, replace it. Keep the style of the rest of the repository.

When possible, source code from highly used, verified libraries (like scipy, scikit-learn, etc) instead of writing your own code.

When explaining things, do not assume knowledge. Explain via deduction and don't gloss over details. It should be clear from an outside observer, who may not be completely familiar with this repository, what you changed and why it is scientifically/mathematically justified.

Always ask how much code the user wants you to edit. Do not edit more than they ask you to without asking them first. Always pose this question before you implement something.

# pypolar

Multi-objective Bayesian optimization for human-in-the-loop robotics. The GP
layer is **BoTorch and nothing else** — there is no hand-written kernel algebra,
no custom posterior solver, and no custom acquisition code in the package. One
factory, `build_botorch_gp`, configures every `SingleTaskGP` in the repo, and two
wrappers use it: `BoTorchGP` (one objective, one `SingleTaskGP`) and
`DecoupledMOGP` (many objectives, a `ModelListGP`).

## Project structure

```
pyPolar/
├── pyproject.toml                  # Package config, dependencies, pytest settings
├── src/pypolar/                    # Installable package
│   ├── __init__.py                 # Public API
│   ├── optimization/
│   │   ├── objectives.py           # AffineTransform, Objective, DecoupledObjectives
│   │   └── gp.py                   # build_botorch_gp, gp_hyperparameters, BoTorchGP, DecoupledMOGP
│   ├── feedback/
│   │   ├── rewards.py              # InternalReward hierarchy (groundtruth objectives)
│   │   └── oracles.py              # Simulated humans: Bradley-Terry, noisy regression
│   ├── performance/mo.py           # groundtruth_hypervolume, pareto_overlay
│   └── utils/                      # pareto.py, plotting.py
├── hilo/
│   ├── plot_pilot.py               # The one live experiment: pilot data -> fronts + optima
│   ├── gp_diagnostics.py           # Leave-one-out R^2 and calibration on a synthetic objective
│   ├── gp_accuracy.py              # The same leave-one-out, on the measured subject data
│   ├── read_data.py                # Pilot and MT0x CSVs -> Objective / DecoupledObjectives
│   └── output/                     # Figures and recorded runs
├── scratch/                        # Scratch/experimentation files (not maintained)
├── docs/                           # LaTeX writeup and handoff record from the pre-BoTorch GP
├── tablet/                         # iPad control panel (panel.py + index.html), no pypolar import
├── human_data/                     # Pilot CSVs consumed by plot_pilot.py
└── tests/                          # pytest test suite
    ├── test_public_api.py          # Every __all__ name imports
    ├── test_no_stale_references.py # No dangling references in scripts/hilo
    ├── test_objectives.py          # AffineTransform, Objective, DecoupledObjectives
    ├── test_botorch_gp.py          # build_botorch_gp, gp_hyperparameters, BoTorchGP behaviour
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

315 tests: public API surface (32), dangling references (5), objectives (131),
`BoTorchGP` (100), `DecoupledMOGP` (47).

## How to run experiments

```bash
conda activate pypolar
python -m hilo.plot_pilot        # fits the pilot data, writes hilo/output/test.png,
                                 # prints the best action per objective
python -m hilo.gp_diagnostics    # leave-one-out R^2 and calibration on a synthetic
                                 # objective, writes hilo/output/gp_diagnostics.png
python -m hilo.gp_accuracy       # the same leave-one-out, on a subject's CSV
```

`plot_pilot.py` needs `pandas`, which is not a package dependency (it is an
experiment-only import). `optimize_acqf` seeds its restarts from torch's global
state, which the script does not set, so the reported optima wobble in the third
or fourth decimal between runs.

`gp_diagnostics.py` needs `scikit-learn` and `matplotlib`, also experiment-only.
Unless `--refit-folds` is passed, every leave-one-out fold freezes the full-data
fit's hyperparameters, which it reads with `BoTorchGP.get_fitted_hyperparameters()`
— including the noise, so a fitted noise freezes across folds exactly as the
kernel does.

Both scripts fit the noise under a prior by default. `gp_diagnostics.py`'s
`--gp-noise` takes a fraction of the objective's spread to pin, `match` to use
the oracle's own noise (only a simulation knows it, so `gp_accuracy.py` has no
such option), `fit` to fit it unpriored, or `prior:MEDIAN`. `gp_accuracy.py` sets
its `GP_NOISE` constant to a `NoiseModel` directly. `--min-lengthscale` floors
the lengthscales; the right floor is the design spacing, so `gp_diagnostics.py`
leaves it off by default because its `--samples` and `--dim` flags change it.

Worth knowing before reading a number off either: on a 1D Levy with 15 points and
25% added noise, the fitted noise collapses to the likelihood floor and in-sample
R² reaches exactly 1.0 — the GP interpolates and calls the noise zero. Fitting
the noise is not a substitute for a design that can identify it.

## Package API

```python
from pypolar import DecoupledMOGP, BoTorchGP, NoiseModel
from pypolar import DecoupledObjectives, Objective, AffineTransform
from pypolar import sample_actions
```

Everything runs on CPU in float64 (`pypolar.optimization.gp.DTYPE`). numpy is the
boundary in both directions: every public method takes and returns numpy arrays,
and torch never escapes the module. Two arguments are the exception, both inputs
only: `Objective.from_synthetic`'s `function`, a BoTorch `SyntheticTestFunction`
subclass, and `sample_actions`'s `bounds`, a torch tensor. Both are consumed
internally and only numpy comes back out.

### Objectives (`optimization/objectives.py`)

The bookkeeping layer. It owns the two coordinate changes that the GP assumes
have already happened, so nothing downstream has to think about units or signs.

**`sample_actions(bounds, n, kind, seed)`** — `n` actions over a box, returned
`(n, d)`. `bounds` is a `(2, d)` **torch tensor** of `[lower; upper]` rows, the
form BoTorch states bounds in; a numpy array raises. `kind='sobol'` draws a
space-filling Sobol sequence and `kind='uniform'` draws iid uniform points; the
branch is a whitelist, so any other value raises `ValueError` rather than
silently returning a design you did not ask for. Sobol is the default worth
reaching for because a 2^k-point sequence splits every axis exactly in half,
where an iid design leaves clumps and gaps at the sample sizes these experiments
run at.

**`AffineTransform(scale, shift)`** — `x -> (x + shift) * scale`, with `inv()`.
Three constructors: `make_standardized(data, sign=1.0)` (zero mean, unit
variance, times `sign`), `make_centered` (zero mean), `make_normalized` (mapped
onto `[0, 1]`). Each guards against zero-variance/zero-range data by leaving it
unscaled. `inv_scale()` is the inverse for a *spread*: it undoes the scaling but
not the shift, since a shift does not move a standard deviation and a sign flip
cannot make one negative. Means go back through `inv()`, standard deviations
through `inv_scale()`.

**`Objective(name, maximize, ydata, xdata, column=None)`** — one measured
objective: `ydata` is `(N,)`, `xdata` is `(N, K)`. On construction it builds

```
ytransform = AffineTransform.make_standardized(ydata, sign=sign)
standard_y = ytransform(ydata)          # sign * (y - mean) / std
```

where `sign` is `+1` for `maximize=True` and `-1` otherwise. So `standard_y` is
zero-mean, unit-variance, **and larger-is-better regardless of the objective's
direction**. Two things downstream depend on this:

1. **The GP layer can ignore minimization entirely** — a minimized cost has
   already been negated by the time a GP sees it, and `ytransform.inv()` puts a
   prediction back into the objective's own units for reporting.
2. **Objectives are comparable across axes.** Metabolic cost, walk time, and the
   comfort ratings have ranges differing by orders of magnitude. Dividing by each
   objective's own standard deviation is what makes a shared hyperparameter prior
   or a hypervolume reference point meaningful; centering alone would let the
   widest objective dominate any volume computed in objective space.

Built with `from_df(df, column, action_columns, maximize, name)` or
`from_data(actions, values, maximize, name)`. `add_points()` appends and re-runs
`__post_init__`, so the transform tracks the new mean.

`from_synthetic(function, actions, maximize, rel_noise_std, seed, name)` builds
one from a BoTorch `SyntheticTestFunction` **subclass** (not an instance),
evaluated at `actions` with Gaussian noise added. `rel_noise_std` is a fraction
of the truth's own spread, matching the convention `noise_std` uses everywhere
else in the package, so it means the same difficulty across functions whose
ranges differ by orders of magnitude. The noise is drawn from an explicit
`default_rng(seed)` rather than through the function's own `noise_std`, which
would be an absolute magnitude drawn from torch's global state.

`negate` is deliberately never passed: `Objective` already encodes direction
through `maximize`, so `from_synthetic(Levy, ..., maximize=False)` means
"minimize Levy" and negating too would double-flip it.

The bounds handed to the function span **both** the action range and the
function's own default box. Both halves are load-bearing, and neither is
optional in botorch 0.18.1: `evaluate_true` rejects actions outside the bounds,
so they must cover the data (StyblinskiTang's default box is `[-5, 5]`, and
actions on `[-8, 8]` fail without custom bounds); and passing *any* custom
bounds triggers a check that at least one known optimizer lies inside them, so
they must also cover the default box or a design that misses the optimum raises
`ValueError`. Widening is free — bounds gate the domain, they never enter the
formula, so the values are identical whichever bounds are used.

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
is how you fit a GP to a subset. `feedback(objs)` returns `standard_y`,
`actions(objs)` returns normalized actions, and `max`/`min`/`range` report the
standardized values.

`to_raw(mu, std=None, objs=None)` is the way back out: it maps posterior moments
from maximization space into each objective's own units, column by column, using
`inv()` on the means and `inv_scale()` on the standard deviations. This is what
`posterior_at(raw=True)` and `best_actions(raw=True)` call.

### The GP layer (`optimization/gp.py`)

Every GP in the repo comes out of one factory. `build_botorch_gp(actions, values,
noise, signal_var, length_scale, min_length_scale=None)` returns the single
`SingleTaskGP` configuration the package uses: an ARD `ScaleKernel(RBFKernel)`,
`ZeroMean()`, and `Standardize(m=1)`. Both wrappers below call it, so a change
there changes every GP at once.

**`NoiseModel`** is how the noise is expressed — one argument carrying three
representations, so a pinned level and a prior on a fitted one cannot be asked
for at the same time. It is frozen, and coerces the shorthands, so every call
site that passed a bare number still works:

```python
NoiseModel.pinned(0.05)          # train_Yvar, never fitted   (same as noise=0.05)
NoiseModel.fitted()              # free, unpriored            (same as noise=None)
NoiseModel.prior(0.3, sigma=1.0) # fitted under a LogNormal centered on 0.3
```

- **`pinned`** goes through `train_Yvar`, which selects
  `FixedNoiseGaussianLikelihood`, so a fit only ever moves the kernel.
- **`fitted`** attaches a bare `GaussianLikelihood` whose `noise` is learned. The
  likelihood is passed explicitly for a reason: BoTorch's own default carries a
  `LogNormalPrior` centered low, and on a small design that prior, not the data,
  decides the answer — on 45 points of `MT05_incline` the default fits `0.081`
  where a prior-free one fits `0.387`.
- **`prior`** is the conventional choice, and what BoTorch's default does. It
  regularizes without forbidding: a free fit on a small design can collapse onto
  interpolation, thread every measurement, call the residual zero, and put its
  argmax on whichever point drew the luckiest noise. `median` is on the noise
  *standard deviation*, so it becomes a `LogNormalPrior(loc=2*log(median))` on the
  variance the likelihood actually holds. `sigma` is the width in log space, on
  the variance; 1.0 matches BoTorch's own default width.

**`min_length_scale`** floors every ARD lengthscale via a `GreaterThan`
constraint. GPyTorch enforces it by reparameterization (`softplus(raw) + bound`)
rather than by clamping, so the optimizer stays unconstrained and never sits on a
boundary. A `GreaterThan` cannot represent its own edge, so a `length_scale`
start at or below the bound is moved to `1.5 * min_length_scale`, elementwise —
`length_scale` may be one value per action dimension, which is what freezing a
fitted kernel across folds hands back.

The floor exists because a lengthscale below the design spacing is not
identifiable — see the pre-existing note that the 15-point pilot's "fitted"
lengthscales merely echo `LENGTH_SCALE`. Bounding it is what stops a fit from
wandering into that flat region. Measured across `MT03`/`MT04`/`MT05` (45 points,
7 action dims each), leave-one-out R² under a fitted noise and
`min_length_scale = 0.3` runs −0.046 / 0.607 / 0.275, against −0.406 / 0.343 /
0.210 for a noise pinned at 0.1. On MT03 no configuration beats predicting the
training mean; that is a data limitation, not a hyperparameter one.

**`gp_hyperparameters(model)`** — every hyperparameter of one such GP, as a dict:

| key | what it is |
| --- | --- |
| `lengthscale` | `(d,)` ARD lengthscales, in the normalized `[0, 1]^d` action frame |
| `signal_var` | the `ScaleKernel` outputscale |
| `noise_var` | the observation noise, pinned or fitted |
| `standardize_scale` | the divisor `Standardize` applied to the values |

That is the complete set, and it is the same set either way — only whether the
fit moves `noise_var` changes. With a pinned noise, `model.named_hyperparameters()`
yields only `raw_outputscale` and `raw_lengthscale`, because
`FixedNoiseGaussianLikelihood` stores the noise as a buffer rather than a learned
parameter; it is still a hyperparameter of the model, the fit just never moves
it. With a fitted noise a third entry, `raw_noise`, appears. `ZeroMean` has no
parameters, and `Standardize`'s *offset* is zero because `standard_y` reaches the
GP already centered.

Because `Standardize` divides `train_Y` by its own sample standard deviation, a
fitted `noise_var` is directly the squared fraction of the objective's spread —
`sqrt(noise_var)` is comparable to the `noise_std` you would have pinned. Fitted
on noiseless data it lands on `GaussianLikelihood`'s own floor of `1e-4` rather
than on zero, which would be singular.

`standardize_scale` is what makes the two variances interpretable: they are in
post-`Standardize` units, so a `signal_var` of 4.4 is not comparable to
`standard_y`'s variance of 1.0. Multiplying by `standardize_scale ** 2` undoes
the transform and lands back in the units the GP was handed. The scale is close
to but never exactly 1, since `Standardize` divides by the sample (ddof=1)
standard deviation where `AffineTransform.make_standardized` used the population
(ddof=0) one.

**`BoTorchGP(objective, noise, fit_hyperparameters=True, length_scale, signal_var,
min_length_scale=None)`** — the
single-objective wrapper: one `Objective` rather than a collection, so the action
frame is that objective's own normalization instead of a shared one, and the
marginal likelihood is a plain `ExactMarginalLogLikelihood`. `posterior_at`,
`best_actions`, `update_feedback` and `get_fitted_hyperparameters` carry the same
signatures and contracts as on `DecoupledMOGP` below with `m = 1`, so the
returned arrays keep their column axis. `BoTorchGP.model` **is** the
`SingleTaskGP` — there is no sub-model beneath it, unlike `ModelListGP.models`.
It takes the same hyperparameter arguments as `DecoupledMOGP`.

### DecoupledMOGP (`optimization/gp.py`)

Independent per-objective GPs over that shared action frame — a BoTorch
`ModelListGP` of `SingleTaskGP`s, one per objective. "Decoupled" is the modelling
claim: the objectives are assumed independent, so there is no cross-objective
covariance and the joint marginal likelihood factorizes.

```python
mogp = DecoupledMOGP(objectives, fit_hyperparameters=True,
                     noise=NoiseModel.prior(0.3), min_length_scale=0.3)
mu, std     = mogp.posterior_at(X)          # (n, m) each
best, mu, s = mogp.best_actions()           # (m, d), (m,), (m,)
mogp.update_feedback(objectives)            # rebuild after new measurements
```

**Hyperparameter arguments**, all per-instance rather than class attributes, so
one experiment's choice cannot leak into the next: `length_scale` (default
`LENGTH_SCALE = 0.2`, 20% of the unit action box) and `signal_var` (default
`SIGNAL_VAR = 1.0`) are the *starting* values, and `min_length_scale` (default
None) is a *bound* on the fitted ones. All three go straight to
`build_botorch_gp` and are shared by every sub-model. Each sub-model is an ARD
`ScaleKernel(RBFKernel)` — one lengthscale per action dimension — with
`ZeroMean()` and `Standardize(m=1)`.

`mean_module=ZeroMean()` is correct rather than assumed: `Standardize` centers
the training targets, so the prior mean of the standardized values genuinely is
zero.

**Noise.** `noise` is a `NoiseModel` (or a number/None it coerces), and whichever
representation it carries is a fraction of each objective's *own* standard
deviation rather than a raw magnitude. A pinned level reaches `train_Yvar` as
`(std * spread)^2`, and the pre-multiplication by `spread` is load-bearing:
`Standardize` divides `train_Yvar` by the same variance it divides `train_Y` by,
so scaling first is what makes the parameter scale-free. A fitted noise is
scale-free for the same reason from the other direction — post-`Standardize` the
variance *is* the squared fraction, which is why `sqrt(noise_var)` from
`gp_hyperparameters` is directly comparable to a pinned `std`.

One `NoiseModel` is shared by every sub-model, but each still fits its own value
from its own term of the summed likelihood — the objectives are decoupled, so a
noisy objective and a clean one land on different levels even under one prior.

A fitted noise requires `fit_hyperparameters=True`; both wrappers raise
`ValueError` otherwise, since without a fit nothing would determine the noise and
the GP would silently keep the likelihood's arbitrary starting value.

**`fit_hyperparameters`** fits lengthscales and signal variances — and the noise,
unless it is pinned — by marginal likelihood in one `fit_gpytorch_mll` call on
a `SumMarginalLogLikelihood`. The sum splits over the sub-models precisely
because the objectives are independent, so a single call fits all of them.

**`posterior_at(action, normalized=False, raw=False, chunk=2048)`** — posterior
mean and standard deviation at *arbitrary* actions, on or off any grid.
`normalized=False` (the default) maps raw actions through `objectives.xtransform`
first; pass `normalized=True` if they are already in the `[0, 1]^d` box. Returns
`(n, m)` arrays in **maximization space** — larger-is-better and standardized,
matching `objectives.feedback()`. Pass `raw=True` to get them in the units the
measurements were taken in, sign included.

Which space you want depends on what you are doing with them. Pareto and
hypervolume work needs maximization space, where every objective is
larger-is-better and on the same scale; only reporting and plotting want
`raw=True`.

Actions are fed as `q=1` batch elements (`unsqueeze(1)`), so each posterior is
1x1 per objective and gpytorch never materializes the `n x n` test-test block —
that path is quadratic in the number of evaluation points and dominates the cost
on a large scan. `chunk` bounds the per-call batch.

**`best_actions(num_restarts=8, raw_samples=512, raw=False)`** — the action maximizing each
objective's posterior mean, over the **continuous box**, not over the measured
actions. Two stages, which is what `optimize_acqf` does internally: a Sobol scan
of `raw_samples` points to locate the basins, then box-constrained L-BFGS-B from
the `num_restarts` best starting points. One single-output problem per objective.

No sign handling appears here, and none is needed — `feedback()` is already
flipped to larger-is-better, so the argmax of the posterior *is* the optimum for
a minimized objective too. Returns actions in raw units (via `xtransform.inv`),
plus the diagonal of the posterior evaluated at those actions. The actions are in
raw units either way; `raw=True` puts the values there too.

**`update_feedback(objectives)`** rebuilds every sub-model from scratch and
refits. There is no warm start and no incremental conditioning; a closed-form
exact GP is cheap enough that rebuilding is the intended usage.

**`get_fitted_hyperparameters()`** — a length-m list of `gp_hyperparameters`
dicts, one per objective in objective order. No entry is shared between them:
each sub-model has its own kernel, fitted from its own term of the
`SumMarginalLogLikelihood`.

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
- `hilo/plot_pilot.py` additionally uses pandas, `hilo/gp_diagnostics.py`
  scikit-learn

Verified against botorch 0.18.1 / gpytorch 1.15.2 / torch 2.13.0. Three details in
`gp.py` are version-sensitive: `outcome_transform` must be passed explicitly
(BoTorch's default changed to `Standardize`), `train_Yvar` is what selects
`FixedNoiseGaussianLikelihood`, and the `likelihood` on the fitted-noise path
must be passed explicitly too, since BoTorch's default carries a `LogNormalPrior`
on the noise. A fourth is in `objectives.py`: the bounds `from_synthetic` passes
are both validated against the actions (`evaluate_true` rejects points outside
them) and required to contain a known optimizer.

## Known stale code

`scratch/` is scratch work and is not maintained; some of it still names classes
that no longer exist (`ConjugateGP`). `test_no_stale_references.py` searches
`examples/`, `scripts/` and `hilo/`, none of which is `scratch/`, so nothing
there is checked. `docs/` predates the BoTorch rewrite and describes the
hand-written GP that `git rm`'d in "delete custom gps". `tablet/` is an
independent iPad control panel and does not import `pypolar`.
