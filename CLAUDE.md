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
│   │   └── gp.py                   # build_botorch_gp, GPHyperparameters, BoTorchGP, DecoupledMOGP
│   ├── feedback/
│   │   ├── rewards.py              # InternalReward hierarchy (groundtruth objectives)
│   │   ├── oracles.py              # Simulated humans: Bradley-Terry, noisy regression
│   │   ├── acquisition.py          # AcquisitionFunction: a botorch acqf over an Objective
│   │   └── synthetic.py            # truth_at, construct_function, SyntheticFunction
│   ├── performance/
│   │   ├── mo.py                   # groundtruth_hypervolume, pareto_overlay
│   │   ├── loo.py                  # loo: leave-one-out cross-validation
│   │   └── regret.py               # normalized_inference_regret
│   └── utils/                      # pareto.py, plotting.py
├── hilo/
│   ├── plot_pilot.py               # The one live experiment: pilot data -> fronts + optima
│   ├── gp_diagnostics.py           # Leave-one-out R^2 and calibration on a synthetic objective
│   ├── gp_accuracy.py              # The same leave-one-out, on the measured subject data
│   ├── plot_test_functions.py      # Every synthetic test function, at DIM 1 or 2
│   ├── read_data.py                # Pilot and MT0x CSVs -> Objective / DecoupledObjectives
│   └── output/                     # Figures and recorded runs
├── scratch/                        # Scratch/experimentation files (not maintained)
├── docs/                           # LaTeX writeup and handoff record from the pre-BoTorch GP
├── tablet/                         # iPad comfort survey (survey.py + survey.html), no pypolar import
├── human_data/                     # Pilot CSVs consumed by plot_pilot.py
└── tests/                          # pytest test suite
    ├── test_public_api.py          # Every __all__ name imports
    ├── test_no_stale_references.py # No dangling references in scripts/hilo
    ├── test_objectives.py          # AffineTransform, Objective, DecoupledObjectives
    ├── test_botorch_gp.py          # build_botorch_gp, gp_hyperparameters, BoTorchGP behaviour
    ├── test_decoupled_mogp.py      # DecoupledMOGP behaviour
    └── test_loo.py                 # leave-one-out folds, predictions, fit_gp arguments
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

515 tests: objectives (176), `BoTorchGP` (113), `DecoupledMOGP` (56), public
API surface (54), `experiment` (31), `ScalarizedGP` (24), `dataset` (20),
`AcquisitionFunction` (19), `loo` (12), dangling references (10).

`.github/workflows/tests.yml` runs the same command on every push and on every
pull request into main. It installs the CPU torch wheel before the package,
since a runner has no GPU and the default build is a multi-gigabyte download
that nothing here would use. Nothing is pinned, so CI resolves newer numpy and
scipy than a long-lived conda env will have.

## How to run experiments

```bash
conda activate pypolar
python -m hilo.plot_pilot        # fits the pilot data, writes hilo/output/test.png,
                                 # prints the best action per objective
python -m hilo.gp_diagnostics    # leave-one-out R^2 and calibration on a synthetic
                                 # objective, writes hilo/output/gp_diagnostics.png
python -m hilo.gp_accuracy       # the same leave-one-out, on a subject's CSV
python -m hilo.plot_test_functions  # every SYNTHETIC_FUNCTIONS entry with a
                                 # non-constant DIM-dimensional instance, writes
                                 # hilo/output/synthetic_functions.svg
```

`plot_pilot.py` needs `pandas`, which is not a package dependency (it is an
experiment-only import). `optimize_acqf` seeds its restarts from torch's global
state, which the script does not set, so the reported optima wobble in the third
or fourth decimal between runs.

`plot_test_functions.py` needs `tqdm`, also experiment-only. `DIM` is 1 or 2,
since that is what `plot_test_function` can draw; at `DIM = 1` most of the
registry has no instance and the script prints what it skipped.

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
from pypolar import DecoupledMOGP, BoTorchGP, NoiseModel, GPHyperparameters
from pypolar import DecoupledObjectives, Objective, AffineTransform
from pypolar import sample_actions
from pypolar import loo
from pypolar import normalized_inference_regret
from pypolar import truth_at, construct_function, SyntheticFunction
from pypolar import SYNTHETIC_FUNCTIONS, SYNTHETIC_1D_FUNCTIONS
from pypolar import plot_test_function, plot_fit_1d
```

Everything runs on CPU in float64 (`pypolar.optimization.gp.DTYPE`). numpy is the
boundary in both directions: every public method takes and returns numpy arrays,
and torch never escapes the module. Three arguments are the exception, all inputs
only: `Objective.from_synthetic`'s `function`, a BoTorch `SyntheticTestFunction`
subclass; `sample_actions`'s `bounds`, anything `torch.as_tensor` accepts; and
the `truth` taken by `feedback/synthetic.py`, a `SyntheticTestFunction`
*instance*. All are consumed internally and only numpy comes back out.

### Objectives (`optimization/objectives.py`)

The bookkeeping layer. It owns the two coordinate changes that the GP assumes
have already happened, so nothing downstream has to think about units or signs.

**`sample_actions(bounds, n, kind, seed)`** — `n` actions over a box, returned
`(n, d)`. `bounds` is `(2, d)` of `[lower; upper]` rows, the form BoTorch states
bounds in, and is coerced with `torch.as_tensor`, so a torch tensor, a numpy
array and a nested list all give the same design. `kind='sobol'` draws a
space-filling Sobol sequence and `kind='uniform'` draws iid uniform points; the
branch is a whitelist, so any other value raises `ValueError` rather than
silently returning a design you did not ask for. Sobol is the default worth
reaching for because a 2^k-point sequence splits every axis exactly in half,
where an iid design leaves clumps and gaps at the sample sizes these experiments
run at.

**`AffineTransform(scale, shift)`** — `x -> (x + shift) * scale`, with `inv()`.
Four constructors: `make_standardized(data, sign=1.0)` (zero mean, unit
variance, times `sign`), `make_centered` (zero mean), `make_normalized` (mapped
onto `[0, 1]`), and `make_normalized_from_bounds(low, high)` (the same `[0, 1]`
map, from a declared range rather than from the data). Each guards against
zero-variance/zero-range data by leaving it
unscaled. `inv_scale()` is the inverse for a *spread*: it undoes the scaling but
not the shift, since a shift does not move a standard deviation and a sign flip
cannot make one negative. Means go back through `inv()`, standard deviations
through `inv_scale()`.

**`Objective(name, maximize, ydata, xdata, column=None, action_bounds=None)`** —
one measured objective: `ydata` is `(N,)`, `xdata` is `(N, K)`. On construction
it builds

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

Built with `from_df(df, column, action_columns, maximize, name)`,
`from_data(actions, values, maximize, name)`, or `from_empty(name, maximize,
action_bounds=None)` for an objective declared before its first measurement.
`add_points()` appends and re-runs `__post_init__`, so the transform tracks the
new mean.

**`action_bounds`** pins the action frame. Without it, `xtransform` is
`make_normalized(xdata, axis=0)` — the bounding box of the points measured *so
far*, which moves every time a point is added. With it, the frame is
`make_normalized_from_bounds(low, high)` and holds still. `low` and `high` are
in raw action units, scalar or one per action dimension.

This matters because the x frame is the one the GP's hyperparameters are stated
in. `LENGTH_SCALE = 0.2` means "20% of the box", `min_length_scale = 0.3` floors
a physical distance only if the box does not move, and a lengthscale fitted at
step 10 is comparable to one fitted at step 20 only under the same
normalization. A sequential run on a data-derived frame silently changes what
every one of those numbers means.

There is deliberately **no equivalent for the values**, because the y frame is
inert. `Standardize(m=1)` subtracts `train_Y`'s own mean and divides by its own
sample standard deviation before the fit, and the pinned-noise path
pre-multiplies by `spread = train_Y.std()`, so any positive affine rescaling of
`standard_y` cancels out exactly. Measured on 15 points: standardizing,
normalizing and centering the same values gave identical lengthscales
(0.180903), identical `noise_var` (0.081210) and posterior means agreeing to
6e-14. Pinning the y frame would change what `standard_y` reads as and nothing a
model does.

**Actions must lie in the declared box.** `__post_init__` raises `ValueError`
for any action outside `action_bounds`, which covers both construction and every
subsequent `add_points`, and `add_points` rolls its append back before
re-raising so a refused point is not left in the record. The check allows
`BOUNDS_SLACK = 1e-9` of the box span, because an action that `optimize_acqf`
puts *on* a boundary comes back a float epsilon outside it; a real overrun is
orders of magnitude larger than that and still raises.

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
boxes      = [o.action_box() for o in objectives]   # bounds if pinned, else data range
xtransform = AffineTransform.make_normalized_from_bounds(min of lows, max of highs)
```

a single per-dimension normalization spanning every objective's box, so actions
from different objectives land in the same `[0, 1]^d` box and one set of GP
lengthscales is meaningful across all of them. `Objective.action_box()` returns
that objective's `action_bounds` when they pin it and its measured range
otherwise, so a collection of declared objectives gets a frame that holds still
for the same reason a single one does. The union is taken with
`reduce(np.minimum, ...)` so a scalar bound broadcasts against a per-dimension
one.

Selectors (`objs`) accept a name, an index, a list of either, a slice, or `None`
for all. `__getitem__` returns a bare `Objective` for a single selector and a new
`DecoupledObjectives` for a multi-selector, so `objectives[['Cost', 'Comfort']]`
is how you fit a GP to a subset. `feedback(objs)` returns `standard_y`,
`actions(objs)` returns normalized actions, and `max`/`min`/`range` report the
standardized values.

`to_raw(mu, std=None, objs=None)` is the way back out: it maps posterior moments
from maximization space into each objective's own units, column by column, using
`inv()` on the means and `inv_scale()` on the standard deviations. This is what
`posterior_at(raw=True)` and `recommend(raw=True)` call.

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

**`GPHyperparameters`** — every hyperparameter of one such GP, as a frozen
dataclass. `gp_hyperparameters(model)` reads one off a model's own tensors, and
both wrappers' `get_fitted_hyperparameters()` return them:

| field | what it is |
| --- | --- |
| `lengthscale` | `(d,)` ARD lengthscales, in the normalized `[0, 1]^d` action frame |
| `signal_var` | the `ScaleKernel` outputscale |
| `noise_var` | the observation noise, pinned or fitted |
| `standardize_scale` | the divisor `Standardize` applied to the values, default 1.0 |

It is a dataclass rather than a dict so that a hand-written specification — the
one an experiment freezes across cross-validation folds — is checked at
construction instead of failing on a mistyped key deep inside a fit.
`standardize_scale` is the only field with a default, because it is reported
rather than consumed: nothing takes it as an input, and 1.0 is its value for
values already at unit spread. `lengthscale` may be a scalar when the same value
is meant for every dimension. Comparison is by identity (`eq=False`), since a
field-by-field `==` on the `lengthscale` array would return an array whose truth
value is ambiguous.

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
`sample_paths`, `recommend`, `update_feedback` and `get_fitted_hyperparameters`
carry the same signatures and contracts as on `DecoupledMOGP` below with `m = 1`,
so the returned arrays keep their column axis. `BoTorchGP.model` **is** the
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
paths = mogp.sample_paths(X, num_paths=20)  # (num_paths, n, m), joint over X
best, mu, s = mogp.recommend()              # (m, d), (m,), (m,)
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

**`sample_paths(action, num_paths, normalized=False, raw=False)`** — that many
sample paths of the posterior over those same arbitrary actions, returned
`(num_paths, n, m)`. The `normalized` and `raw` flags mean exactly what they mean
on `posterior_at`, and the values come back in maximization space unless `raw` is
set.

`num_paths` is deliberately **not** called `q`, which everywhere else in the
package and in BoTorch means the points evaluated jointly as one candidate set —
`AcquisitionFunction.query(model, q=1)` is that q. Here botorch's q is `n`: the
actions are one q-batch, and `num_paths` is how many times it is sampled.

This is the counterpart of `posterior_at`, not a variant of it: `posterior_at`
reports the **marginal** at each action, and `sample_paths` draws from the
**joint** posterior over the whole set. That is what makes a draw a function
rather than noise — sampling each action from its own marginal independently
would throw away the covariance between them, and a "path" through those points
would be white noise. It is also why `sample_paths` takes no `chunk`: the `n x n`
test-test block that `posterior_at` goes out of its way never to form is the
exact object a joint draw is drawn from, so the actions go in as *one* batch
element where `posterior_at` sends `n` of them. Cost is the price of that —
quadratic in `n`, plus a Cholesky.

The two agree where they overlap, which is the marginals: averaged over enough
draws, `sample_paths` reproduces `posterior_at`'s mean and standard deviation
point by point, and the tests pin that.

On `DecoupledMOGP` a draw is joint over the actions but **independent across the
objectives** — column `j` comes from objective `j`'s own GP and carries no
covariance with column `k`, which is the same decoupling claim the class makes
everywhere else.

Draws come from torch's global generator (`rsample`), so `torch.manual_seed` is
what makes them repeatable — the same way `recommend`'s restarts are seeded.

**`recommend(num_restarts=8, raw_samples=512, raw=False)`** — the action maximizing each
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

**`get_fitted_hyperparameters()`** — a length-m list of `GPHyperparameters`,
one per objective in objective order. No entry is shared between them:
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

**`synthetic.py`** — the other kind of groundtruth: BoTorch's own test functions
rather than an analytic bump. This is the only module that evaluates one, so
`torch` appears here and nowhere else in `feedback/`.

- `SYNTHETIC_FUNCTIONS` — name -> `SyntheticTestFunction` subclass, the registry
  the experiments draw from, 24 entries.
- `SYNTHETIC_1D_FUNCTIONS` — the 7 of those with a non-constant 1D instance:
  Ackley, DixonPrice, Griewank, Levy, Michalewicz, Rastrigin, StyblinskiTang.
  Most of the registry is 2D-or-higher only, so this is the subset a 1D sweep or
  plot can actually use. It is a measured fact rather than a fixed one —
  `construct_function` is the authority, and it takes a box: a box that excludes
  a function's known optimizer drops it, which puts StyblinskiTang (optimizer at
  −2.904) outside the set below a half-width of 2.91.
- `truth_at(truth, X)` — noiseless values of the truth at the `(n, d)` actions
  `X`, returned `(n,)`. The evaluation is `noise=False`, so a metric is never
  scored against a lucky draw. Takes an *instance*, not a subclass.
- `construct_function(func, dim, box, seed=0)` — one non-constant instance of
  `func` at that `dim`, or None when it has none. It tries `[-box, box]^dim`
  first and falls back to the function's own default bounds, since a custom box
  is rejected unless it contains a known optimizer. The non-constant probe is
  needed because `Powell` sums over `range(dim // 4)` and `Rosenbrock` over
  `range(dim - 1)`, so below dim 4 and dim 2 they are identically zero rather
  than an error.
- `SyntheticFunction(truth, rel_noise_std=0.0, measure='std', n_spread=4096,
  seed=0)` — a test function plus an observation noise stated as a fraction of
  the function's own spread. `sf(X)` measures, `sf(X, noise=False)` is
  `truth_at`, and `sf.spread` / `sf.noise_std` report what the fraction resolved
  to. `measure` picks the standard deviation or the peak-to-peak range of a
  Sobol scan of the whole box.

  The spread is measured **once**, at construction. That is what makes
  `rel_noise_std` mean the same difficulty across functions whose ranges differ
  by orders of magnitude — the same convention `Objective.from_synthetic`'s
  `rel_noise_std` uses — and it is the only way a sequential loop can use the
  convention at all, since it adds one point at a time and a single point has no
  spread of its own to take a fraction of. The noise comes from an
  instance-owned `default_rng(seed)`, so repeated calls advance one stream
  rather than repeating a seeded draw.

### Metrics (`performance/mo.py`, `performance/loo.py`, `performance/regret.py`, `utils/pareto.py`)

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

**`loo(objective, fit_gp, noise, hypers=None)`** (`performance/loo.py`) —
leave-one-out cross-validation over one `Objective`'s N measurements. Fold `i`
is fit on every measurement except `i` and then predicts point `i`, so the
returned `(mu, std, models)` — `(N,)`, `(N,)`, and the N fold GPs — are honest
out-of-sample predictions rather than a training fit. `mu` and `std` come back
in raw units (`posterior_at(raw=True)`), which is what pairs them with
`objective.ydata` for an R².

The GP is not built here. `fit_gp` is supplied by the caller and is called as
`fit_gp(objective, noise, hypers)`, with `noise` and `hypers` passed straight
through. Anything else a fold's GP needs — a lengthscale floor, say — is bound
into that callable (`functools.partial`), which is what guarantees every fold
and the full-data fit are the *same* configuration: cross-validating a GP that
differs from the one being evaluated measures nothing. Passing `hypers` freezes
the full-data fit's hyperparameters across the folds; passing None refits them
fold by fold, which is the honest but far slower measurement.

**`normalized_inference_regret(raw_recommended_action, ground_truth,
maximize=False)`** (`performance/regret.py`) — how far the action a run would
recommend right now sits from the truth's optimum, as a fraction of the truth's
own range.

It takes a `SyntheticOracle` rather than a bare `SyntheticTestFunction`, and
that is what makes it self-contained: the oracle measured `sample_min`,
`sample_max` and `ptp` once over a Sobol scan of its whole box at construction,
so the caller supplies only the recommendation. The truth is evaluated with
`noise=False`, so a lucky draw of the observation noise cannot flatter a score.

Dividing by `ptp` is what makes runs comparable. Two truths whose values differ
by orders of magnitude produce raw gaps that differ by the same factor; in
spreads of each truth's own range, a regret of 0.05 means the same thing on
both. This is the same convention `rel_noise_std` uses for the noise.

*Inference* regret, not *simple* regret: it scores the posterior argmax — what a
human-in-the-loop study actually hands back to a subject — not the best point
sampled so far. The two behave differently, and it matters when reading a curve:
simple regret can only improve, while inference regret can get *worse* when a
new measurement moves the posterior argmax.

`maximize` says which direction the truth is optimized in, and picks both the
target — `sample_max` or `sample_min` — and which way the gap is subtracted. A
regret is a distance from the optimum, so it is non-negative either way:
minimizing scores `inferred - sample_min` and maximizing scores `sample_max -
inferred`. It defaults to False because a
botorch synthetic built with `negate=False` — which is what `Objective` expects,
since it encodes direction itself — is being minimized.

### Plotting (`utils/plotting.py`)

`plot_test_function(ax, X, y, title=None)` — a scalar function sampled at the
`(N, d)` actions `X`, as a line for `d = 1` and a filled contour for `d = 2`;
any other `d` raises. It takes the values `y` rather than the function that
produces them, which keeps the module on plain arrays — independent of any GP or
groundtruth class, and free of a `utils` -> `feedback` import.

The scan need not be a grid: 1D is sorted before it is drawn and 2D is contoured
over its own Delaunay triangulation, so a Sobol sequence plots correctly either
way.

`plot_fit_1d(ax, x, mu, std, xdata, ydata, truth=None, paths=None, vlines=None,
band_std=2.0, title=None)` — a 1D model fit: the posterior mean over the `(n,)`
grid `x`, a `band_std`-sigma band around it, the `(S, n)` sample `paths` and the
`(n,)` `truth` curve when given, and the measurements `(xdata, ydata)` it was fit
to. `vlines` is label -> action, drawn styled by `VLINE_STYLES` in the order
given. Returns the `ax`.

Everything arrives already evaluated, for the same reason `plot_test_function`
takes `y`: the caller owns the GP and the groundtruth, so the module stays on
plain arrays and free of a `utils` -> `optimization`/`feedback` import. What the
band means is therefore the caller's choice too — `posterior_at`'s `std` is the
*latent* posterior, so the band is the model's uncertainty about the noiseless
function, which is what a truth curve should fall inside; a predictive interval
for a new measurement would be wider by the observation noise.

`x` need not be sorted: it is ordered on entry and `mu`, `std`, `truth` and
`paths`' columns are reordered with it.

## Typical workflow

1. Build an `Objective` per measured quantity, each with `maximize=True/False`,
   and collect them in a `DecoupledObjectives`.
2. Select the subset you want to model: `objectives[['Cost', 'Comfort']]`.
3. `DecoupledMOGP(subset, fit_hyperparameters=True)`.
4. Read the posterior with `posterior_at`, the optima with `recommend`, and
   the front with `get_nondominated` over a scan of `posterior_at`.
5. After new measurements: `objectives.add_point(...)` then
   `mogp.update_feedback(objectives)`.

## Dependencies

- **Runtime:** numpy, scipy, pymoo, botorch (which pulls torch and gpytorch)
- **Dev:** pytest, matplotlib
- `hilo/plot_pilot.py` additionally uses pandas, `hilo/gp_diagnostics.py`
  scikit-learn, `hilo/plot_test_functions.py` tqdm

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
