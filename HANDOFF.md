# Handoff: choosing the GP's observation noise and lengthscale

**Status: package work DONE and tested. The empirical question is partly settled and
partly open — see §7.**

**Date:** 2026-08-13
**Relevant files:** `src/pypolar/optimization/gp.py`, `src/pypolar/__init__.py`,
`hilo/gp_accuracy.py`, `hilo/gp_diagnostics.py`, `tests/test_botorch_gp.py`,
`tests/test_decoupled_mogp.py`
**Data used:** `human_data/MT03_incline.csv`, `MT04_incline.csv`, `MT05_incline.csv`
(45 points, 7 action dims each), `MT04_met_DR.csv` (replicates), `pilot_mohilo.csv`

---

## 1. What prompted this

`hilo/gp_accuracy.py` had `GP_NOISE = 0.1` as a hand-set constant, and the package
pinned every GP's noise through `train_Yvar` so a fit could never move it. The
question was whether that number could be chosen from data instead of guessed.

It can. It should not be chosen the way the script was measuring, and the answer
matters less than the diagnostic that came out of asking.

---

## 2. What was built

### `NoiseModel` (`optimization/gp.py`, exported from `pypolar`)

A frozen dataclass carrying the three noise representations as named
constructors, so a pinned level and a prior on a fitted one cannot be requested
at once:

```python
NoiseModel.pinned(0.05)           # train_Yvar -> FixedNoiseGaussianLikelihood
NoiseModel.fitted()               # bare GaussianLikelihood, noise free
NoiseModel.prior(0.3, sigma=1.0)  # fitted under a LogNormal centered on 0.3
```

`NoiseModel.coerce` accepts the shorthands — a number means `pinned`, `None`
means `fitted` — so every pre-existing call site kept working through the change.

Two details that are easy to get wrong and are pinned by tests:

- **`median` is on the noise standard deviation; GPyTorch holds the variance.**
  So it becomes `LogNormalPrior(loc=2*log(median))`. The factor of 2 is the
  conversion, not a fudge.
- **`sigma` is the width in log space, on the variance.** The default `1.0`
  matches BoTorch's own default noise prior width rather than being invented.

### Constructor arguments replacing class constants

`LENGTH_SCALE` / `SIGNAL_VAR` / `MIN_LENGTH_SCALE` are no longer class
attributes. `LENGTH_SCALE = 0.2` and `SIGNAL_VAR = 1.0` are module constants used
as defaults:

```python
BoTorchGP(objective, noise, fit_hyperparameters=True,
          length_scale=LENGTH_SCALE, signal_var=SIGNAL_VAR, min_length_scale=None)
DecoupledMOGP(objectives, fit_hyperparameters=True, noise=NOISE_STD,
              length_scale=..., signal_var=..., min_length_scale=None)
```

Per-instance, so one experiment's choice cannot leak into the next. The
`BoundedGP` subclasses both `hilo/` scripts used are deleted.

### `min_length_scale`

Floors every ARD lengthscale via a `GreaterThan` constraint. GPyTorch enforces it
by reparameterization (`softplus(raw) + bound`), not clamping, so the optimizer
stays unconstrained and never sits on a boundary. A `GreaterThan` cannot
represent its own edge, so a start at or below the bound is moved to
`1.5 * min_length_scale`, **elementwise** — `length_scale` may be one value per
dimension, which is what freezing a fitted kernel across LOO folds hands back.

### Guard

A fitted noise requires `fit_hyperparameters=True`. Both wrappers raise
`ValueError` otherwise: without a fit nothing determines the noise, and the GP
would silently keep the likelihood's arbitrary starting value (~0.69 of spread).

**Tests:** 315 passing (`BoTorchGP` 100, `DecoupledMOGP` 47, objectives 131,
public API 32, stale references 5).

---

## 3. What the data actually say

### 3.1 BoTorch's default likelihood carries a noise prior — do not measure through it

The first measurement in this investigation was wrong because of this, so it is
worth stating loudly. Constructing `SingleTaskGP` **without** passing a
`likelihood` gets BoTorch's default, which attaches a `LogNormalPrior` on the
noise centered low. `fit_gpytorch_mll` then maximizes likelihood *plus* log
prior — a MAP estimate, not maximum likelihood.

On 45 points of `MT05_incline`, same data and same kernel:

```
BoTorch default likelihood         noise_std = 0.081
explicit prior-free GaussianLikelihood  noise_std = 0.387
```

The 0.081 was mostly the prior. **This is why `build_botorch_gp` passes the
likelihood explicitly on the fitted path.** As a modelling default the prior is
sensible — it is standard practice, see §5 — but it is not a measurement of what
the data say.

### 3.2 Freely fitted noise, per subject (45 points, 7 dims)

| subject | fitted noise (no floor) | with `min_length_scale=0.3` |
| --- | --- | --- |
| MT03 | 0.001 *(collapsed)* | 0.989 |
| MT04 | 0.445 | 0.445 |
| MT05 | 0.387 | 0.638 |

### 3.3 Independent check from replicates

`human_data/MT04_met_DR.csv` measures each condition twice:

```
pooled within-condition std : 0.267
between-condition std       : 0.577      -> ~0.46 as a fraction of spread
within-condition CV         : 5.4% of the grand mean
```

That is measurement noise with no model at all, and it agrees with the freely
fitted 0.39–0.45 on MT04/MT05. The originally pinned 0.1 is far too small.

### 3.4 Leave-one-out R², all three subjects (45 points each)

| setup | MT03 | MT04 | MT05 |
| --- | --- | --- | --- |
| noise pinned 0.1 *(the old default)* | −0.406 | 0.343 | 0.210 |
| noise free | −0.328 | **0.608** | 0.227 |
| noise ≥ 0.3 | −0.407 | 0.606 | 0.252 |
| lengthscale ≥ 0.3 | **−0.046** | 0.607 | 0.275 |
| both floors | −0.046 | 0.607 | 0.408 |

Baseline (predict the training mean) is 0.000.

### 3.5 A noise floor was measured and rejected

`build_botorch_gp` grew `min_length_scale` but deliberately **no**
`min_noise_std`. The floor is inactive exactly where it would need to help: on
MT04 and MT05 the free fit already lands above 0.3, so the constraint changes
nothing. On MT03, the one subject where it binds, it makes things *worse*
(−0.328 free → −0.407 floored).

An earlier recommendation of "noise ≥ 0.3 **and** lengthscale ≥ 0.3" was
overfitting to MT05. That apparent interaction (0.408 vs 0.275 for the
lengthscale floor alone) appears on MT05 only; on MT03 and MT04 the two rows are
identical to three decimals, and the bootstrap intervals overlap almost entirely.

**Do not add a noise floor back without new evidence.** If a small design
collapses the noise toward zero, that is over-flexibility, and the lengthscale
bound is the lever.

---

## 4. When you only want the best action

Everything above scores *prediction*. If the GP feeds an acquisition function and
you only care about the recommended optimum, the target is simple regret, and
three things change.

### 4.1 R² and regret rank hyperparameters differently — sometimes inverted

Levy, 7 dims, 45 points, 40% noise, 8 seeds:

| setting | LOO R² | simple regret |
| --- | --- | --- |
| noise 0.05, no floor | **0.328** (best) | 0.685 |
| noise fitted, no floor | 0.310 | **0.778** (worst) |
| noise fitted, ls ≥ 0.3 | **0.191** (worst) | **0.368** (best) |

The mechanism is specific and worth internalizing: **noise controls smoothing.**
Assume little noise and the surface threads every measurement, so its highest
point is the highest *measurement* — which under heavy noise is most likely the
point that drew the luckiest noise. Assume more and the peak sits where several
measurements agree.

**Consequence: tuning `GP_NOISE` on `gp_accuracy.py`'s R² does not improve the
optimizer.** That script answers a different question.

### 4.2 But the effect is not statistically real

Paired (same data per seed), which is the honest analysis:

- **One-shot**, 24 seeds × 4 test functions, 45 points: nothing beat pinned-0.1
  significantly. `|t| ≤ 1.47`, win rates 8/24 to 17/24.
- **Sequential EI**, 12 seeds × 3 functions, 8 initial + 25 iterations: best was
  `fitted + ls ≥ 0.3` on Ackley (`t = −1.80`, 8/12 wins). The *same* setting was
  worst on Rastrigin.

The ranking flips by test function. For finding the optimum this choice matters
much less than the problem's own structure.

### 4.3 What does survive

**Don't let the noise collapse.** `fitted, no floor` was worst or near-worst in 3
of 4 one-shot testbeds, with its fitted noise at 0.010 — the likelihood's
numerical floor, i.e. interpolation.

### 4.4 Sequential loops add a failure mode fixed designs don't have

Under an acquisition function the model chooses its own next point, so the design
stops being independent of the model: an overconfident posterior stops exploring,
and the clustered data it then collects looks self-consistent. Concretely,
`fit_gpytorch_mll` raised `ModelFittingError` **5 times across 180 EI runs** and
**zero times in any fixed-design run**. (The failures concentrated on pinned 0.10
rather than on the smallest noise, so the exact mechanism is not established —
treat it as a robustness cost, not a clean story.)

### 4.5 The number that actually matters — bootstrap recommendation stability

Resample each subject's 45 points with replacement 25 times, refit, re-run
`best_actions()`, and measure how far the recommendation moves as a fraction of
the box diagonal:

| setting | MT03 | MT04 | MT05 |
| --- | --- | --- | --- |
| noise 0.10 | 0.367 | 0.284 | 0.273 |
| noise 0.30 | 0.357 | 0.302 | 0.263 |
| noise fitted | 0.356 | 0.276 | 0.258 |
| noise fitted, ls ≥ 0.3 | **0.192** | 0.260 | 0.246 |
| noise 0.30, ls ≥ 0.3 | 0.250 | 0.274 | **0.223** |

MT03 predicted-value spread also falls from 2.785 (fitted, no floor) to 1.134
with the floor.

**On every subject and every setting the recommended action moves 20–37% of the
parameter box under resampling of the same data.** No hyperparameter choice fixes
that. This is the same conclusion §3.4 reaches from the prediction side, and it
is the number to act on. This check needs no groundtruth and measures the
quantity of interest directly, so it is a better gate than R² for optimization
work.

### 4.6 A related trap in `gp_diagnostics.py`

At 1D Levy, 15 points, 25% added noise, the fitted noise collapses to the
likelihood floor (0.010) and **in-sample R² reaches exactly 1.0000** — the GP
interpolates and declares the noise zero. Fitting the noise does not rescue a
design that cannot identify it.

---

## 5. Standard practice, for reference

For GPs on noisy human/physiological data driving an acquisition function:

- **Fit the noise, don't pin it.** Essentially every BO library does. Pinning was
  the unusual choice.
- **Regularize with a prior, not a hard bound.** A prior says "0.3 unless the data
  argue otherwise"; a floor says "below 0.3 is impossible," which is far stronger
  than the evidence supports. BoTorch's default (§3.1) is doing the right thing
  as a default even though it corrupted the measurement.
- **Use a noise-robust acquisition.** Plain EI's `best_f` is itself a noisy draw.
  The standard replacement is noisy EI — `qLogNoisyExpectedImprovement` in
  BoTorch. **Nothing in the package does this yet;** the only acquisition here is
  `PosteriorMean` inside `best_actions`.
- **Recommend the argmax of the posterior mean, never the best observed point** —
  the latter is optimistically biased. `best_actions` already does this.
- **Estimate noise from replicates when the protocol allows** (as in §3.3).
- **Keep the search space small when the budget is small.** 7 parameters with 45
  measurements is outside the regime these methods are comfortable in, which is
  what §4.5 is reporting.

---

## 6. Current defaults in the repo

- **Package:** unchanged behaviour by default. `DecoupledMOGP` still defaults to
  `noise=NOISE_STD` (0.05, pinned) and `min_length_scale=None`. Everything new is
  opt-in.
- **`hilo/gp_accuracy.py`:** `GP_NOISE = plr.NoiseModel.prior(...)`,
  `MIN_LENGTHSCALE = 0.1`.
- **`hilo/gp_diagnostics.py`:** `GP_NOISE = 'prior:0.3'`, `MIN_LENGTHSCALE = None`
  (the right floor is the design spacing, which `--samples`/`--dim` change).
  `--gp-noise` accepts a float, `match` (the oracle's own noise, simulation
  only), `fit`, or `prior:MEDIAN`.

---

## 7. Open questions and what to do next

1. **The measurements in §3–§4 live only in this document.** They were produced by
   throwaway scripts in a session scratchpad that no longer exists. If any of
   these numbers need to be reproduced or extended, the scripts must be rewritten
   — consider promoting the two that earn it: the bootstrap stability check
   (§4.5) and the cross-subject LOO sweep (§3.4).
2. **The bootstrap stability check is not in the repo.** It is the most
   decision-relevant diagnostic found here and currently exists nowhere. Adding it
   to `gp_accuracy.py` as a reporting mode was offered and not yet done.
3. **Prior center is not calibrated.** `prior(0.3)` is a round number near the
   replicate estimate (0.46) and the free fits (0.39–0.45). Nothing has tested
   whether 0.3, 0.45, or something else is better, or how much `sigma` matters.
4. **MT03 has no out-of-sample skill under any configuration** (best R² −0.046).
   Worth understanding why it differs from MT04/MT05 before drawing subject-level
   conclusions from it.
5. **No noisy-EI acquisition exists in the package** (§5). If the sequential loop
   is going live, that is the gap to close, not the noise setting.
6. **The 0.3 lengthscale floor was chosen with an eye on the same LOO scores used
   to evaluate it**, so its margin is somewhat optimistic. The direction is
   well-supported by the identifiability argument (see the pre-existing note that
   the 15-point pilot's fitted lengthscales merely echo their initialization); the
   exact threshold is not.
