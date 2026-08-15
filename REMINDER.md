# REMINDER

Ideas I thought about implementing but haven't had time to. This is a parking lot,
not a changelog — an entry stays here until the work is actually done, then it gets
deleted.

## How to add an entry

Every entry is a `###` heading with the short description, followed by four fields:

- **Time** — when the reminder was written, as `YYYY-MM-DD HH:MM TZ`. Absolute, never
  "yesterday" or "last week", so entries stay readable months later.
- **Priority** — `1`, `2`, or `3`, where **1 is the most important** and 3 the least.
  Always ask me for this; do not guess it.
  - **1 — most important.** Blocking or near-blocking. Something downstream is wrong
    or untrustworthy until this is done, so it comes before further work in that area.
  - **2 — medium.** Should happen soon, but current results are still usable without
    it. Typically a known misspecification whose effect I can reason around for now.
  - **3 — least important.** Nice-to-have: cleanup, deduplication, ergonomics.
    Revisit when there's spare time; nothing depends on it.
- **File** — the file (or files) where the change lands, or where the problem shows up.
  Use repo-relative paths, with line numbers if they help.
- **Why** — why this matters. Infer it from the description of the reminder if you can;
  ask if the reason isn't clear from what I said.

Newest entries go at the top.

---

### Rewrite the throwaway scripts behind HANDOFF.md §3–§4

- **Time:** 2026-08-13 16:35 EDT
- **Priority:** 1
- **File:** new script(s) under `hilo/`; the numbers they back are `HANDOFF.md` §3.2–§3.5,
  §4.1–§4.5
- **Why:** Every noise sweep, cross-subject LOO table and regret comparison in the handoff
  was produced by scripts in a session scratchpad that no longer exists, so the numbers can
  be read but not reproduced, extended to a new subject, or re-run after a change to
  `gp.py`. The two worth promoting are the cross-subject LOO sweep (§3.4) and the bootstrap
  stability check (§4.5, its own reminder below). Until they exist, a decision that cites
  those tables cannot be re-checked.

### Regret-minimization script: bootstrap stability of `best_actions()`

- **Time:** 2026-08-13 16:35 EDT
- **Priority:** 1
- **File:** a new `hilo/regret_minimization.py` (not a mode inside `hilo/gp_accuracy.py`);
  uses `DecoupledMOGP.best_actions` in `src/pypolar/optimization/gp.py`
- **Why:** Resample a subject's measurements with replacement, refit, re-run
  `best_actions()`, and report how far the recommendation moves as a fraction of the box
  diagonal. It needs no groundtruth and measures the quantity we actually care about, so it
  is a better gate than LOO R² for optimization work — §4.1 shows R² and regret can rank
  hyperparameters in opposite orders. Measured once at 20–37% of the box across all three
  subjects and every setting (§4.5), which is the strongest evidence we have that the
  design is underpowered; it currently exists nowhere in the repo.

### Compare acquisition functions at human-scale noise

- **Time:** 2026-08-13 16:35 EDT
- **Priority:** 1
- **File:** `hilo/regret_minimization.py`; the package has no acquisition module — the only
  acquisition is `PosteriorMean` inside `best_actions` (`src/pypolar/optimization/gp.py`)
- **Why:** Run a few acquisitions (plain EI, `qLogNoisyExpectedImprovement`, UCB) across
  many seeds and several synthetic functions, at an observation noise of **0.3–0.7 of the
  objective's spread** — the range the human data actually shows (freely fitted 0.39–0.45,
  replicates ~0.46, `HANDOFF.md` §3.2–§3.3). Noise that high is the regime plain EI is worst
  in, since its `best_f` is itself a noisy draw, so an acquisition ranked at the usual
  textbook noise levels says nothing about this problem. Many seeds because §4.2 found the
  ranking flips by test function and no setting won significantly at 12–24 seeds. Pairs
  with the oracle reminder below, which supplies a groundtruth with the right geometry.

### Oracle backed by the GP fitted to the human data

- **Time:** 2026-08-11 11:29 EDT
- **Priority:** 1
- **File:** `src/pypolar/feedback/rewards.py` (a new `InternalReward` subclass wrapping
  a fitted `DecoupledMOGP`); `src/pypolar/feedback/oracles.py:84` (`NoisyRegressionOracle`
  — usable unchanged, since the noise is a per-objective scalar); the fit itself is the
  one `hilo/plot_pilot.py` already builds
- **Why:** To compare acquisition functions on regret, I need a groundtruth objective
  that (a) can be evaluated at any action, not just the measured ones, and (b) has a
  known optimum, since regret is `f(x*) - f(x_t)`. The existing `InternalReward`s
  (`IdealPoint`, `BoundedIdealPoint`, ...) satisfy both but are analytic bumps with no
  relationship to how a human actually responds to exoskeleton parameters. Ranking
  acquisition functions on them measures which one suits a synthetic quadratic, which is
  not the question.
  The posterior mean of the GP fitted to the pilot data satisfies both conditions *and*
  has the right shape:
  1. **Right geometry.** Correct action dimension, and a smoothness set by the ARD
     lengthscales that marginal likelihood actually fitted to human measurements rather
     than one I picked.
  2. **Exact optimum.** `best_actions()` maximizes the posterior mean over the continuous
     box with `optimize_acqf`, so `f(x*)` is known to optimizer precision. Regret is then
     exact, not read off a grid.
  3. **Multi-objective for free.** One `DecoupledMOGP` over `m` objectives is `m`
     groundtruth surfaces at once, so the same oracle serves scalar regret and
     hypervolume regret. The true front comes from one scan of the posterior mean, which
     is what a hypervolume-regret curve needs as its normalizer.
  Regression feedback is `mu(x) + N(0, sigma^2)` with **sigma constant** — the groundtruth
  surface is the fitted posterior mean, and the noise is homoscedastic. `sigma` is a
  per-objective scalar, so `NoisyRegressionOracle` works as written. Deliberately *not*
  the pointwise posterior std, for two reasons:
  1. It would not mean what it looks like it means. The posterior variance is
     `k(x,x) - k(x,X)[K + s^2 I]^-1 k(X,x)`, which contains no `y` — with hyperparameters
     fixed, a smooth dataset and a heavily scattered one on the same design give
     *bit-identical* std fields (checked). It tracks where the pilot sampled *least*, not
     where the human was least consistent, since the pinned homoscedastic `train_Yvar`
     (`gp.py:60`) leaves the model no way to represent a locally noisier region.
  2. It would confound the experiment. The acquisition functions being compared are
     uncertainty-seeking, so tying oracle noise to a coverage-driven std field penalizes
     exploratory acquisitions preferentially in the regions they are designed to visit.
  **Scale `sigma` to the spread of `mu` over the box, not to the training targets.** The
  targets are unit-variance by construction, but `mu` reverts toward the zero prior mean
  away from data, so it carries much less. Measured on the pilot fit (IQR of `mu` over a
  20k-point scan, standardized units): `Comfort Floor` 1.507, `Comfort Treadmill` 1.324,
  `Metabolic Cost` 0.395, `10m walk test` 0.085. A `sigma` set as a fraction of that IQR
  is comparable across objectives; one set against the unit target variance is not.
  **Do not let marginal likelihood choose the groundtruth surface — on this design it
  cannot.** The pilot is 15 points in 3 action dims with nearest-neighbour distances of
  0.21-0.38 in the normalized box. A lengthscale is only estimable when points sit within
  a lengthscale of each other, so every candidate below the sampling spacing tells the
  same story ("all decorrelated") and the likelihood cannot separate them. Measured: for
  `10m walk test`, six starting lengthscales spanning 0.02 to 2.0 all converge to the same
  `MLL/n` to four decimals (1.7302 vs 1.7303) while landing on lengthscales three orders of
  magnitude apart and on 75%-99.9% of the box at the prior. The hyperparameters are
  non-identifiable, so the class default `LENGTH_SCALE = 0.2` is picking the answer, not
  the data. (Checked and ruled out: this is *not* the pinned noise forcing a collapse —
  sweeping `noise_std` 0.05 to 0.5 leaves the lengthscale at ~0.037 throughout.)
  So the oracle should **fix the lengthscale explicitly**, at or above the ~0.3 design
  spacing, and state it as an assumption of the benchmark rather than inheriting whatever
  `fit_gpytorch_mll` returns.
  **Then check the surface is not degenerate.** Degenerate here means the fitted
  lengthscale is short enough that `k(x, X) ~= 0` and the posterior falls back to the
  prior: `mu ~= 0` everywhere except spikes at the training points, a surface with no
  optimum worth regretting against. At the current default, `10m walk test` is exactly
  that (`lengthscales=[0.038, 24.9, 0.085]`, median `|mu|` 0.037, **81.6%** of the box at
  the prior) and `Metabolic Cost` is marginal (54.5%); only the two comfort objectives give
  real surfaces (2.4% and 0.9%). Note the naive diagnostics miss this: `10m walk test` has
  `std(mu)=0.329` and `range(mu)=3.123`, both spike-driven. Percent of the box at the
  prior, or median `|mu|`, is what separates a surface from a set of spikes.
  Priority 1 because until this exists, any claim about one acquisition function beating
  another on this problem is untested.



### Write out the LOO R^2 metric and validate against a series of 1D GPs with worse and worse fits

- **Time:** 2026-08-12 2:45 EDT
- **Priority:** 1
- **File:** fill in @claude
- **Why:** fill in @claude


### Create a script for 1D acquisition function testing and fitting. Use the same noise profile as the collected data.

### Create a multi objective version with mo acq f

### Implement a decoupled version of mo acq f and test

### Ensure DecoupledObjectives matches the capabilities afforded by Objectives but in the MO sense

### Figure out why prior 1.0 Fitted noise results in lower noise than asserting a lower bound

### Look into small n std calculation. if there are only a few datapoints the standaradization is bad

### Move objective.py to feedback/ and move truth_at as well?

### Make random acquisition function compatible