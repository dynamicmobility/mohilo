# Handoff: scalarized single-output GP from a `DecoupledMOGP`

Goal: a `make_single_gp(w)`-style construct whose output is the scalar
`w · [GPs]`, usable inside an acquisition function conditioned on a tradeoff
`w`. Nothing below is implemented yet.

## Why it works, and why no refit is needed

`DecoupledMOGP` holds `m` independent GPs, so the joint posterior at an action
`x` is Gaussian with mean `mu(x)` in `R^m` and a **diagonal** covariance
`Sigma(x) = diag(sigma_1^2(x), ..., sigma_m^2(x))` -- diagonal is exactly the
decoupling claim the class makes everywhere else.

A fixed linear scalarization `g(x) = w^T f(x)` is a linear functional of a GP,
and linear functionals of GPs are GPs. So `g` is itself a single-output GP:

- mean:     `w^T mu(x)`
- variance: `w^T Sigma(x) w = sum_j w_j^2 sigma_j^2(x)`
- kernel:   `k_w(x, x') = sum_j w_j^2 k_j(x, x')`

This is exact, not an approximation, and it is a linear readout of models that
are already fit. **Changing `w` therefore costs nothing** -- no refit, no new
hyperparameters -- which is what makes sweeping many `w` per iteration viable.

Precondition, already satisfied: `objectives.feedback()` is standardized and
sign-flipped to larger-is-better, so `w^T y` is dimensionally meaningful and
larger-is-better too. On raw units (metabolic cost vs. seconds vs. a 1-7 rating)
a weighted sum would mean nothing.

## Why not fit a GP to scalarized data

The alternative reading is a real `SingleTaskGP` trained on `w^T Y`. Two reasons
that is wrong here:

1. The objectives are **decoupled** -- each `Objective` carries its own `xdata`
   and nothing requires a shared design, so there is generally no action where
   all `m` objectives were measured and `w^T Y` has no rows to sit on.
2. Even with a shared design it would need one fit per `w`, and would carry
   different hyperparameters than the models evaluated everywhere else.

## Implementation

BoTorch supplies the primitive: `ScalarizedPosteriorTransform(weights=w)` from
`botorch.acquisition.objective` maps a multi-output posterior to exactly the
scalar posterior above. Every analytic acquisition (`ExpectedImprovement`,
`UpperConfidenceBound`, `PosteriorMean`, `LogEI`) and every MC acquisition takes
it as `posterior_transform=`. The work is plumbing, not math.

### 1. `optimization/gp.py` -- a `ScalarizedGP` wrapper

Either a class, or `DecoupledMOGP.scalarized(w)` returning one:

```python
class ScalarizedGP:
    """One DecoupledMOGP read through fixed weights w, as a single-output GP."""
    def __init__(self, mogp, w):
        w = np.asarray(w, dtype=float)         # (m,), in maximization space
        # validate len(w) == len(mogp.objectives)
        self.mogp      = mogp
        self.model     = mogp.model            # the same ModelListGP, unchanged
        self.transform = ScalarizedPosteriorTransform(
            torch.as_tensor(w, dtype=DTYPE))
```

`posterior_at`, `sample_paths` and `best_actions` mirror `BoTorchGP`'s
signatures and contracts, returning a single column; internally each is just
`self.model.posterior(X, posterior_transform=self.transform)`. `best_actions`
collapses to one `optimize_acqf(PosteriorMean(model, posterior_transform=...))`
rather than `m` of them.

There is deliberately **no `raw=True`**: `w^T y` mixes units and has no single
objective's frame to invert into. State that in the docstring rather than
inventing one.

### 2. `feedback/acquisition.py` -- pass the transform through

`query` builds `self.acqf(model.model, **self._incumbent(objective))`. It needs
to add `posterior_transform=model.transform` when the model carries one, and
`_incumbent` needs a scalarized `best_f`.

That incumbent is the one awkward piece, and it follows from decoupling: EI's
`best_f` wants the best scalarized value observed so far, but no action carries
all `m` measurements. The defensible fix is to take it from the **posterior
mean** -- evaluate `w^T mu` over the union of every objective's measured actions
and take the max. That is the standard noisy-EI incumbent, and the honest one
here, since with decoupled data the incumbent is inferred rather than observed.
`qNoisyExpectedImprovement` sidesteps the question entirely by taking
`X_baseline` instead, and is the recommended acquisition for this path.

### 3. Where `w` comes from

For a conditioned acquisition, `w >= 0` with `sum(w) == 1` -- a point on the
simplex. Random draws are `rng.dirichlet(np.ones(m))`; a fixed sweep is a
simplex lattice.

## Limitation

This covers **linear** scalarizations only. Linear weights can never recover a
point in a concave region of the Pareto front, whatever `w` is. Chebyshev /
augmented Chebyshev (`min_j w_j (f_j - z_j)`) fixes that but is not a linear
functional, so `ScalarizedPosteriorTransform` does not apply; it needs an MC
acquisition with a `GenericMCObjective`. Small, but a separate change.

## Scope options

1. `ScalarizedGP` in `gp.py` only (+ `__all__` export, + tests) -- ~80 lines,
   `acquisition.py` untouched.
2. That plus the `posterior_transform` / scalarized-incumbent plumbing in
   `AcquisitionFunction` -- the full path to a `w`-conditioned acquisition.
3. Either, plus the Chebyshev variant.
