"""Multi-objective Bayesian optimization over a continuous 2D action space.

The multi-objective counterpart of `bo_continuous_2d_botorch.py`, and the
BoTorch parallel to pypolar's `MultiObjectiveGP` + `QNEHVISampler`:

| pypolar                          | BoTorch                                    |
|----------------------------------|--------------------------------------------|
| `MultiObjectiveGP`               | `ModelListGP`                              |
| `.gps` (list, one per objective) | the models passed to `ModelListGP(*models)`|
| `.setup(...)` / `.fit()`         | rebuild the `ModelListGP`                  |
| `.mu_at(X)` -> list of means     | `posterior(X).mean` -> an (n, m) block     |
| `QNEHVISampler`                  | `qLogNoisyExpectedHypervolumeImprovement`  |
| `optimize_acqf_discrete`         | `optimize_acqf`  (continuous)              |

`ModelListGP` is exactly what `MultiObjectiveGP` is: independent single-output
GPs over a shared action space, with per-objective hyperparameters. Neither
models correlations between objectives. The difference is that `ModelListGP` is
itself a BoTorch `Model` whose posterior is (n, m) rather than a list of n-vectors,
which is what lets an acquisition function reason about the objectives jointly.
`QNEHVISampler` already builds one internally — this script differs in that the
acquisition is maximized over the *continuous* box rather than enumerated over a
grid, so no action is ever snapped to a lattice.

`pypolar` is still used for the problem (`BoundedIdealPoint`,
`NoisyRegressionOracle`, `derive_gp_hyperparams`). Nothing in the loop is
pypolar's, including the metrics: hypervolume comes from BoTorch's
`DominatedPartitioning` and the fronts from `is_non_dominated`.

The objective is two competing ideal points — objective 1 peaks at one corner of
the box, objective 2 at another — so no single action maximizes both and the
Pareto set is the ridge between them. That is the point of a multi-objective run:
the answer is a *set* of actions, not one.

Run from the repo root:

    python scratch/bo_continuous_mo_2d.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from botorch.acquisition.multi_objective import (
    qLogNoisyExpectedHypervolumeImprovement,
)
from botorch.models import ModelListGP, SingleTaskGP
from botorch.optim import optimize_acqf
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils.multi_objective.box_decompositions.dominated import (
    DominatedPartitioning,
)
from botorch.utils.multi_objective.pareto import is_non_dominated
from botorch.utils.sampling import draw_sobol_samples
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.means import ZeroMean

import pypolar as plr

LOW             = np.array([-2.0, -2.0])
HIGH            = np.array([ 2.0,  2.0])
IDEAL_POINTS    = np.array([[-0.90,  0.60],   # objective 1 peaks here
                            [ 1.10, -0.80]])  # objective 2 peaks here
REWARD_RANGE    = (-1.0, 1.0)               # BoundedIdealPoint output bounds
NOISE_STD       = 0.05
NUM_OBJS        = len(IDEAL_POINTS)
NUM_QUERIES     = 32
LOG2_INITIAL    = 3                         # 2^3 = 8 initial design points
RAW_SAMPLES     = 256                       # stage-1 Sobol samples per query
NUM_RESTARTS    = 8                         # stage-2 L-BFGS-B starting points
MC_SAMPLES      = 128                       # qLogNEHVI quasi-MC samples
LOG2_REFERENCE  = 14                        # 2^14 samples for the true front
SEED            = 95
DTYPE           = torch.float64
BOUNDS          = torch.as_tensor(np.stack([LOW, HIGH]), dtype=DTYPE)  # (2, d)

# BoundedIdealPoint is bounded below by `lower_bound`, so the worst attainable
# outcome is known a priori and the reference point can be fixed. That keeps the
# hypervolume comparable across iterations and against the true optimum.
# `infer_reference_point` (what QNEHVISampler defaults to) is for when the
# objective scale is not known up front.
REF_POINT       = torch.full((NUM_OBJS,), REWARD_RANGE[0], dtype=DTYPE)


def build_model_list(X, Y, sigma2, signal_variances, length_scales):
    """One fixed-hyperparameter GP per objective, wrapped as a single model.

    The direct parallel to `MultiObjectiveGP._build` followed by `.setup()`:
    each objective gets its own `SingleTaskGP` over the same inputs, with its own
    kernel hyperparameters, and `ModelListGP` presents the collection as one
    `Model` with `num_outputs == m`.

    Each per-objective model is configured exactly as in the single-objective
    script — `train_Yvar` pins the noise, `ZeroMean` matches the other backends,
    and `outcome_transform=None` is explicit because recent BoTorch defaults to
    `Standardize`, which would rescale `Y` out from under `signal_variance`.

    Args:
        X: (n, d) observed actions.
        Y: (n, m) observed values, one column per objective.
        sigma2: observation noise variance, shared across objectives.
        signal_variances: length-m kernel amplitudes.
        length_scales: length-m kernel lengthscales.

    Returns:
        A `ModelListGP` in eval mode.
    """
    train_X = torch.as_tensor(X, dtype=DTYPE)
    models = []

    for j, (sv, ls) in enumerate(zip(signal_variances, length_scales)):
        train_Y = torch.as_tensor(Y[:, j], dtype=DTYPE).reshape(-1, 1)

        covar = ScaleKernel(RBFKernel()).to(DTYPE)
        # float64 tensors, not Python floats: gpytorch inverts the Positive
        # constraint at the value's own dtype, and a float would land there as
        # float32.
        covar.base_kernel.lengthscale = torch.tensor(ls, dtype=DTYPE)
        covar.outputscale = torch.tensor(sv ** 2, dtype=DTYPE)

        models.append(SingleTaskGP(
            train_X=train_X,
            train_Y=train_Y,
            train_Yvar=torch.full_like(train_Y, sigma2),
            covar_module=covar,
            mean_module=ZeroMean(),
            outcome_transform=None,
            input_transform=None,
        ))

    return ModelListGP(*models).eval()


def maximize(acqf):
    """Two-stage maximization of an acquisition function over the box.

    Identical to the single-objective script: `optimize_acqf` scans
    `raw_samples` Sobol points, picks `num_restarts` starting points, and runs
    box-constrained L-BFGS-B on each. qLogNEHVI is a Monte-Carlo acquisition, so
    its gradients come from the reparameterization trick — the quasi-MC base
    samples are fixed inside the sampler, which is what makes the objective
    deterministic enough for a quasi-Newton method to descend.

    Args:
        acqf: an `AcquisitionFunction`, scored on `(b, q, d)` batches.

    Returns:
        The (d,) maximizer, as numpy.
    """
    candidate, _ = optimize_acqf(
        acq_function=acqf,
        bounds=BOUNDS,
        q=1,                            # one action per query, not a batch
        num_restarts=NUM_RESTARTS,
        raw_samples=RAW_SAMPLES,
    )
    return candidate.squeeze(0).detach().numpy()


def hypervolume(Y):
    """Hypervolume dominated by ``Y`` above ``REF_POINT``.

    `DominatedPartitioning` splits the dominated region into disjoint boxes and
    sums their volumes, which is exact rather than sampled.

    Args:
        Y: (n, m) array of objective vectors.

    Returns:
        The dominated hypervolume, as a float.
    """
    Y = torch.as_tensor(np.atleast_2d(Y), dtype=DTYPE)
    return DominatedPartitioning(ref_point=REF_POINT, Y=Y).compute_hypervolume().item()


def posterior_mean_at(model, X, chunk=2048):
    """Per-objective posterior means at arbitrary points, as an (n, m) array.

    The `ModelListGP` counterpart of `MultiObjectiveGP.mu_at`. Points are fed as
    `q=1` batch elements so each posterior is 1x1 per objective and gpytorch
    never forms the n x n test-test block.

    Args:
        model: a `ModelListGP`.
        X: (n, d) array of points.
        chunk: batch elements evaluated at once.

    Returns:
        (n, m) array of posterior means.
    """
    X = torch.as_tensor(np.atleast_2d(X), dtype=DTYPE).unsqueeze(1)  # (n, 1, d)
    with torch.no_grad():
        out = [model.posterior(X[i:i + chunk]).mean.squeeze(1)
               for i in range(0, X.shape[0], chunk)]
    return torch.cat(out).numpy()


def plot(X, Y_true, hv_history, true_hv, dense_X, dense_Y, path):
    """Pareto set in action space, Pareto front in objective space, hypervolume.

    Args:
        X: (n, d) queried actions, initial design first.
        Y_true: (n, m) noiseless objective values at those actions.
        hv_history: attained hypervolume after each query.
        true_hv: hypervolume of the densely sampled true front.
        dense_X: (K, d) dense sample of the box.
        dense_Y: (K, m) noiseless objectives on that sample.
        path: where to write the figure.
    """
    fig, axs = plt.subplots(ncols=3, figsize=(16, 5), layout='constrained')
    n_init = 2 ** LOG2_INITIAL
    front = is_non_dominated(torch.as_tensor(dense_Y, dtype=DTYPE)).numpy()
    queried_front = is_non_dominated(torch.as_tensor(Y_true, dtype=DTYPE)).numpy()

    # Action space: the true Pareto set is the ridge between the two ideal
    # points, which is what the queries should concentrate on.
    axs[0].scatter(*dense_X[~front].T, s=1, color='0.88', label='Dominated')
    axs[0].scatter(*dense_X[front].T, s=4, color='tab:orange',
                   label='True Pareto set')
    axs[0].scatter(*X[:n_init].T, marker='o', facecolor='none',
                   edgecolor='tab:blue', linewidth=1.4, s=45,
                   label='Sobol initial design')
    axs[0].scatter(*X[n_init:].T, marker='o', color='k', s=26,
                   label='qLogNEHVI queries')
    axs[0].scatter(*IDEAL_POINTS.T, marker='X', color='w', edgecolor='k',
                   linewidth=0.8, s=170, zorder=3, label='Per-objective optima')
    axs[0].set_xlim(LOW[0], HIGH[0])
    axs[0].set_ylim(LOW[1], HIGH[1])
    axs[0].set_xlabel('$a_1$')
    axs[0].set_ylabel('$a_2$')
    axs[0].set_aspect('equal')
    axs[0].set_title('Action space')
    axs[0].legend(loc='upper left', framealpha=0.9, fontsize=8)

    # Objective space: the attainable set, its upper-right boundary, and where
    # the queries landed on it.
    axs[1].scatter(*dense_Y[~front].T, s=1, color='0.88', label='Attainable')
    order = np.argsort(dense_Y[front][:, 0])
    axs[1].plot(*dense_Y[front][order].T, color='tab:orange', linewidth=2,
                label='True Pareto front')
    axs[1].scatter(*Y_true[~queried_front].T, marker='o', color='k', s=26,
                   label='Queried (dominated)')
    axs[1].scatter(*Y_true[queried_front].T, marker='o', color='tab:blue',
                   edgecolor='k', linewidth=0.6, s=70,
                   label='Queried (on front)')
    axs[1].scatter(*REF_POINT.numpy(), marker='s', color='tab:red', s=45,
                   zorder=4, clip_on=False, label='Reference point')
    axs[1].set_xlabel('Objective 1')
    axs[1].set_ylabel('Objective 2')
    axs[1].set_title('Objective space')
    axs[1].legend(loc='lower left', framealpha=0.9, fontsize=8)

    # Hypervolume is the standard multi-objective progress measure: it rewards
    # both getting close to the front and spreading along it, which neither
    # objective alone captures.
    axs[2].plot(np.arange(1, len(hv_history) + 1),
                np.array(hv_history) / true_hv, linewidth=2, color='k')
    axs[2].axhline(1.0, color='tab:orange', linestyle='--', linewidth=1.5,
                   label='True front')
    axs[2].axvline(n_init, color='grey', linestyle='--', linewidth=1)
    axs[2].annotate('qLogNEHVI takes over', xy=(n_init + 0.6, 0.06), fontsize=8,
                    xycoords=('data', 'axes fraction'), color='grey',
                    ha='left', va='bottom')
    axs[2].set_xlabel('Query')
    axs[2].set_ylabel('Attained / true hypervolume')
    axs[2].set_title('Hypervolume attained')
    axs[2].grid(alpha=0.2)
    axs[2].legend(loc='lower right', fontsize=8)

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    print(f'Saved figure to {path.resolve()}')


def main():
    rng = np.random.default_rng(SEED)
    torch.manual_seed(SEED)  # optimize_acqf's Sobol draws come from torch

    length_scale, signal_variance, precision = plr.derive_gp_hyperparams(
        domain_size    = float(np.max(HIGH - LOW)),
        expected_range = float(REWARD_RANGE[1] - REWARD_RANGE[0]),
        noise_var      = NOISE_STD ** 2,
    )
    # pypolar states the likelihood as `precision * ||Sr - y||^2`, a Gaussian
    # NLL with sigma^2 = 1/(2*precision). BoTorch wants the variance directly.
    sigma2 = 1.0 / (2.0 * precision)
    signal_variances = [signal_variance] * NUM_OBJS
    length_scales    = [length_scale] * NUM_OBJS

    # A (m, d) stack of ideal points gives one objective per row.
    groundtruth = plr.BoundedIdealPoint(
        w           = IDEAL_POINTS,
        delta       = 0.6,
        gamma       = 0.0,
        lower_bound = REWARD_RANGE[0],
        upper_bound = REWARD_RANGE[1],
    )
    oracle = plr.NoisyRegressionOracle(groundtruth, NOISE_STD, rng)

    # A dense sample stands in for the true front: the box is continuous, so the
    # attainable set has no closed form to compare against.
    dense_X = draw_sobol_samples(
        BOUNDS, n=2 ** LOG2_REFERENCE, q=1, seed=SEED + 1
    ).squeeze(1).numpy()
    dense_Y = groundtruth(dense_X)
    true_hv = hypervolume(dense_Y[is_non_dominated(
        torch.as_tensor(dense_Y, dtype=DTYPE)).numpy()])

    X = draw_sobol_samples(
        BOUNDS, n=2 ** LOG2_INITIAL, q=1, seed=SEED
    ).squeeze(1).numpy()
    Y = np.array([oracle.query(x) for x in X])

    # Tracked on the noiseless objective so oracle noise cannot flatter it.
    hv_history = [hypervolume(groundtruth(X[:i + 1])) for i in range(len(X))]

    for _ in range(NUM_QUERIES - len(X)):
        model = build_model_list(X, Y, sigma2, signal_variances, length_scales)

        # Noisy EHVI, not plain EHVI: the front implied by the observations is
        # itself uncertain under noise, so the acquisition marginalizes over the
        # posterior at the already-queried actions rather than treating the
        # observed values as the true front.
        acqf = qLogNoisyExpectedHypervolumeImprovement(
            model=model,
            ref_point=REF_POINT.tolist(),
            X_baseline=torch.as_tensor(X, dtype=DTYPE),
            sampler=SobolQMCNormalSampler(
                sample_shape=torch.Size([MC_SAMPLES]),
                seed=int(rng.integers(2 ** 32)),
            ),
            prune_baseline=True,
        )
        action = maximize(acqf)

        X = np.vstack([X, action])
        Y = np.vstack([Y, oracle.query(action)])
        hv_history.append(hypervolume(groundtruth(X)))

    model = build_model_list(X, Y, sigma2, signal_variances, length_scales)

    # The recommendation is a set, not a point: every queried action whose
    # posterior mean vector is non-dominated by any other.
    mu = posterior_mean_at(model, X)
    recommended = np.flatnonzero(
        is_non_dominated(torch.as_tensor(mu, dtype=DTYPE)).numpy()
    )
    Y_true = groundtruth(X)

    print(f'attained hypervolume  {hv_history[-1]:.4f} '
          f'({hv_history[-1] / true_hv:.1%} of the true front)')
    print(f'predicted Pareto set  {len(recommended)} of {len(X)} queried actions')
    for i in recommended[np.argsort(mu[recommended, 0])]:
        print(f'   a = [{X[i, 0]: .4f}, {X[i, 1]: .4f}]   '
              f'f(a) = [{Y_true[i, 0]: .4f}, {Y_true[i, 1]: .4f}]')

    plot(X, Y_true, hv_history, true_hv, dense_X, dense_Y,
         Path('scratch/output/bo_continuous_mo_2d.png'))


if __name__ == '__main__':
    main()
