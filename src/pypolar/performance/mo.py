"""Multi-objective metrics against a known groundtruth: the hypervolume the
model's inferred Pareto set attains, and the hypervolume the queried points
attain, both as a fraction of the truth's own."""

import numpy as np

from pypolar.feedback.synthetic import MOSyntheticOracle
from pypolar.optimization.objectives import DecoupledObjectives
from pypolar.utils.pareto import (gd_plus, get_nondominated, get_nondominated_tol,
                                  get_pareto_statistics,
                                  hypervolume_from_nondominated, reference_point)


def _signs(objectives, ground_truth: MOSyntheticOracle):
    """(m,) direction of each objective
    """
    signs = objectives.signs
    if len(signs) != len(ground_truth):
        raise ValueError(f'{len(signs)} objectives {objectives.names} against '
                         f'{len(ground_truth)} truth columns; they are '
                         "positional, so one per column in the truth's order")

    return signs


def _attained_fraction(
    pred_values       : np.ndarray,
    ground_truth    : MOSyntheticOracle,
    # signs           : np.ndarray
):
    """Fraction of the truth's own hypervolume that a set of objective
    vectors from the truth function covers (usually, the predicted Pareto
    optimal actions that the GP predicted).

    Args:
        true_objs: (n, m) *true* objective values, in the truth's own raw units in maximization space
        ground_truth: the oracle, carrying `ref_point` and the scan's
            `max_hypervolume`.
        signs: (m,) +1 on a maximized objective, -1 on a minimized one.
    """
    # values = signs * np.asarray(true_objs, dtype=float) # force than attained frac is always in max space and build a transform for this in objs
    front  = get_nondominated(pred_values)
    hv = hypervolume_from_nondominated(
        ref_point - pred_values[front]
    )
    return hv / max_hv

    return hypervolume_from_nondominated(
        signs * ground_truth.ref_point - values[front]
    ) / ground_truth.max_hypervolume(signs)


def normalized_hypervolume_regret( # TODO: potentially add a k-sampling to avoid the fact that hypervolume strictly increases with more points
    # gp,
    # ground_truth    : MOSyntheticOracle,
    pred_values     : np.ndarray,
    objectives, # plr.DecoupledObjectives
    max_hv          : np.ndarray,
    tol             : float = 0.0,
    ref_point       : np.ndarray = None
):
    """The inference hypervolume regret of a model: one minus the ratio between
    the ground truth's max hypervolume and the hypervolume produced by the GP's
    predicted Pareto optimal actions. Between 0 and 1 (0 is better).

    Args:
        pred_values: raw values predicted to be on the front
        objectives: a DecoupledObjectives structure holding the maximization direction of each objective
        max_hv: the maximum possible hypervolume (replace with groundtruth?)
        tol: tolerance for Pareto dominated exclusion
        ref: the reference point for hypervolume calc. if None, it is inferred from objectives and pred_values
    Returns:
        The regret, as a fraction of the truth's hypervolume in [0, 1].
    """

    # values = signs * np.asarray(true_objs, dtype=float) # force than attained frac is always in max space and build a transform for this in objs
    if ref_point is None:
        ref_point = reference_point(
            values = pred_values,
            maximize = objectives.maximize
        )

    ms_values = objectives.maximization_space(pred_values)
    front  = get_nondominated(ms_values)
    hv = hypervolume_from_nondominated(
        ref_point - ms_values[front]
    )
    return hv / max_hv

    mu, _ = gp.posterior_at(ground_truth.scan_actions)
    front = get_nondominated(mu, tol)
    
    max_hv = ground_truth.max_hypervolume()
    return 1.0 - _attained_fraction(
        pred_values=
    )



    signs = _signs(gp.objectives, ground_truth)
    mu, _ = gp.posterior_at(ground_truth.scan_actions)
    front = get_nondominated_tol(mu, tol)

    return 1.0 - _attained_fraction(ground_truth.scan_values[front],
                                    ground_truth, signs)


def attained_hypervolume_regret(
    raw_actions     : np.ndarray,
    ground_truth    : MOSyntheticOracle,
    objectives      : DecoupledObjectives
):
    """The hypervolume regret of the points a run has actually queried.

    Args:
        raw_actions: (N, d) every action queried so far, in raw action units.
        ground_truth: the oracle, evaluated with `noise=False` so a lucky draw
            cannot flatter the score.
        objectives: the `DecoupledObjectives` whose `signs` the truth is scored
            in, one per truth column in the truth's own order.

    Returns:
        The regret, as a fraction of the truth's hypervolume.
    """
    return 1.0 - _attained_fraction(
        ground_truth(raw_actions, noise=False), ground_truth,
        _signs(objectives, ground_truth)
    )



def front_alignment_regret(
    gp,
    ground_truth    : MOSyntheticOracle,
    tol             : float = 0.0
):
    """The GD+ indicator. How far a model's inferred Pareto front sits from the
    true Pareto front (in 'objective space').

    Args:
        gp: a `DecoupledMOGP`, or anything whose `posterior_at` returns
            (n, m) means in maximization space.
        ground_truth: the oracle, whose scan supplies both the true front and
            the ranges the distance is stated in.
        tol: relaxes non-domination on the *estimated* front only (see
            `get_nondominated_tol`), keeping near-ties in the recommendation.
            The true front stays strict, since relaxing it would hand the
            indicator extra targets to be close to.

    Returns:
        The regret, in fractions of the truth's objective ranges. 0.0 exactly
        when every recommended action lies on the true front.
    """
    signs  = _signs(gp.objectives, ground_truth)
    mu, _  = gp.posterior_at(ground_truth.scan_actions)
    values = signs * ground_truth.scan_values
    
    corners = np.stack([signs * np.array([o.sample_min for o in ground_truth.objectives]),
                        signs * np.array([o.sample_max for o in ground_truth.objectives])])

    return gd_plus(
        F           = -values[get_nondominated_tol(mu, tol)],
        true_front  = -values[get_nondominated(values)],
        ideal       = -corners.max(axis=0),
        nadir       = -corners.min(axis=0)
    )

def groundtruth_hypervolume(estimated_objs, true_objs, tol=0.0):
    """Groundtruth hypervolume attained under the predicted Pareto-opimal action set.

    ``tol`` relaxes non-domination by a fraction of each objective's range, so
    actions that are only "just barely" dominated in the estimated space stay on
    the estimated front (see :func:`get_nondominated_tol`). ``tol=0`` reproduces
    strict domination.
    """
    est_nidxs = get_nondominated_tol(estimated_objs, tol)
    true_nidxs = get_nondominated_tol(true_objs, tol)
    # est_hv = hypervolume_from_nondominated(true_objs[est_nidxs])
    est_hv, _  = get_pareto_statistics(true_objs[est_nidxs])
    true_hv, _ = get_pareto_statistics(true_objs[true_nidxs])
    return est_hv / true_hv

def pareto_overlay(estimated_objs, true_objs, tol=0.0):
    """Jaccard overlap between the estimated and true Pareto fronts.

    An action counts against the score when it is (1) estimated-optimal but not
    truly optimal and (2) when it is truly optimal but missed by the estimate.
    Returns a value in ``[0, 1]`` (1.0 = the fronts coincide exactly).

    ``tol`` relaxes non-domination by a fraction of each objective's range (see
    :func:`get_nondominated_tol`), applied to both fronts, so near-ties are not
    penalized. ``tol=0`` reproduces the strict Jaccard overlap.
    """
    est = set(get_nondominated_tol(estimated_objs, tol).tolist())
    true = set(get_nondominated_tol(true_objs, tol).tolist())
    if not est:
        raise Exception('No estimated pareto front')
    return len(est & true) / len(est | true)
