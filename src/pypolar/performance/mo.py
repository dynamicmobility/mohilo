"""Multi-objective metrics against a known groundtruth: the hypervolume an
inferred Pareto set attains and the hypervolume the queried points attain, both
as a fraction of the truth's own, plus how far an inferred front sits from the
true one.

The hypervolume metrics take objective vectors rather than a model, so nothing
here depends on how they were produced. Which way each column is optimized --
and hence which region a hypervolume measures -- comes from a
`DecoupledObjectives`, since a BoTorch truth states every one of its own
columns in the minimizing sense and carries no direction of its own."""

import numpy as np

from pypolar.feedback.synthetic import MOSyntheticOracle
from pypolar.optimization.objectives import DecoupledObjectives
from pypolar.utils.pareto import (gd_plus, get_nondominated, get_nondominated_tol,
                                  igd_plus,
                                  get_pareto_statistics,
                                  hypervolume_from_nondominated, reference_point)


def _attained_fraction(
    raw_values      : np.ndarray,
    objectives      : DecoupledObjectives,
    ground_truth    : MOSyntheticOracle,
    tol             : float = 0.0,
    ref_point       : np.ndarray = None
):
    """Fraction of the truth's own hypervolume that a set of objective vectors
    attains.

    The reference is resolved here and handed to `max_hypervolume`, so the
    numerator and the denominator are always measured from the same one -- a
    hypervolume ratio is meaningless otherwise.

    Args:
        raw_values: (n, m) objective vectors in the objectives' own raw units,
            one column per objective in the objectives' order.
        objectives: the `DecoupledObjectives` whose directions the values are
            read in, one per truth column in the truth's own order.
        ground_truth: the oracle, whose scan supplies the denominator.
        tol: relaxes non-domination by a fraction of each objective's range
            (see `get_nondominated_tol`), keeping near-ties on the front.
            Optimistic, since hypervolume only grows as the set does.
        ref_point: (m,) the worst value per objective that still counts, in raw
            units. None reads it off the truth's own scan.

    Returns:
        The fraction, 1.0 when the set attains the whole of the truth's.
    """
    if ref_point is None:
        ref_point = reference_point(values   = ground_truth.scan_values,
                                    maximize = objectives.maximize)

    values = objectives.maximization_space(raw_values)
    ref    = objectives.maximization_space(ref_point)[0]
    front  = get_nondominated_tol(values, tol)
    return hypervolume_from_nondominated(
        ref - values[front]
    ) / ground_truth.max_hypervolume(objectives, ref_point)


def normalized_hypervolume_regret( # TODO: potentially add a k-sampling to avoid the fact that hypervolume strictly increases with more points
    raw_actions     : np.ndarray,
    ground_truth    : MOSyntheticOracle,
    objectives      : DecoupledObjectives,
    tol             : float = 0.0,
    ref_point       : np.ndarray = None
):
    """The inference hypervolume regret of the front a run recommends: one
    minus the fraction of the truth's own hypervolume those actions attain.
    Between 0 and 1 (0 is better).

    The actions are scored on the truth's own values there, not on the values a
    run claims for them. That is what bounds the score: a run scored on its own
    claims can exceed the truth's best achievable front, since a lucky draw of
    the observation noise reports a point as better than anything the box holds,
    and the regret then comes out negative.

    Nothing here touches a model. Which actions a run recommends -- a posterior
    argmax, an evolutionary population's own front, anything -- is the caller's.

    Args:
        raw_actions: (n, d) the actions nominated as the front, in raw action
            units. Evaluated with `noise=False`.
        ground_truth: the oracle, whose scan is the denominator.
        objectives: the `DecoupledObjectives` stating which way each column is
            optimized, one per truth column in the truth's own order.
        tol: relaxes non-domination among the nominated actions (see
            `get_nondominated_tol`). Optimistic, since hypervolume only grows
            as the set does.
        ref_point: (m,) hypervolume reference in raw units, None to read it off
            the truth's own scan.

    Returns:
        The regret, as a fraction of the truth's hypervolume in [0, 1].
    """
    return 1.0 - _attained_fraction(
        ground_truth(raw_actions, noise=False), objectives, ground_truth,
        tol, ref_point
    )


def attained_hypervolume_regret(
    raw_actions     : np.ndarray,
    ground_truth    : MOSyntheticOracle,
    objectives      : DecoupledObjectives,
    tol             : float = 0.0,
    ref_point       : np.ndarray = None
):
    """The hypervolume regret of the points a run has actually queried.

    The same measurement `normalized_hypervolume_regret` makes, over every
    action queried rather than over the front the run would recommend. The two
    differ only in what is handed to them, which is the whole distinction: one
    scores what a run has *found*, the other what it would *hand back*.

    Args:
        raw_actions: (N, d) every action queried so far, in raw action units.
        ground_truth, objectives, tol, ref_point: as
            `normalized_hypervolume_regret` takes them.

    Returns:
        The regret, as a fraction of the truth's hypervolume.
    """
    return normalized_hypervolume_regret(raw_actions, ground_truth, objectives,
                                         tol, ref_point)


def _indicator_inputs(
    raw_actions     : np.ndarray,
    ground_truth    : MOSyntheticOracle,
    objectives      : DecoupledObjectives
):
    """The three arrays GD+ and IGD+ both take, in minimization space.

    Returns `(F, true_front, ideal, nadir)`: the nominated actions' true values,
    the truth's own front, and the per-objective range the distances are stated
    in. All four come from the one `scan_values`, so the range normalizing a
    distance is the range of the very points being scored -- each oracle's own
    `sample_min`/`sample_max` come from a scan drawn at `seed + i` rather than
    the shared `seed`, which is a different point set for every objective past
    the first.
    """
    values   = objectives.maximization_space(ground_truth.scan_values)
    attained = objectives.maximization_space(
        ground_truth(raw_actions, noise=False))

    # in maximization space the largest is best, so it is the ideal once negated
    return (-attained,
            -values[get_nondominated(values)],
            -values.max(axis=0),
            -values.min(axis=0))


def front_alignment_regret(
    raw_actions     : np.ndarray,
    ground_truth    : MOSyntheticOracle,
    objectives      : DecoupledObjectives
):
    """The GD+ indicator. How far the front a run recommends sits from the true
    Pareto front, in objective space -- its *precision*.

    Averaged over the nominated actions, so one action off the front raises it
    and a gap in coverage does not: a single action sitting exactly on the front
    scores 0.0 while covering none of it. `front_coverage_regret` is the
    complement that reads the other way, and the two are only meaningful as a
    pair.

    The recommended actions are scored on the *truth's* own values there, which
    is the convention GD+ is stated in: it measures whether the right actions
    were nominated, not whether their values were predicted well. Which actions
    those are -- and how they were chosen, posterior scan or otherwise -- is
    the caller's, so no model reaches this.

    Args:
        raw_actions: (n, d) the actions nominated as the front, in raw action
            units. Evaluated with `noise=False`, so a lucky draw cannot flatter
            the score.
        ground_truth: the oracle, whose scan supplies both the true front and
            the ranges the distance is stated in.
        objectives: the `DecoupledObjectives` stating which way each column is
            optimized, one per truth column in the truth's own order.

    Returns:
        The regret, in fractions of the truth's objective ranges. 0.0 exactly
        when every recommended action lies on the true front.
    """
    F, true_front, ideal, nadir = _indicator_inputs(raw_actions, ground_truth,
                                                    objectives)
    return gd_plus(F=F, true_front=true_front, ideal=ideal, nadir=nadir)


def front_coverage_regret(
    raw_actions     : np.ndarray,
    ground_truth    : MOSyntheticOracle,
    objectives      : DecoupledObjectives
):
    """The IGD+ indicator. How much of the true Pareto front a run's recommended
    front reaches -- its *coverage*.

    `front_alignment_regret` the other way round. The nearest nominated action
    is taken per *front* point rather than the nearest front point per nominated
    action, which inverts what the score is blind to: an action nearest to
    nothing contributes nothing, so a stray costs exactly zero here, while a
    stretch of the front no action comes near costs the distance to it.

    That is the pair's whole point. A small, tightly clustered front is precise
    but covers little, and GD+ alone calls it perfect; a broad front with some
    members off the mark offers a run more usable options, and IGD+ alone calls
    it perfect. Neither is a verdict on its own.

    Args:
        raw_actions, ground_truth, objectives: as `front_alignment_regret`
            takes them.

    Returns:
        The regret, in fractions of the truth's objective ranges. 0.0 when
        every point of the true front has a recommended action on it.
    """
    F, true_front, ideal, nadir = _indicator_inputs(raw_actions, ground_truth,
                                                    objectives)
    return igd_plus(F=F, true_front=true_front, ideal=ideal, nadir=nadir)


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
