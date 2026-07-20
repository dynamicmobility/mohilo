from pypolar.utils.pareto import get_nondominated_tol, hypervolume_from_nondominated, get_pareto_statistics

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
