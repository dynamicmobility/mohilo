from pypolar.utils.pareto import get_nondominated, hypervolume_from_nondominated, get_pareto_statistics

def groundtruth_hypervolume(estimated_objs, true_objs):
    """Groundtruth hypervolume attained under the predicted Pareto-opimal action set"""
    est_nidxs = get_nondominated(estimated_objs)
    true_nidxs = get_nondominated(true_objs)
    # est_hv = hypervolume_from_nondominated(true_objs[est_nidxs])
    est_hv, _  = get_pareto_statistics(true_objs[est_nidxs])
    true_hv, _ = get_pareto_statistics(true_objs[true_nidxs])
    return est_hv / true_hv
    
def pareto_overlay(estimated_objs, true_objs):
    """Jaccard overlap between the estimated and true Pareto fronts.

    An action counts against the score when it is (1) estimated-optimal but not 
    truly optimal and (2) when it is truly optimal but missed by the estimate. 
    Returns a value in ``[0, 1]`` (1.0 = the fronts coincide exactly).
    """
    est = set(get_nondominated(estimated_objs).tolist())
    true = set(get_nondominated(true_objs).tolist())
    print(est)
    print(true)
    print()
    if not est:
        raise Exception('No estimated pareto front')
    return len(est & true) / len(est | true)