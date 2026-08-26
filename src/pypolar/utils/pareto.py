from pymoo.indicators.hv import HV
from pymoo.indicators.gd_plus import GDPlus
from pymoo.util.normalization import normalize
from pymoo.indicators.spacing import SpacingIndicator
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
import numpy as np

def get_nondominated(F, epsilon=None):
    nds = NonDominatedSorting(epsilon=epsilon)
    front_indices = nds.do(-F, only_non_dominated_front=True)
    return front_indices

def get_nondominated_tol(F, tol=0.0, block=512):
    """Non-dominated front with a tolerance for "just barely" dominated points.

    ``tol`` is a fraction of each objective's range (so it is scale-invariant
    across objectives and configs). A point is dropped only when some other
    point beats it by more than ``tol`` in *every* objective; points dominated
    by a hair in even one objective (near-ties) are retained. This keeps
    near-optimal actions on the front instead of excluding them for tiny
    imperfections in the learned objectives.

    With ``tol == 0`` this defers to :func:`get_nondominated` so existing
    behavior is preserved exactly.

    Args:
        F: ``(n_points, n_objectives)`` array of objective vectors (higher is
            better).
        tol: tolerance as a fraction of each objective's range (e.g. ``0.02``
            = within 2% counts as tied). ``0`` reproduces strict domination.

    Returns:
        Indices into ``F`` of the (tolerant) non-dominated points.
    """
    if tol <= 0:
        return get_nondominated(F)

    F = np.asarray(F, dtype=float)
    mn = F.min(axis=0)
    mx = F.max(axis=0)
    rng = np.where(mx > mn, mx - mn, 1.0)
    F_norm = (F - mn) / rng
    n, d = F_norm.shape

    # Point j beats point i by more than tol in every objective -> i is dropped.
    # Blocked over i and accumulated per objective so peak memory is
    # O(block * n) rather than O(n^2 * d).
    keep = np.empty(n, dtype=bool)
    for start in range(0, n, block):
        stop = min(start + block, n)
        dominated = np.ones((stop - start, n), dtype=bool)
        for k in range(d):
            dominated &= (F_norm[None, :, k] - F_norm[start:stop, None, k]) > tol
        keep[start:stop] = ~dominated.any(axis=1)
    return np.where(keep)[0]

def hypervolume_from_nondominated(F_min):
    """Compute the hypervolume of a non-dominated front in minimization space.

    Uses the origin as the reference point, so callers must pass an
    already-non-dominated, already-negated front (see
    ``get_pareto_statistics`` for the typical pipeline).

    Args:
        F_min: ``(n_points, n_objectives)`` array of points in
            minimization space (i.e. negated objective values).

    Returns:
        Hypervolume as a float, computed by ``pymoo.indicators.hv.HV``.
    """
    ref_point = np.zeros(F_min.shape[1])

    hv = HV(ref_point=ref_point)
    hypervolume = hv(F_min)
    return hypervolume

def sparsity_from_normalized_nondominated(F_min_norm):
    spacing = SpacingIndicator()
    sparsity = spacing(F_min_norm)
    return sparsity

def get_pareto_statistics(F):
    """Compute hypervolume and sparsity for a set of objective vectors.

    Extracts the non-dominated front of ``F`` (treated as a maximization
    problem), normalizes it, converts to minimization space, and returns
    the resulting hypervolume and spacing-based sparsity. When the front
    contains only one point, it is duplicated so the spacing indicator
    has at least two points to work with.

    Args:
        F: ``(n_points, n_objectives)`` array of objective vectors
            (higher is better).

    Returns:
        Tuple ``(hypervolume, sparsity)``.
    """
    F_max = F[get_nondominated(F)]
    F_norm = normalize(F_max.copy())

    # Convert to minimization
    F_min = -F_max.copy()
    F_min_norm = -F_norm.copy()

    if F_min_norm.shape[0] == 1:
        # Sparsity always needs 2 points to calculate
        F_min_norm = np.repeat(F_min_norm, 2, axis=0)
    return (
        hypervolume_from_nondominated(F_min), 
        sparsity_from_normalized_nondominated(F_min_norm)
    )

def gd_plus(F, true_front, ideal, nadir):
    """Normalized GD+ of a set against a known Pareto front, in minimization space.

    Args:
        F: ``(n, m)`` points to score, in minimization space.
        true_front: ``(k, m)`` the true Pareto front, same space.
        ideal, nadir: ``(m,)`` the best and worst value per objective.

    Returns:
        The indicator as a float, 0 when every point of ``F`` is on the front.
    """
    return GDPlus(true_front, zero_to_one=True, ideal=ideal, nadir=nadir)(F)
