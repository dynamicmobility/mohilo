"""Actions to measure when testing whether a fit's Pareto front really does beat
what it calls dominated: three from the front, three from the anti-front, three
from neither."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist

import pypolar as plr
from pypolar.experiment.dataset import ExperimentDataset
from pypolar.utils.plotting import plot_mo_space
from scripts.plotting.mo_exp_gif import inferred_front, measured_at, true_front

DATASET = Path('hilo/output/experiments/20260827_001745/MT01.json')
OUTPUT  = Path('hilo/output/front_samples.png')

K       = 3        # actions per set
SCAN    = 4096     # Sobol points the fronts are read off
SEED    = 95
DPI     = 200


def peel(mu, k, sign):
    """Indices of at least k points on the `sign` end of `mu`: non-dominated
    layers peeled until enough have accumulated. A smooth posterior often puts
    a single point on the first anti-Pareto layer.

    Args:
        mu: (n, m) objective values, higher is better.
        sign: +1 for the Pareto front, -1 for the anti-front.
    """
    left, out = np.arange(len(mu)), np.array([], dtype=int)
    while len(out) < k and len(left):
        layer = left[plr.get_nondominated(sign * mu[left])]
        out   = np.concatenate([out, layer])
        left  = np.setdiff1d(left, layer)
    return out


def spread(idxs, F, k):
    """k of `idxs` covering the set: the most central member, then greedily
    whichever is furthest from everything already chosen.

    Args:
        idxs: (n,) candidate indices into F.
        F: (N, m) objective values, normalized onto [0, 1] per column.

    Returns:
        (k,) of `idxs`.
    """
    if len(idxs) <= k:
        return idxs

    S      = F[idxs]
    chosen = [int(np.argmin(cdist(S, S.mean(axis=0, keepdims=True))))]
    while len(chosen) < k:
        chosen.append(int(np.argmax(cdist(S, S[chosen]).min(axis=1))))

    return idxs[chosen]


def score(values, truth, signs):
    """Fraction of the truth's own hypervolume a set of true objective vectors
    covers. 1.0 is the whole front; 0.0 is nothing past the reference point.

    Args:
        values: (n, m) *true* objective values, in the truth's own raw units.
        truth: the oracle, carrying `ref_point` and the scan's own hypervolume.
        signs: (m,) +1 on a maximized objective, -1 on a minimized one.
    """
    F = signs * np.asarray(values, dtype=float)

    return plr.hypervolume_from_nondominated(
        signs * truth.ref_point - F[plr.get_nondominated(F)]
    ) / truth.max_hypervolume(signs)


def dominated_by(F, reference):
    """How many rows of `F` some row of `reference` beats on every objective.
    Both are in maximization space.
    """
    return int(np.sum(np.any(np.all(reference[None] > F[:, None], axis=2), axis=1)))


def report(label, idxs, actions, values, names):
    print(f'\n=== {label} ===')
    print('  ' + ''.join(f'{f"x{j}":>16}' for j in range(actions.shape[1]))
          + ''.join(f'{n:>22}' for n in names))
    for i in idxs:
        print('  ' + ''.join(f'{a:>16.4f}' for a in actions[i])
              + ''.join(f'{v:>22.3f}' for v in values[i]))


def main():
    dataset = ExperimentDataset.load(DATASET)
    mogp    = dataset.get_model()          # last trial, rebuilt from its state dict
    names   = mogp.objectives.names

    X         = plr.sample_actions(mogp.action_bounds, SCAN, 'sobol', SEED)
    mu, _     = mogp.posterior_at(X)             # maximization space
    raw_mu, _ = mogp.posterior_at(X, raw=True)   # objectives' own units

    front = peel(mu, K, +1)
    anti  = peel(mu, K, -1)
    rest  = np.setdiff1d(np.arange(len(X)), np.union1d(front, anti))

    # a shared [0, 1] metric per objective, so both fronts are selected alike
    F   = (mu - mu.min(axis=0)) / np.ptp(mu, axis=0)
    rng = np.random.default_rng(SEED)

    sets = {
        'pareto front'      : spread(front, F, K),
        'anti-pareto front' : spread(anti, F, K),
        'neither front'     : rng.choice(rest, K, replace=False),
    }

    print(f'{dataset.name}, trial {len(dataset) - 1}, scan {len(X)}: {len(front)} on '
          f'the front, {len(anti)} on the anti-front, {len(rest)} on neither')
    for label, idxs in sets.items():
        report(label, idxs, X, raw_mu, names)

    if dataset.groundtruth is None:
        return

    truth  = dataset.get_groundtruth()
    signs  = mogp.objectives.signs
    scored = {label: truth(X[idxs], noise=False) for label, idxs in sets.items()}

    print('\n=== against the groundtruth ===')
    print(f'  {"set":<20}{"hypervolume":>14}{"dominated by front":>22}')
    for label, values in scored.items():
        print(f'  {label:<20}{score(values, truth, signs):>14.3f}'
              f'{dominated_by(signs * values, signs * scored["pareto front"]):>19} / {K}')

    fig, ax = plt.subplots(figsize=(5.5, 5), constrained_layout=True)
    plot_mo_space(
        ax,
        true_front = true_front(truth, dataset.get_objectives()),
        measured   = measured_at(mogp, truth),
        predicted  = inferred_front(mogp, truth),
        overlays   = scored,
        names      = [f'{name} ({"higher" if s > 0 else "lower"} is better)'
                      for name, s in zip(dataset.groundtruth.objectives, signs)],
        title      = f'{dataset.groundtruth.func}, {dataset.name}, '
                     f'trial {len(dataset) - 1}'
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=DPI)
    print(f'\nwrote {OUTPUT}')


if __name__ == '__main__':
    main()
