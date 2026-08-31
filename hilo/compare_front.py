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

I = 10
DATASET = Path('hilo/output/experiments/Aug28_Neil/MT01.json')
OUTPUT  = Path(f'hilo/output/front_samples{I}.png')

K       = 3        # actions per set
SCAN    = 2**14     # Sobol points the fronts are read off
SEED    = 95
DPI     = 500
FRONT_CMAP = 'RdYlGn'   # front points colored by their index along it


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

def fit_gp(
    objective   : plr.DecoupledObjectives,
    noise       : plr.NoiseModel = None,
    hypers      : plr.GPHyperparameters = None
):
    return plr.DecoupledMOGP(
        objectives          = objective,
        noise               = plr.NoiseModel.prior(0.5),
        fit_hyperparameters = True,
        min_length_scale    = 0.1,
    )
    
def main():
    dataset = ExperimentDataset.load(DATASET)
    mogp    = dataset.get_model()          # last trial, rebuilt from its state dict
    names   = mogp.objectives.names
    mogp_old = dataset.get_model(trial=I)
    new_obj: plr.DecoupledObjectives = mogp.objectives + mogp_old.objectives
    
    mogp = fit_gp(new_obj)

    # BOUNDS = [[0.15, 0.2, 0], [0.25, 0.25, 30]]
    X         = plr.sample_actions(mogp.action_bounds, SCAN, 'sobol', SEED)
    # X         = plr.sample_actions(BOUNDS, SCAN, 'sobol', SEED)
    mu, _     = mogp.posterior_at(X)             # maximization space
    raw_mu, _ = mogp.posterior_at(X, raw=True)   # objectives' own units

    front = peel(mu, K, +1)
    anti  = peel(mu, K, -1)
    rest  = np.setdiff1d(np.arange(len(X)), np.union1d(front, anti))

    # a shared [0, 1] metric per objective, so both fronts are selected alike
    F   = (mu - mu.min(axis=0)) / np.ptp(mu, axis=0)
    rng = np.random.default_rng(SEED)

    pareto     = spread(front, F, K)
    antipareto = spread(anti, F, K)
    neither    = rng.choice(rest, K, replace=False)
    
    distances = cdist(new_obj.xtransform(X)[pareto], new_obj.actions()[0])
    closest_trial = np.argmin(distances, axis=1)
    closest_distances = np.min(distances, axis=1) / np.sqrt(3)
    print(closest_trial)
    print(closest_distances)
    quit()
    
    sets = {
        'pareto front'      : pareto,
        'anti-pareto front' : antipareto,
        'neither front'     : neither
    }

    print(f'{dataset.name}, trial {len(dataset) - 1}, scan {len(X)}: {len(front)} on '
          f'the front, {len(anti)} on the anti-front, {len(rest)} on neither')
    for label, idxs in sets.items():
        report(label, idxs, X, raw_mu, names)
        
        

    if dataset.groundtruth is None:
        # objective space flat, action space in 3D, so the axes are made one at
        # a time: subplot_kw goes to every subplot alike
        X_tr = mogp.objectives.xtransform(X)
        fig  = plt.figure(figsize=(12, 5), constrained_layout=True)
        p_ax = fig.add_subplot(1, 2, 1)
        a_ax = fig.add_subplot(1, 2, 2, projection='3d')
        nd_idx = plr.get_nondominated_tol(mu)
        nd_idx = nd_idx[np.argsort(raw_mu[nd_idx, 0])]  # one ordering, so the front draws as a line
        nd_mu = raw_mu[nd_idx]
        order = np.arange(len(nd_idx))  # position along the front, the colormap's argument
        p_ax.scatter(raw_mu[:, 0], raw_mu[:, 1], s=3, c=X_tr, alpha=0.1, zorder=0,
                     label='posterior scan')
        p_ax.plot(nd_mu[:, 0], nd_mu[:, 1], lw=2, c='black', zorder=1,
                  label='pareto front')
        # ramp = p_ax.scatter(
        #     nd_mu[:, 0], 
        #     nd_mu[:, 1], 
        #     s=25,
        #     c=order, 
        #     cmap=FRONT_CMAP,
        #     edgecolors='black', 
        #     linewidths=1, 
        #     zorder=2
        # )
        
        ramp = p_ax.scatter(
            nd_mu[:, 0], 
            nd_mu[:, 1], 
            s=25,
            c=X_tr[nd_idx],
            edgecolors='black', 
            linewidths=1, 
            zorder=2
        )
        
        p_ax.set_xlabel(mogp.objectives[0].name)
        p_ax.set_ylabel(mogp.objectives[1].name)
        pts = X[nd_idx]
        actions0 = mogp.objectives[0].xdata
        actions1 = mogp.objectives[1].xdata
        
        idxs = np.array([np.argmin(np.linalg.norm(actions0 - a, axis=1)) for a in actions1])
        
        p_ax.scatter(mogp.objectives[0].ydata[idxs], mogp.objectives[1].ydata, s=15, marker='x', c='C2',label='measured')
        plr.dress_axis(p_ax)
        
        a_ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], lw=2, c='black', zorder=1)
        a_ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=25, c=order, cmap=FRONT_CMAP,
                     edgecolors='black', linewidths=1, zorder=2, depthshade=False)
        
        a_ax.set_xlabel('h_flex_torque_scale')
        a_ax.set_ylabel('h_ext_torque_scale')
        a_ax.set_zlabel('hip_delay_idx')
        
        bounds = np.array(new_obj.action_bounds)
        a_ax.set_xlim(bounds[:, 0])
        a_ax.set_ylim(bounds[:, 1])
        a_ax.set_zlim(bounds[:, 2])

        # the scan's markers carry alpha=0.1, which is invisible at legend size
        legend = p_ax.legend(loc='upper right', framealpha=0.9)
        for handle in legend.legend_handles:
            handle.set_alpha(1.0)
        fig.colorbar(ramp, ax=[p_ax, a_ax], label='index along the front', shrink=0.7)

        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(OUTPUT, dpi=DPI)
        print(f'\nwrote {OUTPUT}')
    else:
        truth  = dataset.get_groundtruth()
        signs  = mogp.objectives.signs
        scored = {label: truth(X[idxs], noise=False) for label, idxs in sets.items()}
        
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

        print('\n=== against the groundtruth ===')
        print(f'  {"set":<20}{"hypervolume":>14}{"dominated by front":>22}')
        for label, values in scored.items():
            print(f'  {label:<20}{score(values, truth, signs):>14.3f}'
                f'{dominated_by(signs * values, signs * scored["pareto front"]):>19} / {K}')


if __name__ == '__main__':
    main()
