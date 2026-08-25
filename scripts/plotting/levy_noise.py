"""What `rel_noise_std` means in a synthetic function's own units, on 1D Levy.

`SyntheticFunction` states its observation noise as a fraction of the
function's spread, measured once over a Sobol scan of the whole box. `spread`
is the scan's standard deviation under `measure='std'` and its peak-to-peak
range under `measure='range'`, so the same `rel_noise_std` is a different
absolute noise under each. This draws both against the function's own range.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import pypolar as plr
from pypolar.utils.plotting import MODEL_COLOR, SAMPLE_COLOR, TRUTH_COLOR

FUNC      = 'Levy'
DIM       = 1
BOX       = 5.0        # half-width of the action box
N_SCAN    = 2 ** 12    # points the truth curve is drawn over
N_SAMPLES = 100        # noisy measurements drawn per panel
REL       = (0.05, 0.1, 0.25, 0.5)
MEASURES  = ('std', 'range')
BAND_STD  = 2.0        # half-width of the shaded noise band, in noise sigmas
SEED      = 95
OUTPUT    = Path('scripts/output/levy_noise.svg')


def draw(truth, X):
    """A `SyntheticFunction` and its `N_SAMPLES` noisy values at X, per
    (measure, rel_noise_std) pair.

    Every function is seeded identically, so all of them scale one shared
    stream of standard normals: a panel differs from its neighbour only by
    `noise_std`, not by which draws it got.
    """
    panels = {}
    for measure in MEASURES:
        for rel in REL:
            sf = plr.SyntheticOracle(
                truth, 
                rel_noise_std   = rel,
                measure         = measure,
                seed            = SEED
            )
            panels[measure, rel] = (sf, sf(X))

    return panels


def plot_range(ax, X, y):
    """The truth curve, with the two spreads `rel_noise_std` can be a fraction
    of marked on it."""
    plr.plot_test_function(ax, X, y, f'{FUNC} at dim {DIM}')
    ax.axhline(y.min(), color=TRUTH_COLOR, ls=':', lw=1.0)
    ax.axhline(y.max(), color=TRUTH_COLOR, ls=':', lw=1.0,
               label=f"range (measure='range') = {np.ptp(y):.3f}")
    ax.axhspan(y.mean() - y.std(), y.mean() + y.std(),
               color=MODEL_COLOR, alpha=0.15,
               label=f"mean $\\pm$ std (measure='std'), std = {y.std():.3f}")
    ax.set_xlabel('action')
    ax.set_ylabel(f'{FUNC}(x)')
    ax.legend(fontsize=7, loc='upper right')


def plot_noise(ax, X, y, Xs, ys, sf, measure, rel):
    """One panel: the truth, a `BAND_STD`-sigma noise band around it, and the
    noisy measurements actually drawn."""
    order = np.argsort(X.ravel())
    ax.fill_between(X.ravel()[order],
                    y[order] - BAND_STD * sf.noise_std,
                    y[order] + BAND_STD * sf.noise_std,
                    color=MODEL_COLOR, alpha=0.18,
                    label=f'truth $\\pm$ {BAND_STD:g}$\\sigma$')
    ax.plot(X.ravel()[order], y[order], color=TRUTH_COLOR, lw=1.2, label='truth')
    ax.scatter(Xs.ravel(), ys, s=8, color=SAMPLE_COLOR, zorder=3,
               label=f'{len(ys)} measurements')

    ax.set_title(f"measure='{measure}', rel_noise_std={rel:g}\n"
                 f'$\\sigma$ = {sf.noise_std:.3f} {FUNC} units '
                 f'($\\pm${BAND_STD * sf.noise_std:.3f} at {BAND_STD:g}$\\sigma$)',
                 fontsize=8)
    ax.tick_params(labelsize=6)
    ax.set_xlabel('action', fontsize=8)


def report(y, Xs, ys_true, panels):
    """The same translation as a table, in the function's own units."""
    lo, hi = y.min(), y.max()
    print(f'{FUNC} at dim {DIM} on [{-BOX:g}, {BOX:g}]: '
          f'min {lo:.4f}, max {hi:.4f}, range {np.ptp(y):.4f}, std {y.std():.4f}')

    for measure in MEASURES:
        spread = panels[measure, REL[0]][0].measure_spread
        print(f"\nmeasure='{measure}': spread = {spread:.4f} {FUNC} units")
        print(f'{"rel":>6}{"sigma":>10}{"+/- 2 sigma":>14}{"% of range":>12}'
              f'{"drawn sigma":>14}')
        for rel in REL:
            sf, ys = panels[measure, rel]
            print(f'{rel:>6.2f}{sf.noise_std:>10.4f}'
                  f'{2 * sf.noise_std:>14.4f}'
                  f'{100 * 2 * sf.noise_std / np.ptp(y):>12.1f}'
                  f'{(ys - ys_true).std(ddof=1):>14.4f}')


def main():
    truth = plr.construct_function(plr.SYNTHETIC_FUNCTIONS[FUNC], DIM, BOX, seed=SEED)
    X     = plr.sample_actions(bounds=truth.bounds, n=N_SCAN, kind='sobol', seed=SEED)
    y     = plr.truth_at(truth, X)

    # a separate design from the curve's, so the measurements are not a subset
    Xs      = plr.sample_actions(bounds=truth.bounds, n=N_SAMPLES, kind='sobol',
                                 seed=SEED + 1)
    ys_true = plr.truth_at(truth, Xs)
    panels  = draw(truth, Xs)

    fig = plt.figure(figsize=(3.4 * len(REL), 2.8 * (len(MEASURES) + 1)))
    gs  = fig.add_gridspec(len(MEASURES) + 1, len(REL))
    plot_range(fig.add_subplot(gs[0, :]), X, y)

    shared = None
    for i, measure in enumerate(MEASURES):
        for j, rel in enumerate(REL):
            ax = fig.add_subplot(gs[i + 1, j], sharey=shared)
            shared = ax if shared is None else shared   # one y scale for every panel
            sf, ys = panels[measure, rel]
            plot_noise(ax, X, y, Xs, ys, sf, measure, rel)
            if j == 0:
                ax.set_ylabel(f'{FUNC}(x)', fontsize=8)
                ax.legend(fontsize=6, loc='upper right')

    fig.tight_layout()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT)
    report(y, Xs, ys_true, panels)
    print(f'\nwrote {OUTPUT}')


if __name__ == '__main__':
    main()
