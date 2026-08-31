"""Walk a fitted run's Pareto front from the iPad slider.

The slider is one continuous choice between the two objectives, and the front
is the set of actions that trade them off. Both are one-dimensional, so the
slider becomes an *index* into the front: 0 at the objective-1 end, 100 at the
objective-2 end, and every position in between is a front point that is
actually attainable rather than a blend of two endpoint actions.

    conda activate pypolar
    python -m hilo.explore_front --dataset hilo/output/experiments/<run>/MT01.json

Each Send prints the front point the slider sits at and hands its action to
`send_to_exo`.
"""

import argparse
from pathlib import Path

import numpy as np
import pypolar as plr
from pypolar.experiment.dataset import ExperimentDataset
from tablet.preference import Preference
import logging
import hilo.shared.log as log
from hilo.fit_mogp import connect_to_exo, find_dataset, send_to_exo, sio
import hilo.fit_mogp as fit_mogp

logger = logging.getLogger(__name__)
SCAN = 4096     # Sobol points the front is read off
SEED = 95


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--dataset', required=True, type=Path,
                        help='the fitted run whose front the slider walks')
    parser.add_argument('--connect', action=argparse.BooleanOptionalAction, default=True,
                        help='send actions to the exo over the socket')
    parser.add_argument('--emulate', action=argparse.BooleanOptionalAction, default=False,
                        help='take the constants from the simulation rather than the hardware')

    return parser.parse_args(argv)


def build_front(mogp, scan=SCAN, seed=SEED):
    """The model's inferred Pareto front, ordered objective 1 -> objective 2.

    Args:
        mogp: a fitted `DecoupledMOGP` over exactly two objectives.

    Returns:
        `(actions, values)`, `(k, d)` actions in raw units and `(k, 2)`
        posterior means in the objectives' own units.
    """
    if len(mogp.objectives) != 2:
        raise ValueError(f'the slider indexes a two-objective front, got '
                         f'{len(mogp.objectives)}')

    X         = plr.sample_actions(mogp.action_bounds, scan, 'sobol', seed)
    mu, _     = mogp.posterior_at(X)              # maximization space
    raw_mu, _ = mogp.posterior_at(X, raw=True)    # objectives' own units

    front = plr.get_nondominated(mu)
    front = front[np.argsort(-mu[front, 0])]

    return X[front], raw_mu[front]


def index_for(slider, k):
    """The front index slider position `slider`, 0 - 100, lands on.

    The 101 slider positions are spread evenly over the k front points, so both
    ends are reachable and every point is reachable when k <= 101.
    """
    return int(round(float(slider) / 100.0 * (k - 1)))


def header(dim, names):
    """The column header a `row` line sits under."""
    return ('  ' + f'{"idx":>5}'
            + ''.join(f'{f"x{j}":>12}' for j in range(dim))
            + ''.join(f'{n:>22}' for n in names))


def row(i, actions, values, mark=' '):
    """One front point as a line, `mark`ed in the first column."""
    return (mark + ' ' + f'{i:>5}'
            + ''.join(f'{a:>12.4f}' for a in actions[i])
            + ''.join(f'{v:>22.3f}' for v in values[i]))


def report(actions, values, names, idx=None):
    """The whole front as a table, with `idx` marked when given."""
    print(header(actions.shape[1], names))
    for i in range(len(actions)):
        print(row(i, actions, values, '>' if i == idx else ' '))


def main(argv=None):
    args = parse_args(argv)
    # nothing here reads the backend's constants; `send_to_exo` checks its
    # BOUNDS, so what --emulate picks is which module that one sees
    fit_mogp.load_backend(args.emulate)

    path    = find_dataset(args.dataset)
    dataset = ExperimentDataset.load(path)
    # beside the run being explored, so a session is recorded where its dataset
    # is rather than in a new directory of its own
    log.setup_logger(path.parent / 'explore.log')

    if args.connect:
        connect_to_exo()

    mogp  = dataset.get_model()          # last trial, rebuilt from its state dict
    names = mogp.objectives.names

    actions, values = build_front(mogp)
    print(f'{dataset.name}, trial {len(dataset) - 1}, scan {SCAN}: '
          f'{len(actions)} points on the front')
    report(actions, values, names)

    pref = Preference(labels=names)
    print('Waiting for the iPad...')
    pref.wait_for_ipad()

    try:
        while True:
            slider = pref.wait_for_send()
            if slider is None:
                continue

            idx = index_for(slider, len(actions))
            print(f'\nslider {slider} -> front index {idx} of {len(actions) - 1}')
            print(header(actions.shape[1], names))
            print(row(idx, actions, values, '>'))
            send_to_exo(actions[idx])
    except KeyboardInterrupt:
        pass
    finally:
        pref.close()
        sio.disconnect() # a no-op on a client that never connected


if __name__ == '__main__':
    main()
