"""Walk a fitted run's Pareto front from the iPad slider.

The slider is one continuous choice between the two objectives, and the front
is the set of actions that trade them off. Both are one-dimensional, so the
slider becomes an *index* into the front: 0 at the objective-1 end, 100 at the
objective-2 end, and every position in between is a front point that is
actually attainable rather than a blend of two endpoint actions.

    conda activate pypolar
    python -m hilo.explore_front

Each Send prints the front point the slider sits at and hands its action to
`send_to_exo`.
"""

from pathlib import Path

import numpy as np
import socketio
import time
import pypolar as plr
from pypolar.experiment.dataset import ExperimentDataset
from tablet.preference import Preference
import logging
# from hilo.log import TO_BOTH, setup_logger
import hilo.shared.log as log
logger = logging.getLogger(__name__)

SUBJECT = 'MT01'
DATASET = Path('hilo/output/experiments/20260827_154607/MT01.json')
CONNECT = True
EXO_IP      = "192.168.1.122:5000" # move to hilo.hardware
SCAN = 4096     # Sobol points the front is read off
SEED = 95
EMULATE = False


sio = socketio.Client()
if EMULATE:
    import hilo.shared.simulation as hilo
else:
    import hilo.shared.hardware as hilo

if CONNECT:
    i = 0
    while True:
        try:
            i += 1
            sio.connect(f"ws://{EXO_IP}")
            break
        except socketio.exceptions.ConnectionError:
            # prints rather than logs: this runs at import, before setup_logger
            if i > 5:
                print('Exceeded maximum number of retries. Exiting...')
                exit()
            print('Connection failed. Retrying...')
            time.sleep(1)
    else:
        sio = None


def send_to_exo(action):
    logger.info(f'Exo got action {action}')
    ans = log.logged_input(f'Send {action} (y/n)? ')
    while True and ans.lower() != 'y':
        action = log.logged_input(f'Enter an alternative action as an array, like [1, 2, 3]: ')
        try:
            action = np.array(eval(action))
            if np.any(action < hilo.BOUNDS[0]) or np.any(action > hilo.BOUNDS[1]):
                raise ValueError('Action out of bounds')
            logger.info(f'Got {action}. Sending to exo...', extra=log.TO_BOTH)
            break
        except ValueError as e:
            logger.error(f'Action out of bounds. Note that bounds (low, high) = {hilo.BOUNDS}', extra=log.TO_BOTH)
            continue
        except Exception as e:
            logger.error(e, extra=log.TO_BOTH)
            logger.error(f'{action} did not compile. Try again.', extra=log.TO_BOTH)
            continue
        
    if CONNECT:
        action_dict = {
            'h_flex_torque_scale': action[0], # make this a dict when sending to the exo
            'h_ext_torque_scale' : action[1],
            'hip_delay_idx'      : action[2]
        }
        sio.emit("update_inputs", action_dict)
        logger.info('Successfully sent action.', extra=log.TO_BOTH)
        return True
    else:
        logger.info('Disabled!', extra=log.TO_BOTH)
        return True



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


def main():
    hilo.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    log.setup_logger(hilo.OUTPUT_DIR / f'{SUBJECT}.log')
    dataset = ExperimentDataset.load(DATASET)
    mogp    = dataset.get_model()          # last trial, rebuilt from its state dict
    names   = mogp.objectives.names

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
        pref.close()


if __name__ == '__main__':
    main()
