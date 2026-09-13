"""Space of hip exo torque profiles reachable by the delay/scale parameterization, for KOS's 1.25 m/s walk.

Each profile scales the biological hip flexion moment by one gain where it is positive and another
where it is negative, then delays it. Every combination on a 5-point grid over the parameter bounds
is drawn, with its envelope shaded and the default parameters' profile (low-pass filtered) on top.
"""

from pathlib import Path
import pypolar as plr
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from scipy.signal import butter, filtfilt

DATA_PATH = Path(__file__).parent / 'KOS_walk_1_2_mean_cycle.mat'

SAMPLING_FREQ   = 455       # Hz, converts delay to samples
STEPS_PER_PARAM = 5
LOWER_BOUNDS    = [0.0, 0.0, 0.0]       # delay (s), positive gain, negative gain
UPPER_BOUNDS    = [0.375, 0.3, 0.3]
DEFAULT_PARAMS  = [0.1, 0.2, 0.2]
FILTER_FREQ     = 10        # Hz
FILTER_ORDER    = 2
ALPHA           = 0.75


def directional_torque_delay(moment, fs, delay, gain_pos, gain_neg):
    """`moment` scaled by `gain_pos` where >= 0 and `gain_neg` where < 0, circularly delayed by `delay` s."""
    torque = np.where(moment >= 0, gain_pos * moment, gain_neg * moment)

    return np.roll(torque, int(np.floor(delay * fs)))


def parameter_grid(lb, ub, steps):
    """(steps^N, N) grid over the box, first parameter varying fastest (MATLAB ndgrid order)."""
    axes = [np.linspace(lo, hi, steps) for lo, hi in zip(lb, ub)]

    return np.stack([g.ravel(order='F') for g in np.meshgrid(*axes, indexing='ij')], axis=1)


def main():
    moment = loadmat(DATA_PATH, squeeze_me=True, struct_as_record=False)['mean_cycle'].hip_flexion_moment
    xq     = np.linspace(0, 1, len(moment))

    params = parameter_grid(LOWER_BOUNDS, UPPER_BOUNDS, STEPS_PER_PARAM)
    curves = np.stack([directional_torque_delay(moment, SAMPLING_FREQ, *p) for p in params])

    b, a    = butter(FILTER_ORDER, FILTER_FREQ, btype='low', fs=SAMPLING_FREQ)
    default = filtfilt(b, a, directional_torque_delay(moment, SAMPLING_FREQ, *DEFAULT_PARAMS))

    fig, ax = plt.subplots()
    colors  = plt.get_cmap('cool')(np.linspace(0, 1, len(curves)))
    for curve, color in zip(curves, colors):
        ax.plot(xq, curve, color=color, alpha=ALPHA, linewidth=1)

    ax.plot(xq, moment, color='tab:blue', linewidth=2, label='Bio Joint Moment (Nm/kg)')
    ax.plot(xq, default, color='k', linewidth=3, label='Default Parameters')
    ax.fill_between(xq, curves.min(axis=0), curves.max(axis=0), color='darkviolet', alpha=0.2,
                    edgecolor='none', label='Combination Space')
    ax.set_xlabel('Cycle (%)')
    ax.set_title('Parameterization Space for 1.25 m/s walk')
    ax.legend(loc='best')
    fig.tight_layout()
    plr.dress_axis(ax)
    fig.set_size_inches((6,3))
    path = 'human_data/KOS/ctrl.svg'
    plt.savefig(path, transparent=True)
    print('saved to', path)


if __name__ == '__main__':
    main()
