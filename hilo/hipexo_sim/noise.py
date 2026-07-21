import os
os.environ["JAX_PLATFORMS"] = "cpu"
import argparse
from datetime import datetime
from pathlib import Path

import numpy as np

from config.hipexo import hipexo_sim_idealized, MOHILO
from hilo.hipexo_sim.shared import (
    run_sweep,
    load_sweep_results,
    plot_sweep_comparison,
    specialize_action_size,
)

SAVENAME = 'noise_comparison.svg'
LEGEND_TITLE = 'Oracle noise'

# Action size is held fixed here; noise is the swept variable.
ACTION_SIZE = 2
DEFAULT_NOISE_LEVELS = [0.0, 0.05, 0.1, 0.25, 0.5, 1.0]


def specialize_noise(config: MOHILO, noise, rng):
    """Fix the action space at ACTION_SIZE dims, set the oracle's noise std.

    The action-space setup (and the groundtruth draw it makes) happens first and
    identically at every noise level, so trials stay paired across the sweep.
    """
    config = specialize_action_size(config, ACTION_SIZE, rng)
    config.oracle.noise_std = np.ones(config.num_objs) * noise
    # config.optimizer.signal_variances = np.array([20.0, 20.0])
    config.problem.precisions = np.ones(2) * 1.0
    config.validate()
    return config


def fmt_label(noise):
    return f'$\\sigma$ = {noise:g}'


def recover(config_dict):
    """Sweep value of a saved run: its oracle noise std (equal across objectives)."""
    return float(np.asarray(config_dict['oracle']['noise_std']).ravel()[0])


def parse_args():
    parser = argparse.ArgumentParser(
        description=f'Sweep HILO over oracle noise at action size {ACTION_SIZE}.'
    )
    parser.add_argument(
        '--out-dir', type=Path, default=Path('hilo/output/experiments'),
        help='Directory under which the experiment folder is created.'
    )
    parser.add_argument(
        '--name', type=str, default=None,
        help='Experiment folder name (default: timestamp).'
    )
    parser.add_argument('--trials-per-noise-level', type=int, default=10,
                        help='Number of trials per noise level (default: 10).')
    parser.add_argument('--n-queries', type=int, default=40,
                        help='Number of queries per trial (default: 40).')
    parser.add_argument('--seed', type=int, default=95, help='Base RNG seed.')
    parser.add_argument('--noise-levels', type=float, nargs='+',
                        default=DEFAULT_NOISE_LEVELS,
                        help='Oracle noise std values to sweep, applied equally to '
                             f'every objective (default: {DEFAULT_NOISE_LEVELS}).')
    parser.add_argument('--tol', type=float, default=0.05,
                        help='Non-domination tolerance for hv/overlay metrics, as a '
                             'fraction of each objective range (0 = strict).')
    parser.add_argument('--replot', type=Path, default=None,
                        help='Path to an existing experiment folder. Skips the '
                             'experiments and just regenerates the comparison figure.')
    return parser.parse_args()


def main():
    args = parse_args()

    if args.replot is not None:
        plot_sweep_comparison(args.replot, load_sweep_results(args.replot, recover),
                              LEGEND_TITLE, fmt_label, SAVENAME)
        return

    name = args.name or datetime.now().strftime('%Y%m%d_%H%M%S')
    experiment_dir = args.out_dir / name

    experiment_dir, results = run_sweep(
        experiment_dir   = experiment_dir,
        base_config      = hipexo_sim_idealized,
        sweep_values     = args.noise_levels,
        specialize       = specialize_noise,
        trials_per_value = args.trials_per_noise_level,
        n_queries        = args.n_queries,
        seed             = args.seed,
        tol              = args.tol,
        extra_meta       = {'sweep': 'noise_std', 'action_size': ACTION_SIZE},
    )
    plot_sweep_comparison(experiment_dir, results, LEGEND_TITLE, fmt_label, SAVENAME)
    print(f'\nExperiment complete: {experiment_dir.resolve()}')


if __name__ == '__main__':
    main()
