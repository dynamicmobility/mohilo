import os
os.environ["JAX_PLATFORMS"] = "cpu"
import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import pypolar as plr
from hilo.create import create_hipexo_sim
from config.hipexo import hipexo_sim_idealized
from config.base import MOHILO
from hilo.hipexo_sim.experiment import run_experiment, save_run, plot_summary

def specialize_config(config: MOHILO, action_size, rng):
    config.problem.action_low   = np.zeros(action_size)
    config.problem.action_high  = np.ones(action_size) * 5
    config.problem.action_dims  = np.ones(action_size, dtype=int) * 8

    w1 = rng.uniform(low=config.problem.action_low, high=config.problem.action_high)
    w2 = rng.uniform(low=config.problem.action_low, high=config.problem.action_high)
    config.objective.w = np.array([w1, w2])
    
    config.validate()

    return config

def run_experiments(experiment_dir, trials_per_action_dim, n_queries, seed, action_size_range, tol=0.0):
    """Run all trials, saving each into its own run_<idx> subfolder."""
    experiment_dir = Path(experiment_dir)
    experiment_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    hvs, overlays = [], []

    (experiment_dir / 'experiment.json').write_text(json.dumps({
        'created': datetime.now().isoformat(timespec='seconds'),
        'trials_per_action_dim': trials_per_action_dim,
        'n_queries': n_queries,
        'seed': seed,
        'action_size_range': action_size_range,
        'tol': tol,
    }, indent=2))

    N_RUNS = trials_per_action_dim * (action_size_range[1] - action_size_range[0])
    action_sizes = np.array(range(*action_size_range))
    action_sizes = np.repeat(action_sizes, repeats=trials_per_action_dim)
    pbar = tqdm(action_sizes, desc="Initializing")

    # action_size -> {'times': [...], 'hvs': [...], 'overlays': [...]}, one row
    # per trial, each row a per-iteration series.
    results = {}

    for trial, action_size in enumerate(pbar):
        pbar.set_description(f"Action size = {action_size}")
        config = hipexo_sim_idealized.model_copy(deep=True)
        config = specialize_config(config, action_size, rng)

        trial_seed = round(rng.random() * 1000)
        run_data, metrics = run_experiment(
            seed          = trial_seed,
            config        = config,
            num_queries   = n_queries,
            tol           = tol,
            pbar          = pbar,
        )
        save_run(experiment_dir / f'run_{trial:03d}', run_data, metrics, trial_seed, config)
        hvs.append(metrics['hv'])
        overlays.append(metrics['overlay'])

        bucket = results.setdefault(int(action_size), {'times': [], 'hvs': [], 'overlays': []})
        for key in bucket:
            bucket[key].append(run_data[key])

    plot_summary(experiment_dir, np.asarray(hvs), np.asarray(overlays))
    plot_action_size_comparison(experiment_dir, results)
    print(f'\nExperiment complete: {experiment_dir.resolve()}')
    return experiment_dir


def load_experiment_results(experiment_dir):
    """Regroup a saved experiment's runs by action size, for replotting.

    Returns ``{action_size: {'times': (n_trials, n_queries), 'hvs': ..., 'overlays': ...}}``.
    The action size is recovered from each run's stored config.
    """
    experiment_dir = Path(experiment_dir)
    results = {}
    for run_dir in sorted(experiment_dir.glob('run_*')):
        meta = json.loads((run_dir / 'run.json').read_text())
        action_size = len(meta['config']['problem']['action_low'])
        with np.load(run_dir / 'data.npz') as npz:
            bucket = results.setdefault(action_size, {'times': [], 'hvs': [], 'overlays': []})
            for key in bucket:
                bucket[key].append(npz[key])
    return results


def plot_action_size_comparison(experiment_dir, results,
                                savename='action_size_comparison.pdf'):
    """Per-iteration time / hypervolume / overlay, one line+colour per action size.

    Each line is the mean across that action size's trials, with a +/- 1 std band.
    The query budget is identical across action sizes, so the x-axes align.
    """
    experiment_dir = Path(experiment_dir)
    action_sizes = sorted(results)
    colors = plt.cm.viridis(np.linspace(0, 0.85, max(len(action_sizes), 1)))

    fig, axs = plt.subplots(ncols=3, figsize=(18, 5))
    panels = (
        (axs[0], 'times',    'Query Time',              'Time (s)'),
        (axs[1], 'hvs',      'Groundtruth Hypervolume', 'Hypervolume'),
        (axs[2], 'overlays', 'Pareto Overlay',          'Fraction correct'),
    )

    for ax, key, title, ylabel in panels:
        for color, action_size in zip(colors, action_sizes):
            series = np.asarray(results[action_size][key])  # (n_trials, n_queries)
            iterations = np.arange(series.shape[1]) + 1
            mean = series.mean(axis=0)
            std = series.std(axis=0)
            ax.fill_between(iterations, mean - std, mean + std, color=color, alpha=0.15)
            ax.plot(iterations, mean, color=color, linewidth=2.5,
                    label=f'{action_size}D')
        ax.set_title(title)
        ax.set_xlabel('Iteration')
        ax.set_ylabel(ylabel)
        ax.margins(x=0)

    axs[0].legend(title='Action size')

    fig.tight_layout()
    savepath = experiment_dir / savename
    fig.savefig(savepath, dpi=150)
    plt.close(fig)
    print(f'Saved action-size comparison to {savepath.resolve()}')
    return savepath

def parse_args():
    parser = argparse.ArgumentParser(
        description='Run the HILO experiment, saving per-iteration GP state per run.'
    )
    parser.add_argument(
        '--out-dir', type=Path, default=Path('hilo/output/experiments'),
        help='Directory under which the experiment folder is created.'
    )
    parser.add_argument(
        '--name', type=str, default=None,
        help='Experiment folder name (default: timestamp).'
    )
    parser.add_argument('--trials-per-action-size', type=int, default=10,
                        help='Number of trials/runs (default: 20).')
    parser.add_argument('--n-queries', type=int, default=20,
                        help='Number of queries per trial (default: 20).')
    parser.add_argument('--seed', type=int, default=95, help='Base RNG seed.')
    parser.add_argument('--action-size-range', type=int, nargs=2, default=(1, 4),
                        metavar=('START', 'STOP'),
                        help='Action sizes to sweep, as a half-open range '
                             '[START, STOP) (default: 1 3, i.e. 1D and 2D).')
    parser.add_argument('--tol', type=float, default=0.02,
                        help='Non-domination tolerance for hv/overlay metrics, as a '
                             'fraction of each objective range (0 = strict).')
    parser.add_argument('--replot', type=Path, default=None,
                        help='Path to an existing experiment folder. Skips the '
                             'experiments and just regenerates the comparison figure.')
    return parser.parse_args()


def main():
    args = parse_args()

    if args.replot is not None:
        plot_action_size_comparison(args.replot, load_experiment_results(args.replot))
        return

    name = args.name or datetime.now().strftime('%Y%m%d_%H%M%S')
    experiment_dir = args.out_dir / name
    run_experiments(
        experiment_dir          = experiment_dir,
        trials_per_action_dim   = args.trials_per_action_size,
        n_queries               = args.n_queries,
        seed                    = args.seed,
        action_size_range       = args.action_size_range,
        tol                     = args.tol,
    )


if __name__ == '__main__':
    main()