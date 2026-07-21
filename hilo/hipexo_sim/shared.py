"""Shared machinery for sweep experiments (action size, oracle noise, ...).

A "sweep" runs the same HILO experiment repeatedly while varying one config
knob, with several trials per value of that knob. Everything except the knob
itself is generic, so scripts supply three small callbacks:

  specialize(config, value, rng) -> config   how a sweep value mutates the config
  recover(config_dict) -> value              how to read the value back off a
                                             saved run, for --replot
  fmt_label(value) -> str                    legend text

Trials are *paired* across sweep values: trial k derives its RNG from
``SeedSequence(seed).spawn(n_trials)[k]``, so it re-draws the same groundtruth
and the same trial seed no matter which sweep value is being run. Differences
between sweep values therefore can't be an artifact of one value happening to
draw an easier groundtruth. (Where the sweep changes the *shape* of the draw —
as action size does — the draws necessarily differ; pairing is a no-op there.)
"""

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from hilo.hipexo_sim.experiment import run_experiment, save_run, plot_summary

# Discretization points per action dimension, shared by every sweep so the
# action-size and noise experiments stay comparable.
ACTION_DIMS = 15


def specialize_action_size(config, action_size, rng):
    """Resize the action space to `action_size` dims and redraw the groundtruth."""
    config.problem.action_low   = np.zeros(action_size)
    config.problem.action_high  = np.ones(action_size) * 5
    config.problem.action_dims  = np.ones(action_size, dtype=int) * ACTION_DIMS

    w1 = rng.uniform(low=config.problem.action_low, high=config.problem.action_high)
    w2 = rng.uniform(low=config.problem.action_low, high=config.problem.action_high)
    config.objective.w = np.array([w1, w2])

    config.validate()
    return config


def run_sweep(experiment_dir, base_config, sweep_values, specialize,
              trials_per_value, n_queries, seed, tol=0.0, extra_meta=None):
    """Run `trials_per_value` trials at each sweep value, one run_NNN/ folder each.

    Returns ``(experiment_dir, results)`` where results maps each sweep value to
    ``{'times': [...], 'hvs': [...], 'overlays': [...]}``, one row per trial.
    """
    experiment_dir = Path(experiment_dir)
    experiment_dir.mkdir(parents=True, exist_ok=True)

    (experiment_dir / 'experiment.json').write_text(json.dumps({
        'created': datetime.now().isoformat(timespec='seconds'),
        'sweep_values': [float(v) for v in sweep_values],
        'trials_per_value': trials_per_value,
        'n_queries': n_queries,
        'seed': seed,
        'tol': tol,
        **(extra_meta or {}),
    }, indent=2))

    # One RNG per trial index, identical across sweep values -> paired trials.
    trial_seeds = np.random.SeedSequence(seed).spawn(trials_per_value)

    hvs, overlays = [], []
    results = {}

    total = len(sweep_values) * trials_per_value
    pbar = tqdm(total=total, desc='Initializing')
    run_idx = 0

    for value in sweep_values:
        for trial in range(trials_per_value):
            pbar.set_description(f'{experiment_dir.name}: value={value}, trial={trial}')
            rng = np.random.default_rng(trial_seeds[trial])

            config = base_config.model_copy(deep=True)
            config = specialize(config, value, rng)

            trial_seed = round(rng.random() * 1000)
            run_data, metrics = run_experiment(
                seed        = trial_seed,
                config      = config,
                num_queries = n_queries,
                tol         = tol,
                pbar        = pbar,
            )
            save_run(experiment_dir / f'run_{run_idx:03d}', run_data, metrics,
                     trial_seed, config)
            run_idx += 1
            pbar.update(1)

            hvs.append(metrics['hv'])
            overlays.append(metrics['overlay'])

            bucket = results.setdefault(value, {'times': [], 'hvs': [], 'overlays': []})
            for key in bucket:
                bucket[key].append(run_data[key])

    pbar.close()
    plot_summary(experiment_dir, np.asarray(hvs), np.asarray(overlays))
    return experiment_dir, results


def load_sweep_results(experiment_dir, recover):
    """Regroup a saved experiment's runs by sweep value, for replotting.

    `recover` maps a run's stored config dict to its sweep value.
    """
    experiment_dir = Path(experiment_dir)
    results = {}
    for run_dir in sorted(experiment_dir.glob('run_*')):
        meta = json.loads((run_dir / 'run.json').read_text())
        value = recover(meta['config'])
        with np.load(run_dir / 'data.npz') as npz:
            bucket = results.setdefault(value, {'times': [], 'hvs': [], 'overlays': []})
            for key in bucket:
                bucket[key].append(npz[key])
    return results


def plot_sweep_comparison(experiment_dir, results, legend_title, fmt_label,
                          savename='sweep_comparison.svg'):
    """Per-iteration time / hypervolume / overlay, one line+colour per sweep value.

    Each line is the mean across that value's trials, with a +/- 1 std band. The
    query budget is identical across sweep values, so the x-axes align.
    """
    experiment_dir = Path(experiment_dir)
    values = sorted(results)
    colors = plt.cm.viridis(np.linspace(0, 0.85, max(len(values), 1)))

    fig, axs = plt.subplots(ncols=3, figsize=(18, 5))
    panels = (
        (axs[0], 'times',    'Query Time',              'Time (s)'),
        (axs[1], 'hvs',      'Groundtruth Hypervolume', 'Hypervolume'),
        (axs[2], 'overlays', 'Pareto Overlay',          'Fraction correct'),
    )

    for ax, key, title, ylabel in panels:
        for color, value in zip(colors, values):
            series = np.asarray(results[value][key])  # (n_trials, n_queries)
            iterations = np.arange(series.shape[1]) + 1
            mean = series.mean(axis=0)
            std = series.std(axis=0)
            ax.fill_between(iterations, mean - std, mean + std, color=color, alpha=0.15)
            ax.plot(iterations, mean, color=color, linewidth=2.5,
                    label=fmt_label(value))
        ax.set_title(title)
        ax.set_xlabel('Iteration')
        ax.set_ylabel(ylabel)
        ax.margins(x=0)

    axs[0].legend(title=legend_title)

    fig.tight_layout()
    savepath = experiment_dir / savename
    fig.savefig(savepath, dpi=150)
    plt.close(fig)
    print(f'Saved sweep comparison to {savepath.resolve()}')
    return savepath
