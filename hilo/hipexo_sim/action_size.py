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
    config.problem.action_dims  = np.ones(action_size, dtype=int) * 100

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
    pbar = tqdm(enumerate(action_sizes), desc="Initializing")

    for trial, action_size in pbar:
        pbar.set_description(f"Action size = {action_size}")
        config = hipexo_sim_idealized.model_copy(deep=True)
        config = specialize_config(config, action_size, rng)

        trial_seed = round(rng.random() * 1000)
        run_data, metrics = run_experiment(
            seed          = trial_seed,
            config        = config,
            num_queries   = n_queries,
            tol           = tol,
        )
        save_run(experiment_dir / f'run_{trial:03d}', run_data, metrics, trial_seed, config)
        hvs.append(metrics['hv'])
        overlays.append(metrics['overlay'])

    # plot_summary(experiment_dir, np.asarray(hvs), np.asarray(overlays))
    print(f'\nExperiment complete: {experiment_dir.resolve()}')
    return experiment_dir

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
    parser.add_argument('--trials-per-action-size', type=int, default=20,
                        help='Number of trials/runs (default: 20).')
    parser.add_argument('--n-queries', type=int, default=20,
                        help='Number of queries per trial (default: 20).')
    parser.add_argument('--seed', type=int, default=95, help='Base RNG seed.')
    parser.add_argument('--action-size-range', type=tuple, default=(1,5))
    parser.add_argument('--tol', type=float, default=0.02,
                        help='Non-domination tolerance for hv/overlay metrics, as a '
                             'fraction of each objective range (0 = strict).')
    return parser.parse_args()


def main():
    args = parse_args()
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