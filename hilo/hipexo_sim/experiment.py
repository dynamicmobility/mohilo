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


# ---------------------------------------------------------------------------    #
# Experiment folder layout                                                       #
#                                                                                #
#   <experiment_dir>/                                                            #
#     experiment.json          # experiment-level metadata (args, base cfg)      #
#     run_000/                                                                   #
#       run.json               # seed, metrics, full config (incl. groundtruth)  #
#       data.npz               # mu, std, feedback, action_space, true_objs      #
#     run_001/                                                                   #
#     ...                                                                        #
#                                                                                #
# Per-run data.npz arrays:                                                       #
#   mu           (n_queries, num_objs, N)  - GP mean per iteration/objective     #
#   std          (n_queries, num_objs, N)  - GP posterior std per iter/obj       #
#   feedback     (n_queries, num_objs + 1) - cumulative; row i = feedback added  #
#                                            at iteration i: [action_idx, *vals] #
#   action_space (N, d)                    - discretized action space            #
#   true_objs    (N, num_objs)             - groundtruth objectives on the grid  #
# ---------------------------------------------------------------------------    #


def run_experiment(seed, config, num_queries, tol=0.0):
    """Run a single trial, recording the GP mean/std and feedback each iteration.

    ``tol`` is the non-domination tolerance (fraction of each objective's range)
    used for the final-iteration hv/overlay metrics.

    Returns a dict of numpy arrays ready to hand to ``save_run``.
    """
    rng = np.random.default_rng(seed)

    regression, optimizer, sampler, groundtruth, oracle = create_hipexo_sim(
        rng, config
    )

    num_objs = len(optimizer.gps)

    mus = []
    stds = []

    for _ in tqdm(range(num_queries), disable=True):
        # Sample an action and measure the (noisy) human performance
        sample_action = sampler.sample(regression.action_space)
        values = oracle.query(sample_action)

        regression.add_feedback(sample_action, values)
        optimizer.setup(
            action_space=regression.action_space,
            likelihoods=regression.get_likelihood_functions(),
        )
        optimizer.fit(method='trust-constr', options={'disp': False})
        sampler.update_posterior()

        # (1) GP mean and (2) GP std per objective, this iteration
        mus.append(np.array([optimizer.gps[i].mu for i in range(num_objs)]))
        stds.append(np.array([optimizer.gps[i].std() for i in range(num_objs)]))

    true_objs = np.asarray(groundtruth(regression.action_space))  # (N, num_objs)
    # Final-iteration performance against the groundtruth Pareto front
    estimated_objs = mus[-1].T  # (N, num_objs)
    metrics = {
        'hv': float(plr.groundtruth_hypervolume(estimated_objs, true_objs, tol=tol)),
        'overlay': float(plr.pareto_overlay(estimated_objs, true_objs, tol=tol)),
        'tol': tol,
    }

    run_data = {
        'mu': np.asarray(mus),                         # (n_queries, num_objs, N)
        'std': np.asarray(stds),                       # (n_queries, num_objs, N)
        'feedback': np.asarray(regression.feedback_data),  # (n_queries, num_objs+1)
        'action_space': np.asarray(regression.action_space),   # (N, d)
        'true_objs': true_objs,
    }
    return run_data, metrics


def save_run(run_dir, run_data, metrics, seed, config):
    """Persist a single run: arrays to data.npz, metadata to run.json."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    np.savez(run_dir / 'data.npz', **run_data)

    # Full config includes config.objective, i.e. the groundtruth objective
    # parameters (w, delta, gamma, bounds) — no need to store them separately.
    meta = {
        'seed': int(seed),
        'metrics': metrics,
        'config': config.to_jsonable_dict(),
    }
    (run_dir / 'run.json').write_text(json.dumps(meta, indent=2))


def run_experiments(experiment_dir, n_trials, n_queries, seed, action_size, tol=0.0):
    """Run all trials, saving each into its own run_<idx> subfolder."""
    experiment_dir = Path(experiment_dir)
    experiment_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    hvs, overlays = [], []

    (experiment_dir / 'experiment.json').write_text(json.dumps({
        'created': datetime.now().isoformat(timespec='seconds'),
        'n_trials': n_trials,
        'n_queries': n_queries,
        'seed': seed,
        'action_size': action_size,
        'tol': tol,
    }, indent=2))

    pbar = tqdm(range(n_trials), desc="Initializing")

    for trial in pbar:
        pbar.set_description(f"Running experiment {experiment_dir.name}")
        config = hipexo_sim_idealized.model_copy(deep=True)
        config.problem.action_high = np.ones(action_size) * 5.0

        # Randomize the groundtruth ideal point per trial
        w1 = rng.uniform(low=config.problem.action_low, high=config.problem.action_high)
        w2 = rng.uniform(low=config.problem.action_low, high=config.problem.action_high)
        config.objective.w = np.array([w1, w2])

        trial_seed = round(rng.random() * 1000)
        run_data, metrics = run_experiment(
            seed=trial_seed,
            config=config,
            num_queries=n_queries,
            tol=tol,
        )
        save_run(experiment_dir / f'run_{trial:03d}', run_data, metrics, trial_seed, config)
        hvs.append(metrics['hv'])
        overlays.append(metrics['overlay'])

    plot_summary(experiment_dir, np.asarray(hvs), np.asarray(overlays))
    print(f'\nExperiment complete: {experiment_dir.resolve()}')
    return experiment_dir


def plot_summary(experiment_dir, hvs, overlays, savename='summary.png'):
    """Bar plot of per-trial final performance.

    Left subplot ranks trials by hypervolume; right ranks by pareto overlay.
    Each bar is labelled with its trial index so the ordering stays legible.
    """
    experiment_dir = Path(experiment_dir)
    trials = np.arange(len(hvs))

    fig, (hv_ax, ov_ax) = plt.subplots(ncols=2, figsize=(14, 5))

    for ax, series, title, ylabel, color in (
        (hv_ax, hvs, 'Ranked by hypervolume', 'Groundtruth hypervolume', 'tab:blue'),
        (ov_ax, overlays, 'Ranked by pareto overlay', 'Pareto overlay', 'tab:green'),
    ):
        order = np.argsort(series)[::-1]  # best (highest) first
        positions = np.arange(len(series))
        ax.bar(positions, series[order], color=color, alpha=0.85)
        ax.set_xticks(positions)
        ax.set_xticklabels([f'{t:d}' for t in trials[order]], rotation=90, fontsize=7)
        ax.set_xlabel('Trial')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.margins(x=0.01)

    fig.tight_layout()
    savepath = experiment_dir / savename
    fig.savefig(savepath, dpi=150)
    plt.close(fig)
    print(f'Saved summary figure to {savepath.resolve()}')
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
    parser.add_argument('--n-trials', type=int, default=20,
                        help='Number of trials/runs (default: 20).')
    parser.add_argument('--n-queries', type=int, default=20,
                        help='Number of queries per trial (default: 20).')
    parser.add_argument('--seed', type=int, default=95, help='Base RNG seed.')
    parser.add_argument('--action-size', type=int, default=1)
    parser.add_argument('--tol', type=float, default=0.02,
                        help='Non-domination tolerance for hv/overlay metrics, as a '
                             'fraction of each objective range (0 = strict).')
    return parser.parse_args()


def main():
    args = parse_args()
    name = args.name or datetime.now().strftime('%Y%m%d_%H%M%S')
    experiment_dir = args.out_dir / name
    run_experiments(
        experiment_dir=experiment_dir,
        n_trials=args.n_trials,
        n_queries=args.n_queries,
        seed=args.seed,
        action_size=args.action_size,
        tol=args.tol,
    )


if __name__ == '__main__':
    main()
