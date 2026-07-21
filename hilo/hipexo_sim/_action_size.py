import os
os.environ["JAX_PLATFORMS"] = "cpu"
import argparse
import json
import pypolar as plr
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from hilo.create import create_hipexo_sim
from config.hipexo import hipexo_sim_idealized
import time

def run_experiment(seed, config, num_queries):
    rng = np.random.default_rng(seed)

    regression, optimizer, sampler, groundtruth, oracle = create_hipexo_sim(
        rng, config
    )

    times = []
    hvs = []
    overlays = []

    # Run simulation
    for _ in tqdm(range(num_queries), disable=True):
        start = time.time()
        # Sample an action
        sample_action = sampler.sample(regression.action_space)

        # Measure the human performance
        mct_hat, sp_hat = oracle.query(sample_action)

        regression.add_feedback(sample_action, [mct_hat, sp_hat])
        optimizer.setup(
            action_space = regression.action_space,
            likelihoods  = regression.get_likelihood_functions()
        )
        optimizer.fit(method='trust-constr', options={'disp': False})
        sampler.update_posterior()
        end = time.time()

        estimated_objs    = np.array([optimizer.gps[i].mu for i in range(len(optimizer.gps))]).T
        true_objs         = groundtruth(regression.action_space)
        hvs.append(plr.groundtruth_hypervolume(
            estimated_objs    = estimated_objs,
            true_objs         = true_objs
        ))
        overlays.append(plr.pareto_overlay(
            estimated_objs    = estimated_objs,
            true_objs         = true_objs
        ))
        times.append(end - start)

    return times, hvs, overlays

def run_experiments(n_trials, n_queries, seed, action_size):
    """Run all trials and return stacked arrays of shape (n_trials, n_queries)."""
    data = {'times': [], 'hvs': [], 'overlays': [], 'ws': []}
    rng = np.random.default_rng(seed)
    configs = []
    seeds = []

    for _ in tqdm(range(n_trials)):
        config = hipexo_sim_idealized.model_copy(deep=True)
        # config.problem.action_low = np.zeros(action_size)
        config.problem.action_high = np.ones(action_size) * 5.0
        # config.problem.action_dims = np.array([100] * action_size)
        w1 = rng.uniform(
            low=config.problem.action_low,
            high=config.problem.action_high,
        )
        w2 = rng.uniform(
            low=config.problem.action_low,
            high=config.problem.action_high,
        )
        config.objective.w = np.array([w1, w2])
        trial_seed = round(rng.random() * 1000)
        times, hvs, overlays = run_experiment(
            seed          = trial_seed,
            config        = config,
            num_queries   = n_queries,
        )
        data['times'].append(times)
        data['hvs'].append(hvs)
        data['overlays'].append(overlays)
        data['ws'].append(config.objective.w)
        configs.append(config.to_jsonable_dict())
        seeds.append(trial_seed)

    out = {k: np.asarray(v) for k, v in data.items()}
    out['seeds'] = np.asarray(seeds)
    out['configs'] = np.array(json.dumps(configs))
    return out

def save_data(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **data)
    print(f'Saved data to {path.resolve()}')

def load_data(path):
    path = Path(path)
    with np.load(path) as npz:
        data = {k: npz[k] for k in npz.files}
    print(f'Loaded data from {path.resolve()}')
    return data

def plot_panel(ax, series, title, ylabel, color):
    """Plot each trial faintly plus a thick mean line with std band."""
    series = np.asarray(series)
    iterations = np.arange(series.shape[1]) + 1

    # Faint individual trials
    for row in series:
        ax.plot(iterations, row, color=color, alpha=0.15, linewidth=1)

    # Thick averaged line
    mean = series.mean(axis=0)
    std = series.std(axis=0)
    ax.fill_between(iterations, mean - std, mean + std, color=color, alpha=0.15)
    ax.plot(iterations, mean, color=color, linewidth=2.5, label='mean')

    ax.set_title(title)
    ax.set_xlabel('Iteration')
    ax.set_ylabel(ylabel)
    ax.margins(x=0)

def summarize(data):
    """Print best/worst trials by final-iteration hypervolume, with their w."""
    if 'ws' not in data:
        return
    hvs = np.asarray(data['hvs'])
    ws = np.asarray(data['ws'])
    seeds = np.asarray(data['seeds']) if 'seeds' in data else None
    final_hv = hvs[:, -1]
    order = np.argsort(final_hv)  # ascending: worst first, best last

    def fmt(w):
        w = np.asarray(w).ravel()
        return ', '.join(f'{v:.3f}' for v in w)

    def line(label, idx):
        seed_str = f'  seed={int(seeds[idx])}' if seeds is not None else ''
        return (f'  {label} trial {idx:>3d}: hv={final_hv[idx]:.4f}  '
                f'w=[{fmt(ws[idx])}]{seed_str}')

    print('\nFinal-iteration hypervolume summary:')
    print(line('best ', order[-1]))
    print(line('worst', order[0]))

def plot_data(data, savepath):
    fig, axs = plt.subplots(ncols=3, figsize=(15, 5))
    time_ax, hv_ax, overlay_ax = axs

    plot_panel(time_ax, data['times'], 'Query Time', 'Time (s)', 'tab:red')
    plot_panel(hv_ax, data['hvs'], 'Groundtruth Hypervolume', 'Hypervolume', 'tab:blue')
    plot_panel(overlay_ax, data['overlays'], 'Pareto Overlay', 'Fraction correct', 'tab:green')

    fig.tight_layout()
    savepath = Path(savepath)
    savepath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(savepath)
    print(f'Saved figure to {savepath.resolve()}')

def parse_args():
    parser = argparse.ArgumentParser(
        description='Run (or replot) the action-vs-performance experiment.'
    )
    parser.add_argument(
        '--data', type=Path, default=None,
        help='Path to a saved .npz data file. If provided, skip experiments '
             'and plot this data instead.'
    )
    parser.add_argument(
        '--n-trials', type=int, default=20,
        help='Number of trials to run (default: 20).'
    )
    parser.add_argument(
        '--n-queries', type=int, default=20,
        help='Number of queries per trial (default: 5).'
    )
    parser.add_argument(
        '--seed', type=int, default=95,
        help='Base RNG seed (default: 95).'
    )
    parser.add_argument(
        '--save-data', type=Path, default=Path('hilo/output/action_vs_performance.npz'),
        help='Where to save experiment data (default: '
             'hilo/output/action_vs_performance.npz).'
    )
    parser.add_argument(
        '--action-size', type=int, default=1
    )
    parser.add_argument(
        '--fig', type=Path, default=Path('hilo/output/action_vs_performance.pdf'),
        help='Where to save the figure (default: '
             'hilo/output/action_vs_performance.pdf).'
    )
    return parser.parse_args()

def main():
    args = parse_args()

    if args.data is not None:
        data = load_data(args.data)
    else:
        data = run_experiments(
            args.n_trials, 
            args.n_queries, 
            args.seed,
            args.action_size
        )
        save_data(data, args.save_data)

    summarize(data)
    plot_data(data, args.fig)

if __name__ == '__main__':
    main()