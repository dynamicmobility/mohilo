import os
os.environ["JAX_PLATFORMS"] = "cpu"
import time
import pypolar as plr
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from hilo.create import create_hipexo_sim
from config.base import RandomSampling, DSTS, QNEHVI
from config.hipexo import hipexo_sim_idealized, hipexo_sim_idealized_2d, hipexo_sim_idealized_3d

CONFIG = hipexo_sim_idealized_2d
SAMPLERS = {
    'Random': RandomSampling(),
    'DSTS':   DSTS(rho=0.01),
    'qNEHVI': QNEHVI(num_samples=128),
}
SEEDS = range(10)
NUM_QUERIES = 50
# Fraction of each objective's range within which points count as tied when
# extracting a Pareto front; matches hipexo_sim/experiment.py.
TOL = 0.02
# The shipped hipexo configs are noiseless, where every sampler saturates the
# hypervolume within a few queries and the comparison says nothing. qNEHVI is a
# noisy-observation acquisition, so the oracle gets noise and the GP
# hyperparameters are derived from it, as in aq_1d.py.
NOISE_STD = 1.0
# qNEHVI is built from BoTorch models, so every arm runs on the same backend
# and the comparison isolates the acquisition rather than the GP.
GPTYPE = 'BoTorchGP'


def run(sampler_cfg, seed, num_queries):
    """Run one multi-objective HILO simulation.

    Args:
        sampler_cfg: a sampling config, e.g. ``QNEHVI()``.
        seed: seed for the oracle noise and the sampler.
        num_queries: number of oracle queries.

    Returns:
        A dict with the per-iteration ``hv``, ``overlay``, ``query_idx`` and
        ``time``, plus the final ``regression``, ``optimizer`` and
        ``groundtruth``.
    """
    rng = np.random.default_rng(seed)

    config = CONFIG.model_copy(deep=True)
    config.sampler = sampler_cfg
    config.optimizer.gptype = GPTYPE

    lengthscale, signal_var, precision = plr.derive_gp_hyperparams(
        domain_size       = float(np.max(config.problem.action_high
                                         - config.problem.action_low)),
        expected_range    = float(np.max(config.objective.upper_bound
                                         - config.objective.lower_bound)),
        noise_var         = NOISE_STD ** 2
    )
    config.oracle.noise_std             = np.full(config.num_objs, NOISE_STD)
    config.problem.precisions           = np.full(config.num_objs, precision)
    config.optimizer.signal_variances   = [signal_var] * config.num_objs
    config.optimizer.length_scales      = [lengthscale] * config.num_objs

    regression, optimizer, sampler, groundtruth, oracle = create_hipexo_sim(
        rng,
        cfg=config
    )

    true_objs = np.asarray(groundtruth(regression.action_space))
    hv, overlay, query_idx, times = [], [], [], []

    for _ in range(num_queries):
        start = time.time()
        sample_action = sampler.sample(regression.action_space)
        values = oracle.query(sample_action)

        regression.add_feedback(sample_action, values)
        optimizer.setup(
            action_space    = regression.action_space,
            regressions     = regression.get_regression_data()
        )
        optimizer.fit()
        sampler.update_posterior()
        times.append(time.time() - start)

        estimated_objs = np.array(
            [gp.mu for gp in optimizer.gps]
        ).T                                              # (N, num_objs)
        hv.append(plr.groundtruth_hypervolume(estimated_objs, true_objs, tol=TOL))
        overlay.append(plr.pareto_overlay(estimated_objs, true_objs, tol=TOL))
        query_idx.append(regression.get_idx(sample_action))

    return {
        'hv':          np.array(hv),
        'overlay':     np.array(overlay),
        'query_idx':   np.array(query_idx),
        'time':        np.array(times),
        'regression':  regression,
        'optimizer':   optimizer,
        'groundtruth': groundtruth,
    }


def plot_metrics(results, savepath):
    """Plot hypervolume and Pareto overlay against iteration, mean +/- standard error."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    iters = np.arange(1, NUM_QUERIES + 1)

    for ax, key, label in zip(axes, ['hv', 'overlay'],
                              ['Groundtruth Hypervolume', 'Pareto Overlay']):
        for name, runs in results.items():
            curves = np.array([run_[key] for run_ in runs])
            mean = curves.mean(axis=0)
            err = curves.std(axis=0, ddof=1) / np.sqrt(len(curves))
            ax.plot(iters, mean, label=name)
            ax.fill_between(iters, mean - err, mean + err, alpha=0.2)
        ax.set_xlabel('Query')
        ax.set_ylabel(label)
        ax.legend()

    fig.tight_layout()
    fig.savefig(savepath)
    plt.close(fig)


def plot_fronts(results, savepath):
    """Plot the final estimated Pareto front of the first run of each sampler.

    Only defined for two objectives; a no-op otherwise.
    """
    fig, axes = plt.subplots(
        1, len(results), figsize=(5.0 * len(results), 4.5), sharex=True, sharey=True
    )
    for ax, (name, runs) in zip(np.atleast_1d(axes), results.items()):
        run_ = runs[0]
        plr.plot_pareto_2d(
            ax                = ax,
            optimizer         = run_['optimizer'],
            regression        = run_['regression'],
            mo_ground_truth   = run_['groundtruth']
        )
        ax.set_title(name)
        ax.set_xlabel('Obj 1: MCT Reward')
        ax.set_ylabel('Obj 2: SP Reward')

    fig.tight_layout()
    fig.savefig(savepath)
    plt.close(fig)


def save_curves(results, savepath):
    """Persist the per-seed metric curves so runs can be re-analyzed."""
    arrays = {}
    for name, runs in results.items():
        for key in ('hv', 'overlay', 'time'):
            arrays[f'{name}_{key}'] = np.array([run_[key] for run_ in runs])
    np.savez(savepath, **arrays)


def queries_to_reach(curves, threshold):
    """Mean number of queries each run needed to first reach ``threshold``.

    Runs that never reach it count as ``NUM_QUERIES``, so the number stays
    finite and comparable.

    Args:
        curves: (n_seeds, n_queries) array of a metric over iterations.
        threshold: level to reach.

    Returns:
        The mean, as a float.
    """
    reached = curves >= threshold
    first = np.where(reached.any(axis=1), reached.argmax(axis=1) + 1, curves.shape[1])
    return float(first.mean())


def print_summary(results):
    """Report how fast each sampler got there, and what it cost.

    On an easy problem every sampler saturates the hypervolume well before the
    last query, so the final value alone separates nothing. What distinguishes
    them is the transient: the metric early on, and how many queries it took to
    get there.
    """
    checkpoints = [c for c in (5, 10, 25, NUM_QUERIES) if c <= NUM_QUERIES]

    for key, label, threshold in (('hv', 'Groundtruth hypervolume', 0.99),
                                  ('overlay', 'Pareto overlay', 0.75)):
        print(f'\n{label} (mean +/- standard error over {len(SEEDS)} seeds)')
        header = ''.join(f'{f"@{c}":>16s}' for c in checkpoints)
        print(f'{"sampler":10s}{header}{f"queries to {threshold:g}":>20s}')
        for name, runs in results.items():
            curves = np.array([run_[key] for run_ in runs])
            cells = ''
            for c in checkpoints:
                col = curves[:, c - 1]
                cells += f'{col.mean():9.4f} +/-{col.std(ddof=1) / np.sqrt(len(col)):5.3f}'
            print(f'{name:10s}{cells}{queries_to_reach(curves, threshold):20.1f}')

    print('\nCost')
    print(f'{"sampler":10s}{"ms/query":>12s}{"total s":>12s}')
    for name, runs in results.items():
        times = np.array([run_['time'] for run_ in runs])
        print(f'{name:10s}{times.mean() * 1e3:12.1f}{times.sum(axis=1).mean():12.2f}')


def main():
    results = {}
    for name, sampler_cfg in SAMPLERS.items():
        results[name] = [
            run(sampler_cfg, seed, NUM_QUERIES)
            for seed in tqdm(SEEDS, desc=name)
        ]

    outdir = Path(CONFIG.save_dir)
    plot_metrics(results, outdir / 'acquisition_mo_metrics.svg')
    if CONFIG.num_objs == 2:
        plot_fronts(results, outdir / 'acquisition_mo_fronts.svg')
    save_curves(results, outdir / 'acquisition_mo_curves.npz')
    print_summary(results)
    print(f'\nSaved figures and curves to {outdir.resolve()}')


if __name__ == '__main__':
    main()
