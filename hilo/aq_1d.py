import os
os.environ["JAX_PLATFORMS"] = "cpu"
import pypolar as plr
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from hilo.create import create_1d_sim
from config.base import (
    RandomSampling, ThompsonSampling, ExpectedImprovement, KnowledgeGradient,
    MaxValueEntropy
)
from config.aq_test import sim_1d, sim_2d, sim_3d

CONFIG = sim_3d
SAMPLERS = {
    'Random':   RandomSampling(),
    'Thompson': ThompsonSampling(),
    'EI':       ExpectedImprovement(),
    'KG':       KnowledgeGradient(num_candidates=20),
    # 'MES':      MaxValueEntropy(),
}
SEEDS = range(10)
NUM_QUERIES = 50


def run(sampler_cfg, seed, num_queries):
    """Run one HILO simulation.

    Args:
        sampler_cfg: a sampling config, e.g. ``KnowledgeGradient()``.
        seed: seed for the oracle noise and the sampler.
        num_queries: number of oracle queries.

    Returns:
        A dict with the per-iteration ``regret``, ``rmse`` and ``query_idx``,
        plus the final ``regression``, ``optimizer`` and ``groundtruth``.
    """
    rng = np.random.default_rng(seed)

    config = CONFIG.model_copy(deep=True)
    config.sampler = sampler_cfg
    lengthscale, signal_var, precision = plr.derive_gp_hyperparams(
        domain_size       = float(np.max(config.problem.action_high
                                         - config.problem.action_low)),
        expected_range    = float(config.objective.upper_bound
                                  - config.objective.lower_bound),
        noise_var         = float(config.oracle.noise_std) ** 2
    )
    config.problem.precision            = precision
    config.optimizer.signal_variance    = signal_var
    config.optimizer.length_scale       = lengthscale
    regression, optimizer, sampler, groundtruth, oracle = create_1d_sim(
        rng,
        cfg=config
    )

    reward = groundtruth(regression.action_space).ravel()
    regret, rmse, query_idx = [], [], []

    for _ in range(num_queries):
        sample_action = sampler.sample(regression.action_space)
        mct_hat = oracle.query(sample_action)

        regression.add_feedback(sample_action, mct_hat[0])
        optimizer.set_data(
            action_space    = regression.action_space,
            idx             = regression.feedback_data[:, 0],
            y               = regression.feedback_data[:, 1],
            precision       = regression.precision
        )
        optimizer.fit()
        sampler.update_posterior()

        regret.append(reward.max() - reward[np.argmax(optimizer.mu)])
        rmse.append(np.sqrt(np.mean((optimizer.mu - reward) ** 2)))
        query_idx.append(regression.get_idx(sample_action))

    return {
        'regret':      np.array(regret),
        'rmse':        np.array(rmse),
        'query_idx':   np.array(query_idx),
        'regression':  regression,
        'optimizer':   optimizer,
        'groundtruth': groundtruth,
    }


def plot_metrics(results, savepath):
    """Plot regret and RMSE against iteration, mean +/- standard error."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    iters = np.arange(1, NUM_QUERIES + 1)

    for ax, key, label, scale in zip(axes, ['regret', 'rmse'],
                                     ['Simple Regret', 'Reward RMSE'],
                                     ['linear', 'log']):
        for name, runs in results.items():
            curves = np.array([run_[key] for run_ in runs])
            mean = curves.mean(axis=0)
            err = curves.std(axis=0, ddof=1) / np.sqrt(len(curves))
            ax.plot(iters, mean, label=name)
            ax.fill_between(iters, mean - err, mean + err, alpha=0.2)
        ax.set_xlabel('Query')
        ax.set_ylabel(label)
        ax.set_yscale(scale)
        ax.legend()

    fig.tight_layout()
    fig.savefig(savepath)
    plt.close(fig)


def plot_fits(results, savepath):
    """Plot the final GP fit of the first run of each sampler.

    Only defined for a 1D action space; a no-op otherwise.
    """
    fig, axes = plt.subplots(
        1, len(results), figsize=(4.5 * len(results), 4), sharey=True
    )
    for ax, (name, runs) in zip(np.atleast_1d(axes), results.items()):
        run_ = runs[0]
        regression, optimizer = run_['regression'], run_['optimizer']
        plr.plot_gp_1d(
            ax              = ax,
            mu              = optimizer.mu,
            std             = optimizer.std(),
            action_space    = regression.action_space,
            feedback_idxs   = regression.get_feedback_idxs(),
            feedback_values = regression.get_feedback_values(),
            ground_truth    = run_['groundtruth']
        )
        ax.set_title(name)

    fig.tight_layout()
    fig.savefig(savepath)
    plt.close(fig)


def main():
    results = {}
    for name, sampler_cfg in SAMPLERS.items():
        results[name] = [
            run(sampler_cfg, seed, NUM_QUERIES)
            for seed in tqdm(SEEDS, desc=name)
        ]

    outdir = Path(CONFIG.save_dir)
    plot_metrics(results, outdir / 'acquisition_regret.svg')
    if CONFIG.problem.action_low.shape[0] == 1:
        plot_fits(results, outdir / 'acquisition_fits.svg')
    print(f'Saved figures to {outdir.resolve()}')


if __name__ == '__main__':
    main()
