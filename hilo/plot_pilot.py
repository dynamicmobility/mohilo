import os
os.environ["JAX_PLATFORMS"] = "cpu"
import pypolar as plr
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from dataclasses import dataclass
from tqdm import tqdm
from hilo.create import create_pilot_regression
from config.hipexo import hipexo_pilot
import pandas as pd

NUM_QUERIES = 15
SEED        = 95
NOISE_FRAC  = 0.25   # noise_var as a fraction of the mean objective range


@dataclass
class Objective:
    """One metric column of the pilot data, and which direction is better."""
    name:     str    # label used when printing
    column:   str    # column in the metrics CSV
    maximize: bool   # True if larger is better


@dataclass
class PilotRun:
    """Everything one fit produced, for reporting."""
    objectives:   list
    regression:   object
    optimizer:    object
    best_idxs:    list    # action index of the optimum, per objective
    means:        list    # mean subtracted from each objective before fitting
    max_distance: float   # action space diagonal


def load_pilot_data(num_queries=NUM_QUERIES):
    """Metric values and the actions that produced them, first num_queries runs."""
    metrics_df = pd.read_csv('human_data/pilot_mohilo.csv', index_col='Run')
    actions_df = pd.read_csv('human_data/MH01_walk.csv', index_col='trial_name')

    actions = actions_df.to_numpy(dtype=float)[:num_queries, :-1]
    actions[:, 2] /= 160
    return metrics_df.iloc[:num_queries], actions


def run_pilot(objectives, metrics_df, actions, seed=SEED, noise_frac=NOISE_FRAC):
    """Fit one MultiObjectiveGP over the given objectives and locate each optimum.

    Objectives are centered before the hyperparameters are derived: the GPs have
    a zero prior mean, so on uncentered data the posterior decays toward 0 away
    from the observations and the argmin/argmax lands on the least-explored
    corner rather than the best action.
    """
    rng = np.random.default_rng(seed)
    cfg = hipexo_pilot.model_copy(deep=True)   # each run gets its own config

    values, means = [], []
    for obj in objectives:
        raw  = metrics_df[obj.column].to_numpy(dtype=float)
        mean = raw.mean()
        values.append(raw - mean)
        means.append(mean)

    mean_range = float(np.mean([v.max() - v.min() for v in values]))

    lengthscale, signal_var, precision = plr.derive_gp_hyperparams(
        domain_size       = float(np.max(cfg.problem.action_high
                                            - cfg.problem.action_low)),
        expected_range    = mean_range,
        noise_var         = mean_range * noise_frac
    )
    cfg.num_objs                        = len(objectives)
    cfg.problem.precisions              = np.full(cfg.num_objs, precision)
    cfg.optimizer.signal_variances      = [signal_var] * cfg.num_objs
    cfg.optimizer.length_scales         = [lengthscale] * cfg.num_objs

    regression, optimizer = create_pilot_regression(rng, cfg=cfg)

    label = ' + '.join(obj.name for obj in objectives)
    for i in tqdm(range(len(actions)), desc=label):
        regression.add_feedback(actions[i], [v[i] for v in values])

    optimizer.setup(
        action_space = regression.action_space,
        regressions  = regression.get_regression_data()
    )
    optimizer.fit()

    best_idxs = [
        int(np.argmax(gp.mu) if obj.maximize else np.argmin(gp.mu))
        for obj, gp in zip(objectives, optimizer.gps)
    ]
    max_distance = float(np.linalg.norm(cfg.problem.action_high
                                            - cfg.problem.action_low))

    return PilotRun(objectives, regression, optimizer,
                    best_idxs, means, max_distance)


def report(run):
    """Print the optimum for each objective and how far apart the optima are."""
    actions = [run.regression.action_space[i] for i in run.best_idxs]
    label   = ' + '.join(obj.name for obj in run.objectives)
    width   = max(len(obj.name) for obj in run.objectives)

    print(f'\n=== {label} ===')
    print(f'action space: {run.regression.action_space.shape}')

    # prior std is k(x, x) + jitter, the same for every action, so it is the
    # baseline std() decays from as data comes in
    for obj, gp in zip(run.objectives, run.optimizer.gps):
        print(f'  prior std {obj.name:<{width}} = {np.sqrt(gp.prior_var()):.4f}')

    for obj, gp, idx, action, mean in zip(run.objectives, run.optimizer.gps,
                                          run.best_idxs, actions, run.means):
        direction = 'max' if obj.maximize else 'min'
        print(f'  best {obj.name:<{width}} ({direction}): {action} '
              f'(mu = {gp.mu[idx] + mean:.4f}, std = {gp.std()[idx]:.4f})')

    # how far apart the optima are, relative to the box diagonal (the furthest
    # two points in the action space can possibly be)
    for i in range(len(actions)):
        for j in range(i + 1, len(actions)):
            separation = float(np.linalg.norm(actions[i] - actions[j]))
            pair = f'{run.objectives[i].name} <-> {run.objectives[j].name}'
            print(f'  separation {pair}: {separation:.4f} / {run.max_distance:.4f} '
                  f'= {100 * separation / run.max_distance:.2f}% of the diagonal')


def main():
    metrics_df, actions = load_pilot_data()

    metabolic_comfort = [
        Objective('metabolic cost',   'Cost',              maximize=False),
        Objective('treadmill comfort', 'Comfort Treadmill', maximize=True),
    ]
    speed_comfort = [
        Objective('speed',         'Speed',         maximize=False),
        Objective('floor comfort', 'Comfort Floor', maximize=True),
    ]

    for objectives in (metabolic_comfort, speed_comfort):
        report(run_pilot(objectives, metrics_df, actions))


if __name__ == '__main__':
    main()
