"""Which acquisition function wins on a synthetic objective, over many seeds.
"""

import argparse
import time
from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from ax.api.client import Client
from ax.api.configs import RangeParameterConfig
from botorch.acquisition import (
    LogExpectedImprovement,
    LogNoisyExpectedImprovement,
    PosteriorMean,
    UpperConfidenceBound,
    qLogNoisyExpectedImprovement,
)
from botorch.sampling import SobolQMCNormalSampler
from matplotlib.ticker import MaxNLocator, ScalarFormatter

import pypolar as plr

FUNCTION     = 'Levy'   # the groundtruth, held fixed across every run
DIM          = 3        # action dimension
NUM_STEPS    = 30       # acquisition steps taken
NOISE        = 0.5      # noise added, as a fraction of the truth's spread
GP_NOISE     = plr.NoiseModel.pinned(0.5)  # a float pins, None fits, prior() regularizes
MIN_LENGTHSCALE = None  # lengthscale floor, at the design spacing
BOX          = 5.0      # the action box is [-BOX, BOX]^DIM
SEEDS        = tuple(range(30))
STRATEGIES   = ('random', 'ucb', 'logei', 'lognei', 'qlognei')

STUDY_DIR     = Path('hilo/output/acquisition_study')
CURVES_NAME   = 'curves.csv'   # written into whichever directory was plotted
REGRET_OUTPUT = Path('hilo/output/acquisition_regret.png')
ACTION_OUTPUT = Path('hilo/output/acquisition_action_regret.png')

# acquisition knobs, each read only by the strategy named
UCB_BETA       = 2.0    # ucb: explores sqrt(beta) posterior standard deviations
NUM_FANTASIES  = 20     # lognei: noiseless incumbents drawn; cost is linear in it
MC_SAMPLES     = 128    # qlognei: QMC samples per acquisition evaluation
PRUNE_BASELINE = True   # qlognei: drop measured points that cannot be the best

NUM_RESTARTS = 8        # L-BFGS-B starting points per acquisition optimization
RAW_SAMPLES  = 512      # Sobol samples scanned to pick them
SPREAD_REF   = 4096     # samples used to measure the truth's spread

# 'random' draws its whole design up front, which is what non-adaptive means.
# The offset keeps that draw off the stream the opening action came from, so the
# first acquired point is not a repeat of the opening one.
RANDOM_SEED_OFFSET = 10_000

# the metrics each trial carries in the Ax record
MEASURED         = FUNCTION.lower()             # what the instrument reported
TRUE             = f'{FUNCTION.lower()}_true'   # the noiseless value there
SIMPLE_REGRET    = 'simple_regret'
INFERENCE_REGRET = 'inference_regret'
ACTION_SIMPLE    = 'action_simple'
ACTION_INFERENCE = 'action_inference'
ACQ_SECONDS      = 'acq_seconds'

# the scored state after a trial, which is what a curve is made of
CURVE_COLUMNS = (SIMPLE_REGRET, INFERENCE_REGRET, ACTION_SIMPLE, ACTION_INFERENCE,
                 ACQ_SECONDS)
TRACKED = (TRUE, *CURVE_COLUMNS, *(f'recommended_x{i}' for i in range(DIM)))

# Okabe-Ito, a published colorblind-safe palette; the non-adaptive baseline is
# the dashed black one so it reads as the reference rather than a competitor
STRATEGY_STYLE = {
    'random' : ('#000000', '--'),
    'ucb'    : ('#E69F00', '-'),
    'logei'  : ('#56B4E9', '-'),
    'lognei' : ('#009E73', '-'),
    'qlognei': ('#D55E00', '-')
}
# the rest of the palette, for a strategy read off disk that is not named above
FALLBACK_COLORS = ('#CC79A7', '#F0E442', '#0072B2')

VALUE_CURVES = (
    (SIMPLE_REGRET,    'simple regret (best true value sampled)'),
    (INFERENCE_REGRET, 'inference regret (true value at the posterior argmax)')
)
ACTION_CURVES = (
    (ACTION_SIMPLE,    'closest action sampled'),
    (ACTION_INFERENCE, 'the posterior argmax itself')
)


def acquisition_factory(strategy, seed):
    if strategy == 'lognei' and plr.NoiseModel.coerce(GP_NOISE).is_fitted:
        raise ValueError("'lognei' needs a FixedNoiseGaussianLikelihood, so set "
                         'GP_NOISE = plr.NoiseModel.pinned(...) to use it')

    factories = {
        'ucb'    : partial(UpperConfidenceBound, beta=UCB_BETA),
        'logei'  : LogExpectedImprovement,
        'lognei' : partial(LogNoisyExpectedImprovement, num_fantasies=NUM_FANTASIES),
        'qlognei': partial(
            qLogNoisyExpectedImprovement,
            sampler        = SobolQMCNormalSampler(torch.Size([MC_SAMPLES]), seed=seed),
            prune_baseline = PRUNE_BASELINE
        )
    }
    if strategy not in factories:
        raise ValueError(f'no acquisition for {strategy!r}')

    return factories[strategy]


def fit_gp(objective):
    return plr.BoTorchGP(objective, noise=GP_NOISE, fit_hyperparameters=False,
                         min_length_scale=MIN_LENGTHSCALE, length_scale=0.3, signal_var=1.0)


def snapshot_path(strategy, seed):
    return STUDY_DIR / strategy / f'seed_{seed:03d}.json'


def build_client(strategy, seed, collected_data: list[str], param_names=None):
    """An Ax experiment that records trials and never generates one.
    """
    if param_names is None:
        param_names = [
            RangeParameterConfig(
                name           = f'x{i}',
                parameter_type = 'float',
                bounds         = (-BOX, BOX)
            ) 
            for i in range(DIM)
        ]
        
    client = Client(random_seed=seed)
    client.configure_experiment(
        name       = f'{FUNCTION.lower()}_{strategy}_seed{seed:03d}',
        parameters = param_names
    )
    client.configure_optimization(objective=f'-{MEASURED}') # TODO: determine if this must be run
    client.configure_tracking_metrics(metric_names=collected_data)

    return client


def measure(client, action, observe, truth):
    """Runs one trial: opens it at `action`, measures, and closes it.
    """
    parameters   = {f'x{i}': float(v) for i, v in enumerate(np.ravel(action))}
    trial_index  = client.attach_trial(parameters)
    value        = float(observe(np.atleast_2d(action))[0])

    client.complete_trial(trial_index, raw_data={
        MEASURED: value,
        TRUE    : float(plr.truth_at(truth, np.atleast_2d(action))[0])
    })

    return trial_index, value


def record_state(client, trial_index, truth, objective, inferred, seconds):
    """Attaches the scored state of the run to the trial that produced it.

    Both metrics are read after the trial closed, so row `t` is the state of a
    run that has taken `t + 1` measurements.
    """
    simple_value,  inference_value  = plr.regret(truth, objective, inferred)
    simple_action, inference_action = plr.action_distance(truth, objective, inferred)

    client.attach_data(trial_index, raw_data={
        SIMPLE_REGRET   : float(simple_value),
        INFERENCE_REGRET: float(inference_value),
        ACTION_SIMPLE   : float(simple_action),
        ACTION_INFERENCE: float(inference_action),
        ACQ_SECONDS     : float(seconds),
        **{f'recommended_x{i}': float(v) for i, v in enumerate(inferred)}
    })


def run_one(strategy, seed, truth, box):
    """One (strategy, seed) session, recorded trial by trial into Ax.

    Returns the client, whose snapshot on disk already holds everything the
    caller needs.
    """
    torch.manual_seed(seed)

    observe = plr.SyntheticFunction(truth=truth, rel_noise_std=NOISE,
                                    n_spread=SPREAD_REF, seed=seed)
    objective = plr.Objective.from_empty(name=FUNCTION, maximize=False,
                                         action_bounds=(-BOX, BOX))

    client = build_client(strategy, seed)
    path   = snapshot_path(strategy, seed)
    path.parent.mkdir(parents=True, exist_ok=True)

    opening = plr.sample_actions(box, 1, 'uniform', seed=seed)
    trial_index, value = measure(client, opening[0], observe, truth)
    objective.add_points(opening, np.atleast_1d(value))
    client.save_to_json_file(str(path))

    # the non-adaptive baseline commits to its design before seeing anything
    drawn = (plr.sample_actions(box, NUM_STEPS, 'uniform',
                                seed=seed + RANDOM_SEED_OFFSET)
             if strategy == 'random' else None)
    acq = (None if strategy == 'random' else
           plr.AcquisitionFunction(acquisition_factory(strategy, seed), objective,
                                   num_restarts=NUM_RESTARTS, raw_samples=RAW_SAMPLES))
    recommend = plr.AcquisitionFunction(PosteriorMean, objective,
                                        num_restarts=NUM_RESTARTS,
                                        raw_samples=RAW_SAMPLES)

    seconds = 0.0   # no acquisition chose the opening action
    for step in range(NUM_STEPS + 1):
        gp       = fit_gp(objective)
        inferred = recommend.query(gp, q=1)[0]
        record_state(client, trial_index, truth, objective, inferred, seconds)
        client.save_to_json_file(str(path))

        if step == NUM_STEPS:
            break

        start  = time.perf_counter()
        action = drawn[step] if strategy == 'random' else acq.query(gp, q=1)[0]
        seconds = time.perf_counter() - start

        trial_index, value = measure(client, action, observe, truth)
        objective.add_points(np.atleast_2d(action), np.atleast_1d(value))
        client.save_to_json_file(str(path))

    return client


def curve_from_client(client, strategy, seed):
    """The run's scored history, read back out of the Ax record.

    Only scored trials survive. A trial that was measured but never scored --
    a crash between closing it and attaching its metrics -- is dropped, so a
    run contributes the budgets it actually reached and nothing past them.
    """
    trials = client.summarize()
    trials = trials[trials['trial_status'] == 'COMPLETED'].sort_values('trial_index')

    curve = pd.DataFrame({
        'strategy'    : strategy,
        'seed'        : seed,
        'measurements': trials['trial_index'].to_numpy() + 1,
        # a column is absent entirely when nothing in the run was ever scored
        **{column: (trials[column].to_numpy() if column in trials.columns else np.nan)
           for column in CURVE_COLUMNS}
    })

    return curve.dropna(subset=list(CURVE_COLUMNS))


def curves_from_directory(directory):
    """Every run recorded under `directory`, as one table.

    The layout is the one `snapshot_path` writes: `<strategy>/seed_NNN.json`,
    so the strategy and the seed are read back off the path rather than from
    any constant here. Nothing is re-run and nothing is required to be
    complete, which is what makes this the way to plot a sweep that died.
    """
    directory = Path(directory)
    snapshots = sorted(directory.glob('*/seed_*.json'))
    if not snapshots:
        raise FileNotFoundError(f'no <strategy>/seed_*.json snapshots under {directory}')

    curves, empty = [], []
    for path in snapshots:
        curve = curve_from_client(Client.load_from_json_file(str(path)),
                                  strategy = path.parent.name,
                                  seed     = int(path.stem.split('_')[-1]))
        (curves if len(curve) else empty).append(curve if len(curve) else path)

    print(f'read {len(snapshots)} snapshots from {directory}'
          + (f', {len(empty)} with nothing scored' if empty else ''))
    if not curves:
        raise ValueError(f'no snapshot under {directory} holds a scored trial')

    return pd.concat(curves, ignore_index=True)


def coverage(curves):
    """How many seeds back each budget, per strategy.

    A sweep that died leaves runs of unequal length, so a median at a late
    budget is taken over whichever seeds survived to reach it -- a shrinking
    and not-random subset. Printing the counts is what keeps that visible
    instead of letting a curve drawn from four seeds look like one from thirty.
    """
    table = curves.pivot_table(index='measurements', columns='strategy',
                               values='seed', aggfunc='count').fillna(0).astype(int)

    print('\nseeds backing each budget')
    print('  ' + f'{"budget":>7}' + ''.join(f'{name:>10}' for name in table.columns))
    for budget, row in table.iterrows():
        print('  ' + f'{budget:>7}' + ''.join(f'{n:>10}' for n in row))

    return table


def is_complete(client):
    """Whether a snapshot holds a whole run, so re-running it would add nothing."""
    trials = client.summarize()
    if len(trials) != NUM_STEPS + 1:
        return False
    if any(column not in trials.columns for column in CURVE_COLUMNS):
        return False

    return not trials[list(CURVE_COLUMNS)].isna().to_numpy().any()


def load_or_run(strategy, seed, truth, box):
    """The run's curve, from its snapshot when that snapshot is whole.

    A finished run is never repeated, so the sweep can be stopped and restarted;
    a snapshot a crash left partway through is run again from the opening
    action, since a half-finished session is not a session.
    """
    path = snapshot_path(strategy, seed)
    if path.exists():
        client = Client.load_from_json_file(str(path))
        if is_complete(client):
            print(f'  {strategy:>8s} seed {seed:>3d}  (recorded)')
            return curve_from_client(client, strategy, seed)

    start = time.perf_counter()
    try:
        client = run_one(strategy, seed, truth, box)
    except Exception as error:
        # the sweep is many independent runs, so one failure costs only its own
        # run; whatever it recorded stays on disk and a rerun retries just it
        print(f'  {strategy:>8s} seed {seed:>3d}  FAILED after '
              f'{time.perf_counter() - start:.1f} s: {type(error).__name__}: {error}')
        return (curve_from_client(Client.load_from_json_file(str(path)), strategy, seed)
                if path.exists() else None)

    print(f'  {strategy:>8s} seed {seed:>3d}  {time.perf_counter() - start:6.1f} s')

    return curve_from_client(client, strategy, seed)


def report(curves):
    """Where each strategy ends up, ranked, each at its own deepest budget.

    A strategy whose runs died early is scored at the budget it actually
    reached rather than at the sweep's maximum, and that budget and the seeds
    behind it are printed beside the numbers: two rows of this table are not
    necessarily compared at the same budget, which has to be visible to be
    read correctly.
    """
    print(f'\n=== {FUNCTION}, {DIM}D, noise {NOISE:.2f} of spread, '
          f'{curves["seed"].nunique()} seeds ===')
    print(f'  {"":>8}{"":>7}{"":>5}{"inference regret":^26}{"action distance":^26}')
    print(f'  {"":>8}{"budget":>7}{"n":>5}  {"median":>8}  {"IQR":>14}'
          f'  {"median":>8}  {"IQR":>14}  {"acq (s)":>9}')

    finals = {}
    for strategy, runs in curves.groupby('strategy'):
        finals[strategy] = (runs[runs['measurements'] == runs['measurements'].max()],
                            runs.groupby('seed')[ACQ_SECONDS].sum().mean())

    for strategy in sorted(finals, key=lambda s: finals[s][0][INFERENCE_REGRET].median()):
        deepest, cost = finals[strategy]
        print(f'  {strategy:>8s}{deepest["measurements"].iloc[0]:>7d}{len(deepest):>5d}'
              f'  {deepest[INFERENCE_REGRET].median():>8.4f}'
              f'  [{deepest[INFERENCE_REGRET].quantile(0.25):>5.3f},'
              f' {deepest[INFERENCE_REGRET].quantile(0.75):>5.3f}]'
              f'  {deepest[ACTION_INFERENCE].median():>8.4f}'
              f'  [{deepest[ACTION_INFERENCE].quantile(0.25):>5.3f},'
              f' {deepest[ACTION_INFERENCE].quantile(0.75):>5.3f}]'
              f'  {cost:>9.2f}')
    print()


def make_figure(curves, panels, ylabel, output):
    """Each strategy's median curve with an interquartile band, one panel each.

    The median and the quartiles are taken across seeds at every budget. A mean
    would be pulled around by the one run in thirty that never leaves its
    starting basin, which is a property of the truth's multimodality rather than
    of the strategy being scored.
    """
    fig, axes = plt.subplots(1, len(panels), figsize=(11, 4.5), sharex=True)

    # whatever the data holds, in the declared order, then anything unrecognized
    present = [s for s in STRATEGIES if s in set(curves['strategy'])]
    present += sorted(set(curves['strategy']) - set(present))

    for ax, (column, title) in zip(np.atleast_1d(axes), panels):
        for index, strategy in enumerate(present):
            runs = curves[curves['strategy'] == strategy]

            grouped     = runs.groupby('measurements')[column]
            median      = grouped.median()
            low, high   = grouped.quantile(0.25), grouped.quantile(0.75)
            color, dash = STRATEGY_STYLE.get(
                strategy, (FALLBACK_COLORS[index % len(FALLBACK_COLORS)], '-'))

            ax.plot(median.index, median.to_numpy(), color=color, ls=dash, lw=1.5,
                    label=strategy)
            ax.fill_between(median.index, low.to_numpy(), high.to_numpy(),
                            color=color, alpha=0.12, lw=0)

        ax.set_yscale('log')
        # inside one decade the major ticks label nothing, so the minor ones are
        # labelled instead; across more than one they would round neighbouring
        # values to the same digit and read as a column of repeats
        low, high = ax.get_ylim()
        if high < 10 * low:
            ax.yaxis.set_minor_formatter(ScalarFormatter())
            ax.tick_params(axis='y', which='minor', labelsize=7)
        # a measurement is one trial, so a fractional tick counts nothing
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        ax.set_xlabel('measurements')
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=9)
        ax.grid(alpha=0.3, lw=0.5)
        ax.set_axisbelow(True)

    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle(f'{FUNCTION}, {DIM}D, {curves["seed"].nunique()} seeds '
                 f'(median, interquartile band)', fontsize=10)

    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200)
    print(f'wrote {output}')


def sweep():
    """Every (strategy, seed) run, recorded to disk, as one table of curves."""
    box   = np.array([[-BOX] * DIM, [BOX] * DIM], dtype=float)
    truth = plr.construct_function(
        func = plr.SYNTHETIC_FUNCTIONS[FUNCTION],
        dim  = DIM,
        box  = BOX,
        seed = 0,      # the truth is a held constant, so it is built once
    )

    print(f'{len(STRATEGIES)} strategies x {len(SEEDS)} seeds')
    curves = [load_or_run(strategy, seed, truth, box)
              for strategy in STRATEGIES for seed in SEEDS]
    curves = [curve for curve in curves if curve is not None and len(curve)]
    if not curves:
        raise RuntimeError('every run failed before recording a scored trial')

    return pd.concat(curves, ignore_index=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        '--plot', metavar='DIR', type=Path, default=None,
    )
    args = parser.parse_args(argv)

    directory = args.plot if args.plot is not None else STUDY_DIR
    curves    = curves_from_directory(args.plot) if args.plot is not None else sweep()

    output = directory / CURVES_NAME
    output.parent.mkdir(parents=True, exist_ok=True)
    curves.to_csv(output, index=False)
    print(f'wrote {output}')

    coverage(curves)
    report(curves)
    make_figure(curves, VALUE_CURVES, 'regret', REGRET_OUTPUT)
    make_figure(curves, ACTION_CURVES,
                'distance to the nearest optimizer (box spans)', ACTION_OUTPUT)


if __name__ == '__main__':
    main()
