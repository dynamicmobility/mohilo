"""Which acquisition function wins on a synthetic objective, over many seeds,
recorded trial by trial into a `Ledger`.
"""

import argparse
import time
from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
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
from pypolar.experiment.ledger import TRIAL_CLOSED

FUNCTION     = 'Levy'   # the groundtruth, held fixed across every run
DIM          = 3        # action dimension
NUM_STEPS    = 30       # acquisition steps taken
NOISE        = 0.5      # noise added, as a fraction of the truth's spread
REPEATS      = 1        # values the probe reports per trial, each its own draw
GP_NOISE     = plr.NoiseModel.pinned(0.5)  # a float pins, None fits, prior() regularizes
MIN_LENGTHSCALE = None  # lengthscale floor, at the design spacing
BOX          = 5.0      # the action box is [-BOX, BOX]^DIM
SEEDS        = tuple(range(30))
STRATEGIES   = ('random', 'ucb', 'logei', 'lognei', 'qlognei')

STUDY_DIR     = Path('hilo/output/acquisition_ledger')
CURVES_NAME   = 'curves.csv'   # written into whichever directory was plotted
REGRET_OUTPUT = Path('hilo/output/acquisition_ledger_regret.png')
ACTION_OUTPUT = Path('hilo/output/acquisition_ledger_action_regret.png')

# acquisition knobs, each read only by the strategy named
UCB_BETA       = 2.0    # ucb: explores sqrt(beta) posterior standard deviations
NUM_FANTASIES  = 20     # lognei: noiseless incumbents drawn; cost is linear in it
MC_SAMPLES     = 128    # qlognei: QMC samples per acquisition evaluation
PRUNE_BASELINE = True   # qlognei: drop measured points that cannot be the best

# the knobs each strategy reads, and so the only ones its log is pinned under:
# raising UCB_BETA must invalidate the ucb logs and leave the logei ones alone
ACQ_KNOBS = {
    'random' : {},
    'ucb'    : {'beta': UCB_BETA},
    'logei'  : {},
    'lognei' : {'num_fantasies': NUM_FANTASIES},
    'qlognei': {'mc_samples': MC_SAMPLES, 'prune_baseline': PRUNE_BASELINE}
}

NUM_RESTARTS = 8        # L-BFGS-B starting points per acquisition optimization
RAW_SAMPLES  = 512      # Sobol samples scanned to pick them
SPREAD_REF   = 4096     # samples used to measure the truth's spread

# 'random' draws its whole design up front, which is what non-adaptive means.
# The offset keeps that draw off the stream the opening action came from, so the
# first acquired point is not a repeat of the opening one.
RANDOM_SEED_OFFSET = 10_000

SCORED           = 'scored'     # the event kind carrying a trial's scored state
SIMPLE_REGRET    = 'simple_regret'
INFERENCE_REGRET = 'inference_regret'
ACTION_SIMPLE    = 'action_simple'
ACTION_INFERENCE = 'action_inference'
ACQ_SECONDS      = 'acq_seconds'

# the scored state after a trial, which is what a curve is made of
CURVE_COLUMNS = (SIMPLE_REGRET, INFERENCE_REGRET, ACTION_SIMPLE, ACTION_INFERENCE,
                 ACQ_SECONDS)
# what a curve is indexed by: `trials` is the budget a session pays, and
# `measurements` the rows the GP was fit to, REPEATS times as many
INDEX_COLUMNS = ('trials', 'measurements')

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


def log_path(strategy, seed):
    return STUDY_DIR / strategy / f'seed_{seed:03d}.jsonl'


def config(strategy, seed):
    """What a log is fingerprinted under: every constant that changes what the
    numbers in it mean.

    A run that resumes under a changed setting is refused rather than appended
    to, so two studies cannot end up in one file. `GP_NOISE` goes in by `repr`
    because a `NoiseModel` is not json, and its repr names all three of its
    fields.

    `NUM_STEPS` is deliberately absent: it is where a run stops, not what its
    numbers mean, and `sample_actions` is prefix-stable in `n`, so raising the
    budget continues every existing log -- 'random' included, whose longer
    design opens with the points it already drew -- rather than discarding it.
    """
    return {
        'function'       : FUNCTION,
        'dim'            : DIM,
        'box'            : BOX,
        'noise'          : NOISE,
        'repeats'        : REPEATS,
        'gp_noise'       : repr(plr.NoiseModel.coerce(GP_NOISE)),
        'min_lengthscale': MIN_LENGTHSCALE,
        'strategy'       : strategy,
        'seed'           : seed,
        'acquisition'    : ACQ_KNOBS[strategy]
    }


def score(ledger, trial, trials, truth, objective, inferred, seconds):
    """Appends the scored state of the run, against the trial that produced it.

    Its own event rather than a field on `trial_closed`, so `replay` -- which
    reads measurements and nothing else -- is untouched by it, and a metric
    added later needs no change to how a session is rebuilt.

    Args:
        trial: the ledger's index for the trial just closed.
        trials: how many trials have closed, which is the budget this row is at.
    """
    simple_value,  inference_value  = plr.regret(truth, objective, inferred)
    simple_action, inference_action = plr.action_distance(truth, objective, inferred)

    ledger.append(SCORED, trial=trial,
                  trials       = trials,
                  measurements = len(objective.ydata),
                  recommended  = inferred,
                  **{SIMPLE_REGRET   : float(simple_value),
                     INFERENCE_REGRET: float(inference_value),
                     ACTION_SIMPLE   : float(simple_action),
                     ACTION_INFERENCE: float(inference_action),
                     ACQ_SECONDS     : float(seconds)})


def run_one(strategy, seed, truth, box, done):
    """One (strategy, seed) session, recorded trial by trial into its log.

    Resumes whatever the log already holds: measurements are the only state a
    session carries, so `replay` rebuilds the objective exactly and the loop
    picks up at the next trial. What does *not* resume is torch's global
    generator and the probe's noise stream, both of which restart here -- so a
    resumed run is a valid run of this strategy but not the same one an
    uninterrupted process would have produced. 'random' is the exception, since
    its whole design is drawn from a fixed seed before anything is measured.

    Args:
        done: trials already closed in the log, which is where the loop starts.
    """
    torch.manual_seed(seed)

    probe = plr.SyntheticProbe(
        functions = {FUNCTION: plr.SyntheticFunction(truth=truth, rel_noise_std=NOISE,
                                                     n_spread=SPREAD_REF, seed=seed)},
        repeats   = REPEATS
    )
    objective = plr.Objective.from_empty(name=FUNCTION, maximize=False,
                                         action_bounds=(-BOX, BOX))

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

    with plr.Ledger(log_path(strategy, seed), config(strategy, seed)) as ledger:
        for trial in ledger.replay(objective):
            # opened, never closed: a measurement the last process died partway
            # through is not a measurement
            ledger.abandon_trial('interrupted by an earlier run', trial=trial)

        gp      = fit_gp(objective) if done else None
        seconds = 0.0   # no acquisition chose the opening action
        for index in range(done, NUM_STEPS + 1):
            if index == 0:
                action = plr.sample_actions(box, 1, 'uniform', seed=seed)[0]
            else:
                start   = time.perf_counter()
                action  = drawn[index - 1] if strategy == 'random' else acq.query(gp, q=1)[0]
                seconds = time.perf_counter() - start

            # the trial's own index, which is one past every index the log
            # holds and so skips over any trial abandoned above
            trial  = ledger.open_trial(action, source='seed' if index == 0 else strategy)
            values = probe.measure(action, record=ledger.record)[FUNCTION]
            ledger.close_trial(
                {FUNCTION: values},
                truth = float(plr.truth_at(truth, np.atleast_2d(action))[0])
            )
            objective.add_points(np.tile(action, (len(values), 1)), values)

            gp       = fit_gp(objective)
            inferred = recommend.query(gp, q=1)[0]
            score(ledger, trial, index + 1, truth, objective, inferred, seconds)


def closed_trials(path):
    """How many trials a log holds.

    Read off the file rather than through a `Ledger`, because attaching one
    appends a session event: a run with nothing left to do must not open its
    own log to find that out.
    """
    if not Path(path).exists():
        return 0

    return sum(1 for event in plr.read_events(path) if event['kind'] == TRIAL_CLOSED)


def curve_from_log(path, strategy, seed):
    """The run's scored history, read back out of its log.

    Only scored trials appear. A trial that was measured but never scored -- a
    crash between closing it and scoring it -- contributes nothing, so a run
    contributes the budgets it actually reached and nothing past them.
    """
    scored = [event for event in plr.read_events(path) if event['kind'] == SCORED]
    if not scored:
        return pd.DataFrame(columns=['strategy', 'seed', *INDEX_COLUMNS, *CURVE_COLUMNS])

    return pd.DataFrame({
        'strategy': strategy,
        'seed'    : seed,
        **{column: [event[column] for event in scored]
           for column in (*INDEX_COLUMNS, *CURVE_COLUMNS)}
    })


def curves_from_directory(directory):
    """Every run recorded under `directory`, as one table.

    The layout is the one `log_path` writes: `<strategy>/seed_NNN.jsonl`, so the
    strategy and the seed are read back off the path rather than from any
    constant here. Nothing is re-run and nothing is required to be complete,
    which is what makes this the way to plot a sweep that died.
    """
    directory = Path(directory)
    logs      = sorted(directory.glob('*/seed_*.jsonl'))
    if not logs:
        raise FileNotFoundError(f'no <strategy>/seed_*.jsonl logs under {directory}')

    curves, empty = [], []
    for path in logs:
        curve = curve_from_log(path, strategy = path.parent.name,
                                     seed     = int(path.stem.split('_')[-1]))
        (curves if len(curve) else empty).append(curve if len(curve) else path)

    print(f'read {len(logs)} logs from {directory}'
          + (f', {len(empty)} with nothing scored' if empty else ''))
    if not curves:
        raise ValueError(f'no log under {directory} holds a scored trial')

    return pd.concat(curves, ignore_index=True)


def coverage(curves):
    """How many seeds back each budget, per strategy.

    A sweep that died leaves runs of unequal length, so a median at a late
    budget is taken over whichever seeds survived to reach it -- a shrinking
    and not-random subset. Printing the counts is what keeps that visible
    instead of letting a curve drawn from four seeds look like one from thirty.
    """
    table = curves.pivot_table(index='trials', columns='strategy',
                               values='seed', aggfunc='count').fillna(0).astype(int)

    print('\nseeds backing each budget')
    print('  ' + f'{"trials":>7}' + ''.join(f'{name:>10}' for name in table.columns))
    for budget, row in table.iterrows():
        print('  ' + f'{budget:>7}' + ''.join(f'{n:>10}' for n in row))

    return table


def load_or_run(strategy, seed, truth, box):
    """The run's curve, continuing its log wherever that log stopped.

    A finished run is never re-opened, so the sweep can be stopped and
    restarted; an unfinished one resumes at its next trial rather than starting
    over, since the log holds every measurement it took. A log recorded under a
    different configuration refuses to be resumed, and is reported here as the
    failure it is rather than being appended to.
    """
    path = log_path(strategy, seed)
    done = closed_trials(path)
    if done > NUM_STEPS:
        print(f'  {strategy:>8s} seed {seed:>3d}  (recorded)')
        return curve_from_log(path, strategy, seed)

    start = time.perf_counter()
    try:
        run_one(strategy, seed, truth, box, done)
    except Exception as error:
        # the sweep is many independent runs, so one failure costs only its own
        # run; whatever it recorded stays on disk and a rerun continues from it
        print(f'  {strategy:>8s} seed {seed:>3d}  FAILED after '
              f'{time.perf_counter() - start:.1f} s: {type(error).__name__}: {error}')
        return curve_from_log(path, strategy, seed) if path.exists() else None

    print(f'  {strategy:>8s} seed {seed:>3d}  {time.perf_counter() - start:6.1f} s'
          + (f'  (resumed at {done})' if done else ''))

    return curve_from_log(path, strategy, seed)


def report(curves):
    """Where each strategy ends up, ranked, each at its own deepest budget.

    A strategy whose runs died early is scored at the budget it actually
    reached rather than at the sweep's maximum, and that budget and the seeds
    behind it are printed beside the numbers: two rows of this table are not
    necessarily compared at the same budget, which has to be visible to be
    read correctly.
    """
    print(f'\n=== {FUNCTION}, {DIM}D, noise {NOISE:.2f} of spread, '
          f'{REPEATS} value(s) a trial, {curves["seed"].nunique()} seeds ===')
    print(f'  {"":>8}{"":>7}{"":>5}{"inference regret":^26}{"action distance":^26}')
    print(f'  {"":>8}{"trials":>7}{"n":>5}  {"median":>8}  {"IQR":>14}'
          f'  {"median":>8}  {"IQR":>14}  {"acq (s)":>9}')

    finals = {}
    for strategy, runs in curves.groupby('strategy'):
        finals[strategy] = (runs[runs['trials'] == runs['trials'].max()],
                            runs.groupby('seed')[ACQ_SECONDS].sum().mean())

    for strategy in sorted(finals, key=lambda s: finals[s][0][INFERENCE_REGRET].median()):
        deepest, cost = finals[strategy]
        print(f'  {strategy:>8s}{deepest["trials"].iloc[0]:>7d}{len(deepest):>5d}'
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

    The budget is trials rather than measurements, since a trial is what a
    session pays for; at REPEATS > 1 the GP behind each point was fit to that
    many times as many values.
    """
    fig, axes = plt.subplots(1, len(panels), figsize=(11, 4.5), sharex=True)

    # whatever the data holds, in the declared order, then anything unrecognized
    present = [s for s in STRATEGIES if s in set(curves['strategy'])]
    present += sorted(set(curves['strategy']) - set(present))

    for ax, (column, title) in zip(np.atleast_1d(axes), panels):
        for index, strategy in enumerate(present):
            runs = curves[curves['strategy'] == strategy]

            grouped     = runs.groupby('trials')[column]
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
        # a trial is one action applied, so a fractional tick counts nothing
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        ax.set_xlabel('trials')
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=9)
        ax.grid(alpha=0.3, lw=0.5)
        ax.set_axisbelow(True)

    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle(f'{FUNCTION}, {DIM}D, {curves["seed"].nunique()} seeds'
                 + (f', {REPEATS} values a trial' if REPEATS > 1 else '')
                 + ' (median, interquartile band)', fontsize=10)

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
