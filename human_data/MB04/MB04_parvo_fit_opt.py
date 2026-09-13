"""MB04 metabolic cost per optimization condition, from the Parvo breath-by-breath CSV.

Each condition's Brockway metabolic rate is fit with a first-order response
`a + b*exp(-t/c)`, and the fit's mean over 240-360 s, minus the standing rate,
is the condition's estimated cost.
"""

from pathlib import Path
import pypolar as plr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import quad
from scipy.optimize import curve_fit

HERE        = Path(__file__).parent
DATA_PATH   = HERE / 'MB04__0_20231223_0713.CSV'
OUTPUT_PATH = HERE / 'parvo_results_opt.csv'

MASS                = 89        # kg
HEADER_LINES        = 29
TRIAL_DURATION      = 180       # s of data each fit sees
STAND_DURATION      = 120       # s
STEADY_STATE_WINDOW = 60        # s at the end of a trial that 'avg' and 'std' use
ESTIMATE_WINDOW     = (240, 360)  # s the fitted curve is averaged over
MAX_RISE_TIME       = 52        # s, upper bound on c
PLOT_TRIAL          = 'walk_opt_8'

COLUMNS = ['time', 'vo2', 'vo2_kg', 'mets', 'vco2', 've', 'rer', 'rr', 'vt', 'feo2', 'feco2', 'ackcal']

# Condition start times as (minutes, seconds) from the start of the recording
STAND = (1, 1)
CONDITIONS = {
    'walk_opt_1':  (7, 59),   'walk_opt_2':  (11, 49),  'walk_opt_3':  (15, 7),
    'walk_opt_4':  (18, 27),  'walk_opt_5':  (49, 38),  'walk_opt_6':  (53, 2),
    'walk_opt_7':  (56, 46),  'walk_opt_8':  (59, 47),  'walk_opt_9':  (63, 5),
    'walk_opt_10': (66, 34),  'walk_opt_11': (69, 46),  'walk_opt_12': (73, 21),
    'walk_opt_13': (91, 42),  'walk_opt_14': (95, 4),   'walk_opt_15': (104, 41),
    'walk_opt_16': (108, 2),  'walk_opt_17': (117, 18), 'walk_opt_18': (120, 36),
    'walk_opt_19': (123, 53), 'walk_opt_20': (127, 23), 'walk_opt_21': (130, 42),
    'walk_opt_22': (134, 0),  'walk_opt_23': (137, 20), 'walk_opt_24': (140, 33),
}


def first_order(t, a, b, c):
    return a + b * np.exp(-t / c)


def read_parvo(path: Path, mass: float):
    """Breath table with time in s, VO2/VCO2 in mL/min, and a Brockway `met` column in W/kg.

    Rows after the data (the 'Max VO2' summary) have no numeric time and are dropped.
    """
    df = pd.read_csv(path, skiprows=HEADER_LINES, header=None, names=COLUMNS, skipinitialspace=True)
    df = df.apply(pd.to_numeric, errors='coerce').dropna(subset=['time']).reset_index(drop=True)
    df['vo2']  *= 1000
    df['vco2'] *= 1000
    df['time'] *= 60
    df['met']   = (0.278 * df['vo2'] + 0.075 * df['vco2']) / mass

    return df


def window(df: pd.DataFrame, start: float, end: float):
    """Rows from the breath nearest `start` through the breath nearest `end`, inclusive."""
    i0 = np.argmin(np.abs(df['time'].to_numpy() - start))
    i1 = np.argmin(np.abs(df['time'].to_numpy() - end))

    return df.iloc[i0:i1 + 1]


def fit_condition(df: pd.DataFrame, start: float):
    """Summary of one condition starting at `start` s.

    Returns a dict with the steady-state mean/std of `met`, the fit parameters (a, b, c),
    the fitted curve's mean over ESTIMATE_WINDOW, and the raw trace.
    """
    end   = start + TRIAL_DURATION
    trial = window(df, start, end)
    ss    = window(df, end - STEADY_STATE_WINDOW, end)

    t, met = trial['time'].to_numpy(), trial['met'].to_numpy()
    t      = t - t[0]
    p0     = [met[-1], met[0] - met[-1], min(60, MAX_RISE_TIME)]
    params, _ = curve_fit(first_order, t, met, p0=p0,
                          bounds=([-np.inf, -np.inf, 0], [np.inf, np.inf, MAX_RISE_TIME]))

    lo, hi   = ESTIMATE_WINDOW
    estimate = quad(first_order, lo, hi, args=tuple(params))[0] / (hi - lo)

    ss_dt = np.diff(ss['time'].to_numpy())
    return {
        'avg':         ss['met'].mean(),
        'std':         ss['met'].std(ddof=1),
        'estimate':    estimate,
        'params':      params,
        'avgssrer':    ss['rer'].mean(),
        'avgweighted': np.sum(ss_dt * ss['met'].to_numpy()[1:]) / np.sum(ss_dt),
        'time':        t,
        'data':        met,
    }


def main():
    df = read_parvo(DATA_PATH, MASS)

    stand_start  = 60 * STAND[0] + STAND[1]
    resting_rate = window(df, stand_start, stand_start + STAND_DURATION)['met'].mean()

    fits = {name: fit_condition(df, 60 * m + s) for name, (m, s) in CONDITIONS.items()}

    results = pd.DataFrame({
        'trial':    list(fits),
        'avg':      [f['avg'] - resting_rate for f in fits.values()],
        'estimate': [f['estimate'] - resting_rate for f in fits.values()],
        'std':      [f['std'] for f in fits.values()],
    })
    results.to_csv(OUTPUT_PATH, index=False)
    print(f'Resting rate: {resting_rate:.4f} W/kg')
    print(results.to_string(index=False))

    fig, ax = plt.subplots()
    ax.plot(df['rer'], '-o')
    ax.set_title('RER')

    fig, ax = plt.subplots()
    ax.bar(results['trial'], results['estimate'])
    ax.tick_params(axis='x', labelrotation=90)
    ax.set_title('MB04 Optimization')

    f = fits[PLOT_TRIAL]
    t = np.linspace(f['time'][0], f['time'][-1], 500)
    fig, ax = plt.subplots()
    ax.scatter(f['time'], f['data'], color='b', label='Raw Data')
    ax.plot(t, first_order(t, *f['params']), 'r', label='1st order fit')
    ax.axhline(f['estimate'], color='k', label='Estimated Steady-State Rate')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Metabolic Rate (W/kg)')
    ax.legend()
    ax.set_ylim((3, None))

    path = 'human_data/MB04/met.svg'
    fig.set_size_inches((8, 3))
    plr.dress_axis(ax)
    fig.savefig(path, transparent=True)
    print(path)


if __name__ == '__main__':
    main()
