import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUTPUT_DIR          = Path('hilo/output/')
COLUMNS             = ['time', 'vo2', 'vo2_kg', 'mets', 'vco2', 've', 'rer', 'rr',
                       'vt', 'feo2', 'feco2', 'ackcal']
HEADER_LINES        = 29

# Condition start times in seconds from the start of the recording.
STAND_START         = 6 * 60
WALK_STARTS         = {
    'walk_val_1': 13 * 60 + 21,
    'walk_val_2': 18 * 60 + 47,
    'walk_val_3': 24 * 60 + 23,
    'walk_val_4': 29 * 60 + 54,
    'walk_val_5': 35 * 60 + 21,
    'walk_val_6': 40 * 60 + 50,
}
TRIAL_DURATION      = 180 #300   # s
STAND_DURATION      = 120   # s
STEADY_STATE_WINDOW = 60    # s


def read_parvo(path: Path, mass: float) -> pd.DataFrame:
    """
    Breath-by-breath rows of a Parvo CSV, with time in s, vo2/vco2 in mL/min, and
    `met` the Brockway metabolic power in W/kg.
    """
    df = pd.read_csv(path, skiprows=HEADER_LINES, header=None, names=COLUMNS,
                     on_bad_lines='skip')
    df = df.apply(pd.to_numeric, errors='coerce').dropna().reset_index(drop=True)
    df['vo2']  *= 1000
    df['vco2'] *= 1000
    df['time'] *= 60
    df['met']   = (0.278 * df.vo2 + 0.075 * df.vco2) / mass
    return df


def window(df: pd.DataFrame, start: float, end: float) -> pd.DataFrame:
    """Rows from the breath nearest `start` through the breath nearest `end`, inclusive."""
    i0 = np.argmin(np.abs(df.time - start))
    i1 = np.argmin(np.abs(df.time - end))
    return df.iloc[i0:i1 + 1]


def process(df: pd.DataFrame) -> pd.DataFrame:
    """
    Net metabolic cost per walk: the mean `met` over the last STEADY_STATE_WINDOW s
    of the trial minus the standing mean. `std` is the sample SD of the steady-state
    `met`, not net of standing.
    """
    resting = window(df, STAND_START, STAND_START + STAND_DURATION).met.mean()
    rows = []
    for name, start in WALK_STARTS.items():
        end = start + TRIAL_DURATION
        ss = window(df, end - STEADY_STATE_WINDOW, end).met
        rows.append({'trial': name, 'avg': ss.mean() - resting, 'std': ss.std()})
    return pd.DataFrame(rows)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        'csv',
        type    = Path,
        help    = 'the Parvo breath-by-breath CSV'
    )
    p.add_argument(
        '--mass',
        type     = float,
        required = True,
        help     = 'subject mass in kg'
    )

    return p.parse_args()


def main():
    args    = parse_args()
    subject = args.csv.parent.name
    df      = read_parvo(args.csv, args.mass)
    results = process(df)
    print(results.to_string(index=False))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT_DIR / f'{subject}_parvo_results.csv', index=False)

    fig, ax = plt.subplots()
    ax.plot(df.rer.to_numpy(), '-o')
    ax.set_title('RER')
    fig.savefig(OUTPUT_DIR / f'{subject}_parvo_rer.png', dpi=300)

    fig, ax = plt.subplots()
    ax.bar(results.trial, results.avg)
    ax.errorbar(results.trial, results.avg, results['std'], linestyle='none',
                linewidth=1.5, color='k')
    ax.set_title(f'{subject} Validation')
    ax.set_ylabel('Net metabolic cost (W/kg)')
    fig.savefig(OUTPUT_DIR / f'{subject}_parvo_validation.png', dpi=300)


if __name__ == '__main__':
    main()
