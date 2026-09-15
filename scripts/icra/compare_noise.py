from scripts.icra import comparison as compare
import argparse
import time
import warnings
from pathlib import Path
import pandas as pd
import numpy as np
import torch
from linear_operator.utils.warnings import NumericalWarning
import pypolar as plr
from tqdm import tqdm
warnings.filterwarnings('ignore', category=NumericalWarning)

OUTPUT_DIR    = Path('scripts/output/experiments') / time.strftime('%Y%m%d_%H%M%S')

def parse_args():
    p = argparse.ArgumentParser(description='NSGA-II and MOBO across observation '
                                            'noise levels, one subdirectory each')
    p.add_argument('--noise', type=float, nargs='+', required=True,
                   help='true noise levels, as fractions of each objective\'s '
                        'spread, e.g. --noise 0.01 0.1 0.2')
    p.add_argument('--output', type=Path, default=OUTPUT_DIR,
                   help='the ablation directory to write noise<level>/ into; '
                        'defaults to a new timestamped one')

    return p.parse_args()


def main():
    args = parse_args()
    SAVE_DATA = True
    if not SAVE_DATA:
        print('WARNING NOT SAVING DATA')
    if SAVE_DATA: args.output.mkdir(parents=True, exist_ok=True)
    DIM = 3
    NUM_OBJS = 2
    BOX = (-3,3)
    BOUNDS = plr.as_bounds(BOX, dim=DIM)
    TRUE_NOISE = 0.1
    GUESS_FRAC = 0.3
    SEED = 95
    LOW = (-1,) * NUM_OBJS
    HIGH = (1,) * NUM_OBJS
    MAXIMIZE = False
    NUM_QUERIES = 100
    NUM_RANDOM = 3
    ACTION_NAMES = [f'$x_{i}$' for i in range(DIM)]
    MIN_LENGTHSCALE = 0.1
    POP_SIZE      = 10
    N_TRIALS      = 50
    rng = np.random.default_rng(SEED)
    
    for noise_level in args.noise:
        run_dir = args.output / f'noise{noise_level}'
        run_dir.mkdir(parents=True, exist_ok=True)
        for i in tqdm(range(N_TRIALS)):
            TRUE_NOISE = noise_level
            torch.manual_seed(SEED + i)
            NOISE_ESTIMATE = np.clip(TRUE_NOISE + rng.random() * GUESS_FRAC * TRUE_NOISE, 1e-3, None)
            GP_NOISE   = plr.NoiseModel.prior(NOISE_ESTIMATE)
            gt, params = compare.make_idealpoint_groundtruth(
                seed     = SEED + i,
                box      = BOX,
                noise    = TRUE_NOISE,
                dim      = DIM,
                low      = LOW,
                high     = HIGH,
                num_objs = NUM_OBJS
            )

            evals, curves, queried, measured = compare.run_nsga_trial(
                seed        = SEED + i,
                pop_size    = POP_SIZE,
                groundtruth = gt,
                bounds      = BOUNDS,
                maximize    = MAXIMIZE,
                num_objs    = NUM_OBJS,
                num_queries = 900 #NUM_QUERIES,
            )

            curves['evals'] = evals

            dense, sparse = compare.run_mobo_trial(
                seed        = SEED + i,
                groundtruth = gt,
                gt_params   = params,
                num_objs=NUM_OBJS,
                bounds=BOUNDS,
                maximize=MAXIMIZE,
                num_queries=NUM_QUERIES,
                gp_noise=GP_NOISE,
                action_names=ACTION_NAMES,
                min_lengthscale=MIN_LENGTHSCALE,
                num_random=NUM_RANDOM,
                dim=DIM,
                output_dir=run_dir
            )
            if SAVE_DATA:
                compare.save_trial(run_dir, i, curves, queried, measured,
                                   dense, sparse)


if __name__ == '__main__':
    main()
