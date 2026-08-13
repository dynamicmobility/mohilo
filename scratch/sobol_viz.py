
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import qmc

import pypolar as plr

LOW             = np.array([-2.0, -2.0])
HIGH            = np.array([ 2.0,  2.0])
SEED            = 95
LOG2_INITIAL    = 3 # 2^3 = 8 initial design points
LOG2_CANDIDATES = 4 # 2^9 = 512 candidates per iteration


def sobol(log2_n, rng):
    engine = qmc.Sobol(d=len(LOW), scramble=True, seed=rng)
    return qmc.scale(engine.random_base2(log2_n), LOW, HIGH)

if __name__ == '__main__':
    
    pts = sobol(log2_n=LOG2_CANDIDATES, rng=SEED)
    fig, ax = plt.subplots()
    ax.scatter(pts[:, 0], pts[:, 1])
    plt.show()