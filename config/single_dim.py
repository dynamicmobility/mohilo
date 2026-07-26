from config.base import (
    GaussianProcess,
    Regression,
    NoisyRegressionOracle,
    BoundedIdealPoint,
    ThompsonSampling,
    RandomSampling,
    HILO
)
import numpy as np

sim_1d = HILO(
    problem = Regression(
        action_low    = np.array([0.0]),
        action_high   = np.array([4.0]),
        action_dims   = np.array([200]),
        precision     = np.array([0.1]),
    ),
    optimizer = GaussianProcess(
        kernel             = 'squared_exp',
        signal_variance    = 10.0,
        length_scale       = 1.0,
        x0_init_method     = 'random',
    ),
    sampler = RandomSampling(),
    objective = BoundedIdealPoint(
        w             = np.array([2.0]),
        delta         = -1.0,
        gamma         = 0.0,
        lower_bound   = 0.0,
        upper_bound   = 8.0
    ),
    oracle = NoisyRegressionOracle(
        noise_std = 0.3
    ),
    save_dir = 'hilo/output'
)