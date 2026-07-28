from config.base import (
    MultiObjectiveGaussianProcess,
    MultiObjectiveRegression,
    NoisyRegressionOracle,
    BoundedIdealPoint,
    DSTS,
    QNEHVI,
    MOHILO,
    MOHILOStudy
)
import numpy as np

hipexo_sim_idealized = MOHILO(
    problem = MultiObjectiveRegression(
        action_low    = np.array([0.0]),
        action_high   = np.array([4.0]),
        action_dims   = np.array([100]),
        precisions    = np.array([1e1, 1e1]),
    ),
    optimizer = MultiObjectiveGaussianProcess(
        kernels             = ['squared_exp', 'squared_exp'],
        signal_variances    = [10.0, 10.0],
        length_scales       = [1.0, 1.0],
        x0_init_methods     = ['random', 'random'],
    ),
    # sampler = DSTS(
    #     rho = 0.01
    # ),
    sampler = QNEHVI(
        num_samples = 128
    ),
    objective = BoundedIdealPoint(
        w             = np.array([[2.0], [1.0]]),
        delta         = np.array([1.0, 1.0]),
        gamma         = np.array([0.0, 0.0]),
        lower_bound   = np.array([0.0, 0.0]),
        upper_bound   = np.array([8.0, 8.0])
    ),
    oracle = NoisyRegressionOracle(
        noise_std = np.array([1.0, 1.0]),
    ),
    num_objs = 2,
    save_dir = 'hilo/output'
)

hipexo_sim_idealized_2d = MOHILO(
    problem = MultiObjectiveRegression(
        action_low    = np.array([0.0, 0.0]),
        action_high   = np.array([4.0, 4.0]),
        action_dims   = np.array([20, 20]),
        precisions    = np.array([1e1, 1e1]),
    ),
    optimizer = MultiObjectiveGaussianProcess(
        kernels             = ['squared_exp', 'squared_exp'],
        signal_variances    = [10.0, 10.0],
        length_scales       = [1.0, 1.0],
        x0_init_methods     = ['random', 'random'],
    ),
    sampler = DSTS(
        rho = 0.01
    ),
    objective = BoundedIdealPoint(
        w             = np.array([[2.0, 2.0], [1.0, 1.0]]),
        delta         = np.array([1.0, 1.0]),
        gamma         = np.array([0.0, 0.0]),
        lower_bound   = np.array([0.0, 0.0]),
        upper_bound   = np.array([8.0, 8.0])
    ),
    oracle = NoisyRegressionOracle(
        noise_std = np.array([0.0, 0.0]),
    ),
    num_objs = 2,
    save_dir = 'hilo/output'
)


hipexo_sim_idealized_3d = MOHILO(
    problem = MultiObjectiveRegression(
        action_low    = np.array([0.0, 0.0, 0.0]),
        action_high   = np.array([4.0, 4.0, 4.0]),
        action_dims   = np.array([12, 12, 12]),
        precisions    = np.array([1e1, 1e1, 1e1]),
    ),
    optimizer = MultiObjectiveGaussianProcess(
        kernels             = ['squared_exp', 'squared_exp', 'squared_exp'],
        signal_variances    = [10.0, 10.0, 10.0],
        length_scales       = [1.0, 1.0, 1.0],
        x0_init_methods     = ['random', 'random', 'random'],
    ),
    sampler = DSTS(
        rho = 0.01,
    ),
    objective = BoundedIdealPoint(
        w             = np.array([[2.0, 2.0, 2.0], [1.0, 1.0, 1.0]]),
        delta         = np.array([1.0, 1.0]),
        gamma         = np.array([0.0, 0.0]),
        lower_bound   = np.array([0.0, 0.0]),
        upper_bound   = np.array([8.0, 8.0]),
    ),
    oracle = NoisyRegressionOracle(
        noise_std = np.array([0.0, 0.0]),
    ),
    num_objs = 2,
    save_dir = 'hilo/output',
)

hipexo_pilot = MOHILOStudy(
    problem = MultiObjectiveRegression(
        action_low    = np.array([0.0, 0.0, 0.0]),
        action_high   = np.array([0.5, 0.5, 0.5]),
        action_dims   = np.array([30, 30, 30]),
        precisions    = np.array([1e1, 1e1, 1e1]),
    ),
    optimizer = MultiObjectiveGaussianProcess(
        kernels             = ['squared_exp', 'squared_exp', 'squared_exp'],
        signal_variances    = [10.0, 10.0, 10.0],
        length_scales       = [1.0, 1.0, 1.0],
        x0_init_methods     = ['random', 'random', 'random'],
    ),
    sampler = DSTS(
        rho = 0.01,
    ),
    num_objs = 2,
    save_dir = 'hilo/output',
)