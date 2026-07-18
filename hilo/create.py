import pypolar as plr
import numpy as np

def create_hipexo_sim(rng):
    """
    Multi-objective problem with two objectives: Metabolic Cost of Transport
    (MCT) and Speed (SP) .
    """
    # Regression optimization problem
    regression = plr.MultiObjectiveRegression(
        low           = np.array([0.0]),
        high          = np.array([4.0]),
        action_dims   = np.array([100]),
        precisions    = np.array([1e1, 1e1]),
        num_objs      = 2
    )

    # Solver
    optimizer = plr.MultiObjectiveGP(
        num_objs          = 2,
        kernels           = ['squared_exp', 'squared_exp'],
        signal_variances  = [10.0, 10.0],
        length_scales     = [1.0, 1.0],
        x0_init_method    = ['random', 'random'],
        rng               = rng
    )

    # Sampler/Acquisition function
    sampler = plr.DSTSampler(gps=optimizer.gps, rng=rng, rho=0.05)

    # Groundtruth objectives
    groundtruth = plr.BoundedIdealPoint(
        w             = np.array([[2.0], [1.0]]),
        delta         = np.array([1.0, 1.0]),
        gamma         = np.array([0.0, 0.0]),
        lower_bound   = np.array([0.0, 0.0]),
        upper_bound   = np.array([8.0, 8.0])
    )

    # Simulated oracle with groundtruth
    oracle = plr.NoisyRegressionOracle(
        reward_fn   = groundtruth,
        noise_std   = np.array([0.0, 0.0]),
        rng         = rng
    )


    return regression, optimizer, sampler, groundtruth, oracle