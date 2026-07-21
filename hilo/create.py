import pypolar as plr
import config.base as config


def create_hipexo_sim(rng, cfg: config.MOHILO):
    """
    Multi-objective problem with two objectives: Metabolic Cost of Transport
    (MCT) and Speed (SP) .
    """

    # Regression optimization problem
    regression = plr.MultiObjectiveRegression(
        low           = cfg.problem.action_low,
        high          = cfg.problem.action_high,
        action_dims   = cfg.problem.action_dims,
        precisions    = cfg.problem.precisions,
        num_objs      = cfg.num_objs
    )

    # Solver
    optimizer = plr.MultiObjectiveGP(
        num_objs          = cfg.num_objs,
        kernels           = cfg.optimizer.kernels,
        signal_variances  = cfg.optimizer.signal_variances,
        length_scales     = cfg.optimizer.length_scales,
        x0_init_methods   = cfg.optimizer.x0_init_methods,
        rng               = rng
    )

    # Sampler/Acquisition function
    sampler = plr.DSTSampler(gps=optimizer.gps, rng=rng, rho=cfg.sampler.rho)
    # sampler = plr.RandomSampler(rng)
    # sampler = plr.UniformSampler(n=20, rng=rng)

    # Groundtruth objectives
    groundtruth = plr.BoundedIdealPoint(
        w             = cfg.objective.w,
        delta         = cfg.objective.delta,
        gamma         = cfg.objective.gamma,
        lower_bound   = cfg.objective.lower_bound,
        upper_bound   = cfg.objective.upper_bound
    )

    # Simulated oracle with groundtruth
    oracle = plr.NoisyRegressionOracle(
        reward_fn   = groundtruth,
        noise_std   = cfg.oracle.noise_std,
        rng         = rng
    )

    return regression, optimizer, sampler, groundtruth, oracle
