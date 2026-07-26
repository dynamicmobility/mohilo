import pypolar as plr
import config.base as config


def _make_sampler(cfg_sampler, gp, rng):
    """Build the sampler named by a sampling config."""
    if isinstance(cfg_sampler, config.RandomSampling):
        return plr.RandomSampler(rng)
    if isinstance(cfg_sampler, config.ThompsonSampling):
        return plr.ThompsonSampler(gp, rng)
    if isinstance(cfg_sampler, config.ExpectedImprovement):
        return plr.ExpectedImprovementSampler(gp, rng, xi=cfg_sampler.xi)
    if isinstance(cfg_sampler, config.KnowledgeGradient):
        return plr.KnowledgeGradientSampler(
            gp, rng, num_candidates=cfg_sampler.num_candidates
        )
    if isinstance(cfg_sampler, config.MaxValueEntropy):
        return plr.MaxValueEntropySampler(gp, rng, num_maxima=cfg_sampler.num_maxima)
    raise ValueError(f'{type(cfg_sampler).__name__} is not a valid sampling config')


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
    # sampler = plr.UniformSampler(n=40, rng=rng)

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


def create_1d_sim(
    rng, cfg: config.HILO
) -> tuple[
    plr.Regression, 
    plr.ConjugateGP,
    plr.RandomSampler | plr.ThompsonSampler | plr.AcquisitionSampler,
    plr.BoundedIdealPoint, 
    plr.NoisyRegressionOracle
]:
    """
    Single-objective problem with one objective: Metabolic Cost of Transport
    (MCT) .
    """

    # Regression optimization problem
    regression = plr.Regression(
        low           = cfg.problem.action_low,
        high          = cfg.problem.action_high,
        action_dims   = cfg.problem.action_dims,
        precision     = cfg.problem.precision
    )

    # Solver
    if cfg.optimizer.gptype == 'LaplaceGP':
        optimizer = plr.LaplaceGP(
            kernel           = cfg.optimizer.kernel,
            signal_variance  = cfg.optimizer.signal_variance,
            length_scale     = cfg.optimizer.length_scale,
            x0_init_method   = cfg.optimizer.x0_init_method,
            rng              = rng
        )
    elif cfg.optimizer.gptype == 'ConjugateGP':
        optimizer = plr.ConjugateGP(
            kernel           = cfg.optimizer.kernel,
            signal_variance  = cfg.optimizer.signal_variance,
            length_scale     = cfg.optimizer.length_scale,
            # x0_init_method   = cfg.optimizer.x0_init_method,
            rng              = rng
        )
    else:
        raise Exception(f'{cfg.optimizer.gptype} is not a valid GP class in pyPolar')

    # Sampler/Acquisition function
    sampler = _make_sampler(cfg.sampler, optimizer, rng)

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