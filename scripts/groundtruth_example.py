import pypolar as plr


oracle = plr.MOSyntheticOracle.from_name(
    func              = 'DTLZ2',
    dim               = 3,
    box               = 5,
    num_objectives    = 2,
    measure           = 'range',
    rel_noise_std     = 0.0
)
objs = plr.DecoupledObjectives.from_empty()

objs.add_objective(plr.Objective.from_empty(
    name        = 'f1',
    maximize    = False
))

objs.add_objective(plr.Objective.from_empty(
    name        = 'f2',
    maximize    = False
))

scan_actions = plr.sample_actions(
    bounds    = oracle.bounds,
    n         = 1024,
    kind      = 'sobol',
    seed      = 95
)

Y = oracle(scan_actions)

random_guess = plr.sample_actions(
    bounds    = oracle.bounds,
    n         = 1024,
    kind      = 'sobol',
    seed      = 95
)

# plug in objectives.ytransform(Y) into the metric calc