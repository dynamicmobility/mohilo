import time
import warnings
from functools import partial
from pathlib import Path
from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
from linear_operator.utils.warnings import NumericalWarning
import torch
from pathlib import Path
import pypolar as plr
from tqdm import tqdm
warnings.filterwarnings('ignore', category=NumericalWarning)

DIM              = 1
BOX              = 5.0
SEED             = 95
TRUE_NOISE       = 0.5
GP_NOISE         = plr.NoiseModel.prior(TRUE_NOISE)
# GP_NOISE         = plr.NoiseModel.pinned(TRUE_NOISE)
MIN_LENGTHSCALE  = 0.1
NUM_QUERIES      = DIM * 13
ACQ_STRATS       = ['ucb', 'logei', 'qlognei']
REPEATS          = 1
COMFORT          = 'Comfort'
METABOLIC        = 'Cost'
MULTITHREAD      = False

RUNS_PER_ACQF = 10

METABOLIC_TRUTH = plr.SyntheticFunction(
    truth = plr.construct_function(
        func = plr.SYNTHETIC_1D_FUNCTIONS['Levy'],
        dim  = DIM,
        box  = BOX,
        seed = SEED
    ),
    rel_noise_std = TRUE_NOISE,
)

COMFORT_TRUTH = plr.SyntheticFunction(
    truth = plr.construct_function(
        func = plr.SYNTHETIC_1D_FUNCTIONS['Levy'],
        dim  = DIM,
        box  = BOX,
        seed = SEED
    ),
    rel_noise_std = TRUE_NOISE,
)
GROUND_TRUTHS = {
    METABOLIC   : METABOLIC_TRUTH,
    COMFORT     : COMFORT_TRUTH
}

def make_probes():
    probes = [
        plr.Probe(
            name              = METABOLIC,
            caller            = METABOLIC_TRUTH,
            obj_name          = METABOLIC,
            separate_thread   = MULTITHREAD
        ),
        plr.Probe(
            name              = COMFORT,
            caller            = COMFORT_TRUTH,
            repeats           = REPEATS,
            obj_name          = COMFORT,
            separate_thread   = MULTITHREAD
        ),
    ]
    return probes


def make_experiment(probes: list[plr.Probe]):
    experiment = plr.Logger(
        objectives   = [
            plr.Objective.from_empty(
                name          = METABOLIC,
                maximize      = False,
                action_bounds = (-BOX, BOX)
            ),
            plr.Objective.from_empty(
                name          = COMFORT,
                maximize      = False,
                action_bounds = (-BOX, BOX)
            )
        ],
        probes       = probes,
        device       = plr.Device(),
        action_names = [f'x{i}' for i in range(DIM)],
    )
    return experiment

def fit_gp(objective):
    return plr.BoTorchGP(
        objective           = objective,
        noise               = GP_NOISE,
        fit_hyperparameters = True,
        min_length_scale    = MIN_LENGTHSCALE,
        # signal_var=1.0,
        # length_scale=0.5
    )

def run_experiment(
    experiment        : plr.Logger,
    acqf              : plr.AcquisitionFunction,
    optimal_actions   : dict[str, np.ndarray],
    ground_truths     : dict[str, plr.SyntheticFunction],
    gt_spread         : dict[str, float]
):
    gp = None
    regrets = []
    for i in tqdm(range(NUM_QUERIES)):
        objective = experiment.objectives[COMFORT]
        if len(objective.ydata) < 1:
            # randomly sample if no data is collected
            action = plr.sample_actions(
                bounds = BOX,
                dim    = DIM,
                n      = 1,
                kind   = 'uniform',
                seed   = SEED + i
            )[0]
        else:
            # fit gp + Acquisition strategy for the rest
            gp = fit_gp(objective)
            degenerate_gp = gp.get_fitted_hyperparameters().signal_var < 1e-2
            action = acqf.query(gp, q=1)[0]
            if degenerate_gp:
                print(action, i)
                print('IT HAPPENED\n\n')

            recommended_action, _, _  = gp.recommend(raw=True)
            comfort_gt                = ground_truths[COMFORT]
            optimal_val               = comfort_gt(optimal_actions[COMFORT], noise=False)
            inferred_val              = comfort_gt(recommended_action, noise=False)
            scaled_regret             = (inferred_val - optimal_val) / gt_spread[COMFORT]
            
            regrets.append(scaled_regret)

        experiment.begin_trial(
            action=action,
            args={
                METABOLIC: (action,),
                COMFORT:   (action, i + 1)
            }
        )
        experiment.wait_for_measurements()
        experiment.end_trial() # updates the objectives

    gp = fit_gp(objective)
    
    return experiment, gp, np.asarray(regrets)

def make_fit_figure(truth, objective, gp, inferred, box, output):
    """The 1D fit: the GP and the truth evaluated on a grid, drawn by
    `plr.plot_fit_1d`.
    """
    output = Path(output)
    grid    = np.linspace(box[0, 0], box[1, 0], 1024)[:, None]
    mu, std = gp.posterior_at(grid, raw=True)
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    plr.plot_fit_1d(
        ax       = ax,
        x        = grid[:, 0],
        mu       = mu[:, 0],
        std      = std[:, 0],
        xdata    = objective.xdata[:, 0],
        ydata    = objective.ydata,
        truth    = plr.truth_at(truth, grid),
        # one objective, so the single column is the path itself
        paths    = gp.sample_paths(grid, 16, raw=True)[:, :, 0],
        vlines   = {'recommended action': inferred[0],
                    'true optimizer'    : truth.optimizers[0, 0].item()},
        band_std = 1.0,
        title    = f'{'f'} on Levy, 1D, after {len(objective.ydata)} measurements'
    )

    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200)
    print(f'wrote {output}')

def setup_experiment(acq_strat, seed):
    probes     = make_probes()
    experiment = make_experiment(probes)
    acqf       = plr.AcquisitionFunction(
        acqf      = plr.acquisition_factory_1d(strategy=acq_strat, seed=seed),
        objective = experiment.objectives[COMFORT],
    )

    return experiment, acqf


def get_groundtruth_optimum(
    seed,
    num_samples=4096
):
    actions = plr.sample_actions(
        bounds    = BOX,
        dim       = DIM,
        n         = num_samples,
        kind      = 'sobol',
        seed      = seed,
    )
    gt_values = np.array([GROUND_TRUTHS[obj_name](actions, noise=False) for obj_name in GROUND_TRUTHS.keys()])
    idxs = np.argmin(gt_values, axis=1)
    spread = gt_values.max(axis=1) - gt_values.min(axis=1)
    return (
        {name: actions[i] for name, i in zip(GROUND_TRUTHS.keys(), idxs, strict=True)},
        {name: sp for name, sp in zip(GROUND_TRUTHS.keys(), spread, strict=True)}
    )

@dataclass
class ExperimentDataset:
    pass
    # the model at each iteration, use model.state_dict()!!
    # the feedback points and actions (this might already be included in the model state dict?) --> construct an objective from this?
    # where the sampled actions came from (acquisition or random) list[str]
    # the acquistion function and its inputs (str + kwargs [str, float])
    # the regret list[float]
    # the groundtruth functions and their inputs list[str] of func names and list[kwargs (str, float)] of inputs 

def main():
    torch.manual_seed(SEED)
    optimal_actions, gt_spread = get_groundtruth_optimum(
        seed = SEED,
        num_samples=8192
    )
    
    data = []
    fig, ax = plt.subplots()
    for acq_strat in ACQ_STRATS:
        experiment, acqf = setup_experiment(
            acq_strat   = acq_strat,
            seed        = SEED
        )
        experiment, gp, regrets = run_experiment(
            experiment        = experiment,
            acqf              = acqf,
            optimal_actions   = optimal_actions,
            ground_truths     = GROUND_TRUTHS,
            gt_spread         = gt_spread
        )
        data.append(regrets)
        ax.plot(regrets, label=acq_strat)

        if DIM == 1:
            box = np.array([[-BOX] * DIM, [BOX] * DIM], dtype=float)
            make_fit_figure( # TODO: make gif version of this
                truth       = COMFORT_TRUTH.truth,
                objective   = experiment.objectives[COMFORT],
                gp          = gp,
                inferred    = gp.recommend()[0],
                box         = box,
                output      = f'hilo/output/{acq_strat}-fit.svg'
            )
            # TODO: find out way to save data including gp model + fit
            # so we can plot it later? maybe save every iteration gp...
    fig.legend()
    plt.show()



    best, mu, std = gp.best_actions(raw=True)
    # (2, DIM) of [lower; upper] rows
    

    print('here', best)
    print('here2', COMFORT_TRUTH(best, noise=False))
    actions = plr.sample_actions(
        bounds = [[-BOX]*DIM, [BOX]*DIM],
        n=1024,
        kind='sobol',
        seed=SEED
    )
    values = COMFORT_TRUTH([actions], noise=False)
    print(values.max() - values.min())
    print(values.min())
    print(actions[np.argmin(values)])


if __name__ == '__main__':
    main()
