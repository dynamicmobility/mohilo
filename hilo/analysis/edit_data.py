"""Edits a recorded run: drops trials, refits every model, writes a new file.

The models a run recorded are the fits it made live, under the hyperparameters
it was configured with. This is how a run is read back under different ones --
and it is how a run edited by `delete_trial` gets its models back, since that
clears every fit conditioned on a measurement it removed.

    python -m hilo.analysis.edit_data
"""

import numpy as np

import pypolar as plr

SOURCE = 'human_data/MB03/MB03.json'
DEST   = 'human_data/MB03/MB03_edited.json'

DELETE = [12]   # trials to drop, in the source's own numbering

# ---- the GP every trial is refit under --------------------------------------

FIT_HYPERPARAMETERS = True                   # fit by marginal likelihood
NOISE               = plr.NoiseModel.prior(0.5)
MIN_LENGTHSCALE     = 0.1                   # a bound, fitted or not
LENGTH_SCALE        = 0.2                    # start when fitting, value when not
SIGNAL_VAR          = 1.0                    # same


def fit_gp(objectives):
    """One trial's GP. `min_length_scale` bounds the fitted lengthscales, so it
    holds whether or not `FIT_HYPERPARAMETERS` is set; `length_scale` and
    `signal_var` are starting values under a fit and fixed values without one.
    """
    return plr.DecoupledMOGP(
        objectives          = objectives,
        noise               = NOISE,
        fit_hyperparameters = FIT_HYPERPARAMETERS,
        length_scale        = LENGTH_SCALE,
        signal_var          = SIGNAL_VAR,
        min_length_scale    = MIN_LENGTHSCALE,
    )


def main():
    dataset = plr.ExperimentDataset.load(SOURCE)
    print(dataset)

    # descending, so an earlier deletion does not renumber a later one
    for trial in sorted(DELETE, reverse=True):
        dataset = dataset.delete_trial(trial)

    dataset = dataset.refit(fit_gp)
    refit   = sum(trial.state_dict is not None for trial in dataset)
    print(f'\nRefit {refit} of {len(dataset)} trials, wrote {dataset.save(DEST)}')

    # the last trial's fit, which is the one to every measurement kept
    model = dataset.get_model(-1)
    if model is not None:
        for name, hypers in zip(dataset.get_objectives(-1).names,
                                model.get_fitted_hyperparameters()):
            print(f'  {name:<15}: lengthscale={np.round(hypers.lengthscale, 3)} '
                  f'signal_var={hypers.signal_var:.4g} '
                  f'noise_var={hypers.noise_var:.4g}')


if __name__ == '__main__':
    main()
