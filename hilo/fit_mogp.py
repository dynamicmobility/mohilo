import time
from pathlib import Path
import socketio
from dataclasses import asdict
import numpy as np
import torch
import pypolar as plr
from tablet.survey import Survey
import sys
import logging
import hilo.shared.log as log

logger = logging.getLogger(__name__)

EXO_IP      = "192.168.1.122:5000"
SUBJECT     = 'MT01'
CONNECT     = True
EMULATE     = False

if EMULATE:
    import hilo.shared.simulation as hilo
else:
    import hilo.shared.hardware as hilo

sio = socketio.Client()

if CONNECT:
    i = 0
    while True:
        try:
            i += 1
            sio.connect(f"ws://{EXO_IP}")
            break
        except socketio.exceptions.ConnectionError:
            # prints rather than logs: this runs at import, before setup_logger
            if i > 5:
                print('Exceeded maximum number of retries. Exiting...')
                exit()
            print('Connection failed. Retrying...')
            time.sleep(1)
    else:
        sio = None

def send_to_exo(action):
    logger.info(f'Exo got action {action}')
    ans = log.logged_input(f'Send {action} (y/n)? ')
    while True and ans.lower() != 'y':
        action = log.logged_input(f'Enter an alternative action as an array, like [1, 2, 3]: ')
        try:
            action = np.array(eval(action))
            if np.any(action < hilo.BOUNDS[0]) or np.any(action > hilo.BOUNDS[1]):
                raise ValueError('Action out of bounds')
            logger.info(f'Got {action}. Sending to exo...', extra=log.TO_BOTH)
            break
        except ValueError as e:
            logger.error(f'Action out of bounds. Note that bounds (low, high) = {hilo.BOUNDS}', extra=log.TO_BOTH)
            continue
        except Exception as e:
            logger.error(e, extra=log.TO_BOTH)
            logger.error(f'{action} did not compile. Try again.', extra=log.TO_BOTH)
            continue
        
    if CONNECT:
        action_dict = {
            'h_flex_torque_scale': action[0], # make this a dict when sending to the exo
            'h_ext_torque_scale' : action[1],
            'hip_delay_idx'      : action[2]
        }
        sio.emit("update_inputs", action_dict)
        logger.info('Successfully sent action.', extra=log.TO_BOTH)
        return True
    else:
        logger.info('Disabled!', extra=log.TO_BOTH)
        return True


def fit_gp(
    objective   : plr.DecoupledObjectives,
    noise       : plr.NoiseModel = None,
    hypers      : plr.GPHyperparameters = None
):
    return plr.DecoupledMOGP(
        objectives          = objective,
        noise               = hilo.GP_NOISE,
        fit_hyperparameters = True,
        min_length_scale    = hilo.MIN_LENGTHSCALE,
    )

def run_experiment(
    experiment        : plr.Logger,
    acqf              : plr.AcquisitionFunction,
    dataset           : plr.ExperimentDataset,
):
    gp = None
    for i in range(hilo.NUM_QUERIES):
        if i < hilo.NUM_RANDOM:
            # randomly sample if no data is collected
            source = 'random'
            action = plr.sample_actions(
                bounds = hilo.BOUNDS,
                n      = 1,
                kind   = 'uniform',
                seed   = hilo.SEED + i,
                dim    = hilo.DIM
            )[0]
        else:
            # fit gp + Acquisition strategy for the rest
            source = dataset.acquisition.strategy
            action = acqf.query(gp, q=1)[0]

        experiment.begin_trial(
            action         = action,
            device_send_fn = send_to_exo,
            args           = {
                hilo.METABOLIC: (action, i + 1),
                hilo.COMFORT:   (action, i + 1, hilo.SURVEY_TIMEOUT, hilo.SURVEY_PERIOD)
            }
        )
        
        experiment.wait_for_measurements()
        experiment.end_trial() # updates the objectives
        
        gp = fit_gp(experiment.objectives)
        dataset.add_trial(
            objectives  = experiment.objectives,
            gp          = gp,
            action      = action,
            source      = source,
        )
        logger.info(f'Wrote {dataset.save()}')
    
    # the run's final state: every measurement, and the fit to all of them
    gp = fit_gp(experiment.objectives)
    dataset.add_trial(
        objectives  = experiment.objectives,
        gp          = gp,
    )
    
    return experiment, dataset

def connect_to_ipad():
    if EMULATE:
        return None
    ipad = Survey(
        timeout = hilo.SURVEY_TIMEOUT,
        period  = hilo.SURVEY_PERIOD,
        logger  = logger
    )
    logger.info('Waiting for the iPad...', extra=log.TO_BOTH)
    ipad.wait_for_ipad()
    return ipad

def setup_experiment(ipad):
    probes     = hilo.make_probes_mo(ipad, multithread=hilo.MULTITHREAD)
    experiment = hilo.make_experiment_mo(
        probes    = probes,
        maximize  = hilo.MAXIMIZE
    )
    acqf_params = plr.AcquisitionParams(
        strategy          = hilo.ACQ_STRAT,
        seed              = hilo.SEED,
        num_objectives    = hilo.NUM_OBJECTIVES,
        raw_ref_point     = hilo.REF_POINT,
        **hilo.ACQ_KWARGS
    )
    return experiment, acqf_params

def main():
    torch.manual_seed(hilo.SEED)
    hilo.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    log.setup_logger(hilo.OUTPUT_DIR / f'{SUBJECT}.log')
    
    ipad                    = connect_to_ipad()
    experiment, acqf_params = setup_experiment(ipad)
    
    # dataset = plr.ExperimentDataset(
    #     name         = SUBJECT, # TODO: add a date here
    #     acquisition  = acqf_params,
    #     config       = asdict(
    #         hilo.Config(
    #             gp_noise          = plr.NoiseModel.coerce(hilo.GP_NOISE),
    #             min_lengthscale   = hilo.MIN_LENGTHSCALE,
    #             num_queries       = hilo.NUM_QUERIES,
    #             repeats           = hilo.REPEATS,
    #             dim               = hilo.DIM,
    #         )
    #     ),
    #     groundtruth  = hilo.GROUND_TRUTH_PARAMS if EMULATE else None,
    #     path         = hilo.OUTPUT_DIR / (SUBJECT + f'.json')
    # )
    
    dataset = plr.ExperimentDataset.load('hilo/output/experiments/20260827_154607/MT01.json')

    try:
        experiment, dataset = run_experiment(
            experiment        = experiment,
            acqf              = acqf_params.build(),
            dataset           = dataset,
        )
    finally:
        logger.info(f'Wrote {dataset.save()}') # change this to save per trial...
        if CONNECT: ipad.close()


if __name__ == '__main__':
    main()
