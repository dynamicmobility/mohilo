"""Two trials end to end, against a fake device and two fake instruments.
"""

import time
import numpy as np
import pypolar as plr


class Exo(plr.Device):

    def send(self, action):
        input('Send action?')
        print('EXO GOT ACTION', action)


def get_data_from_cart():
    time.sleep(2.0)
    return np.random.normal(3.0, 0.1)


def ask_subject():
    time.sleep(0.2)
    return np.random.randint(1, 6)


def main():
    probes = [
        plr.Probe(name='Metabolic Cart', caller=get_data_from_cart, obj_name='cost'),
        plr.Probe(name='Survey', caller=ask_subject, repeats=4, obj_name='comfort'),
    ]

    experiment = plr.Logger(
        objectives   = [
            plr.Objective.from_empty(name='cost',    maximize=False),
            plr.Objective.from_empty(name='comfort', maximize=True)
        ],
        probes       = probes,
        device       = Exo(),
        action_names = ['x', 'y', 'z']
    )

    for action in [np.zeros(3), np.ones(3)]:
        experiment.begin_trial(action)
        start = time.time()
        while not experiment.all_measurements_completed:
            print('Waiting...', round(time.time() - start, 1))
            time.sleep(0.1)

        experiment.end_trial()

    for name in experiment.objectives.names:
        objective = experiment.objectives[name]
        print(name, 'values ', objective.ydata)
        print(name, 'actions', objective.xdata.tolist())

    print('done')


if __name__ == '__main__':
    main()
