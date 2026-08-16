import pypolar as plr

DIM = 3
SEED = 95
BOX = 5.0

def main():
    metabolic_cart = plr.SyntheticFunction(
        truth         = plr.construct_function(
            func    = plr.SYNTHETIC_1D_FUNCTIONS['Levy'],
            dim     = DIM,
            box     = BOX,
            seed    = SEED
        ),
        rel_noise_std = 0.5,
        seed          = SEED
    )
    
    comfort_scale = plr.SyntheticFunction(
        truth         = plr.construct_function(
            func    = plr.SYNTHETIC_1D_FUNCTIONS['DixonPrice'],
            dim     = DIM,
            box     = BOX,
            seed    = SEED
        ),
        rel_noise_std = 0.3,
        seed          = SEED
    )
    

    metabolics = plr.SyntheticProbe(
        functions={
            'metabolic cost'    : metabolic_cart,
        },
        repeats   = 1,
        duration  = 1.0
    )
    speed = plr.SyntheticProbe(
        functions={
            'comfort scale': comfort_scale,
        },
        duration = 0.5

    )
    
    ledger = plr.Ledger(
        path = 'test_dir/',
        config = {},
    )

if __name__ == '__main__':
    main()
    
    
import numpy as np
import pypolar as plr

LOG, CONFIG = '/tmp/ledger_demo.jsonl', {'demo': 1, 'box': 5.0}
probe = plr.SyntheticProbe({'cost': lambda X: np.sum(X**2, axis=1)}, repeats=2)

with plr.Ledger(LOG, CONFIG) as led:
    for action in ([1., 1.], [2., -1.]):
        led.open_trial(action, source='manual')
        led.close_trial(probe.measure(action, record=led.record))

    led.open_trial([3., 3.], source='manual')      # opened, never closed: a "crash"
    led.record('sample', objective='cost', value=17.9)

obj = plr.Objective.from_empty('cost', maximize=False, action_bounds=(-5., 5.))
with plr.Ledger(LOG, CONFIG) as led:
    print('interrupted trials:', led.replay(obj))
print('actions:\n', obj.xdata, '\nvalues: ', obj.ydata)