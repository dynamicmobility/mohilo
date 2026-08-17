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
    