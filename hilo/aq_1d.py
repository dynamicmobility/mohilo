import os
os.environ["JAX_PLATFORMS"] = "cpu"
import pypolar as plr
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from hilo.create import create_1d_sim
from config.single_dim import sim_1d

def main():
    rng = np.random.default_rng(95)

    # config = hipexo_sim_idealized_2d
    config = sim_1d
    regression, optimizer, sampler, groundtruth, oracle = create_1d_sim(
        rng,
        cfg=config
    )
    
    # Run simulation
    NUM_QUERIES = 200
    for i in tqdm(range(NUM_QUERIES)):
        # Sample an action
        sample_action = sampler.sample(regression.action_space)
        
        # Measure the human performance
        mct_hat = oracle.query(sample_action)
        
        regression.add_feedback(sample_action, mct_hat[0])
        optimizer.set_data(
            action_space    = regression.action_space,
            idx             = regression.feedback_data[:, 0],
            y               = regression.feedback_data[:, 1],
            precision       = regression.precision
        )
        optimizer.fit()
        sampler.update_posterior()
    


    fig, ax = plt.subplots(figsize=(7,5))
    plr.plot_gp_1d(
        ax              = ax,
        mu              = optimizer.mu,
        std             = optimizer.std(),
        action_space    = regression.action_space,
        feedback_idxs   = regression.get_feedback_idxs(),
        feedback_values = regression.get_feedback_values(),
        ground_truth    = groundtruth
    )
    ax.set_title('MCT Reward')

    fig.tight_layout()
    savepath = Path('hilo/output/mo_gp_fit.svg')
    fig.savefig(savepath)   
    print(f'Saved figure to {savepath.resolve()}')

if __name__ == '__main__':
    main()

