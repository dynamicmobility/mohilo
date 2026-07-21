import os
os.environ["JAX_PLATFORMS"] = "cpu"
import pypolar as plr
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from hilo.create import create_hipexo_sim
from config.hipexo import hipexo_sim_idealized, hipexo_sim_idealized_2d

def main():
    rng = np.random.default_rng(95)

    # config = hipexo_sim_idealized_2d
    config = hipexo_sim_idealized
    regression, optimizer, sampler, groundtruth, oracle = create_hipexo_sim(
        rng,
        cfg=config
    )
    
    # Run simulation
    NUM_QUERIES = 20
    for i in tqdm(range(NUM_QUERIES)):
        # Sample an action
        sample_action = sampler.sample(regression.action_space)
        
        # Measure the human performance
        mct_hat, sp_hat = oracle.query(sample_action)
        
        regression.add_feedback(sample_action, [mct_hat, sp_hat])
        optimizer.setup(
            action_space = regression.action_space,
            regressions  = regression.get_regression_data()
        )
        optimizer.fit()
        sampler.update_posterior()

    true_objs = groundtruth(regression.action_space)
    print(plr.groundtruth_hypervolume(
        estimated_objs = np.array([optimizer.gps[i].mu for i in range(config.num_objs)]).T,
        true_objs = true_objs,
        tol=0.05
    ))
    print(plr.pareto_overlay(
        estimated_objs = np.array([optimizer.gps[i].mu for i in range(config.num_objs)]).T,
        true_objs = true_objs,
        tol=0.05
    ))
    # quit()

    fig, axs = plt.subplots(ncols=3, figsize=(15, 5))
    pareto_ax, mct_ax, sp_ax = axs

    pareto_ax = plr.plot_pareto_2d(
        ax                = pareto_ax,
        optimizer         = optimizer,
        regression        = regression,
        mo_ground_truth   = groundtruth
    )
    pareto_ax.set_xlabel('Obj 1: MCT Reward')
    pareto_ax.set_ylabel('Obj 2: SP Reward')

    for i in range(2):
        plr.plot_gp_1d(
            ax              = axs[i + 1],
            mu              = optimizer.gps[i].mu,
            std             = optimizer.gps[i].std(),
            action_space    = regression.action_space,
            feedback_idxs   = regression.get_feedback_idxs(),
            feedback_values = regression.get_feedback_values(i),
            ground_truth    = lambda x: (groundtruth(x)[:, i])
        )
    
    mct_ax.set_title('Objective 1: MCT Reward')
    sp_ax.set_title('Objective 2: Speed Reward')

    fig.tight_layout()
    savepath = Path('hilo/output/mo_gp_fit.svg')
    fig.savefig(savepath)   
    print(f'Saved figure to {savepath.resolve()}')

if __name__ == '__main__':
    main()

