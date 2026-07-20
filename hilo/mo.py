import os
os.environ["JAX_PLATFORMS"] = "cpu"
import pypolar as plr
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from hilo.create import create_hipexo_sim
from config.hipexo import hipexo_sim_idealized

def main():
    rng = np.random.default_rng(95)
    
    hipexo_sim_idealized.objective.w = np.array([[0.1], [3.9]])
    regression, optimizer, sampler, groundtruth, oracle = create_hipexo_sim(
        rng,
        cfg=hipexo_sim_idealized
    )
    
    # Run simulation
    NUM_QUERIES = 10
    for i in tqdm(range(NUM_QUERIES)):
        # Sample an action
        sample_action = sampler.sample(regression.action_space)
        
        # Measure the human performance
        mct_hat, sp_hat = oracle.query(sample_action)
        
        regression.add_feedback(sample_action, [mct_hat, sp_hat])
        optimizer.setup(
            action_space = regression.action_space,
            likelihoods  = regression.get_likelihood_functions()
        )
        optimizer.fit(method='trust-constr', options={'disp': False})
        sampler.update_posterior()

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
    savepath = Path('hilo/output/mo_gp_fit.pdf')
    fig.savefig(savepath)   
    print(f'Saved figure to {savepath.resolve()}')

if __name__ == '__main__':
    main()

