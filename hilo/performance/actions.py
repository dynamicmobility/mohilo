import os
os.environ["JAX_PLATFORMS"] = "cpu"
import pypolar as plr
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from hilo.create import create_hipexo_sim
from config.hipexo import hipexo_sim_idealized
import time

def run_experiment(seed, config, num_queries):
    rng = np.random.default_rng(seed)
    
    regression, optimizer, sampler, groundtruth, oracle = create_hipexo_sim(
        rng, config
    )
    
    times = []
    hvs = []
    overlays = []
    
    # Run simulation
    for i in tqdm(range(num_queries), disable=True):
        start = time.time()
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
        end = time.time()
        
        estimated_objs    = np.array([optimizer.gps[i].mu for i in range(len(optimizer.gps))]).T
        true_objs         = groundtruth(regression.action_space)
        hvs.append(plr.groundtruth_hypervolume(
            estimated_objs    = estimated_objs,
            true_objs         = true_objs
        ))
        overlays.append(plr.pareto_overlay(
            estimated_objs    = estimated_objs,
            true_objs         = true_objs
        ))
        times.append(end - start)
    
    return times, hvs, overlays

def main():
    data = {'times': [], 'hvs': [], 'overlays': []}
    rng = np.random.default_rng(95)
    N_TRIALS = 20
    N_QUERIES = 5
    
    # Run experiments
    for i in tqdm(range(N_TRIALS)):
        config = hipexo_sim_idealized.model_copy(deep=True)
        # config.objective.w = np.array([[]])
        times, hvs, overlays = run_experiment(
            seed = round(rng.random() * 1000),
            config = config,
            num_queries = N_QUERIES, 
        )
        data['times'].append(times)
        data['hvs'].append(hvs)
        data['overlays'].append(overlays)

    # Plot and save
    fig, axs = plt.subplots(ncols=3, figsize=(15, 5))
    time_ax, hv_ax, overlay_ax = axs
    iterations = np.arange(len(times)) + 1
    
    for times in data['times']:
        time_ax.plot(iterations, times, color='red')
    
    for hvs in data['hvs']:
        hv_ax.plot(iterations, hvs, color='red')
    
    for overlays in data['overlays']:
        overlay_ax.plot(iterations, overlays, color='red')

    time_ax.set_title('Query Time')
    time_ax.set_xlabel('Iteration')
    time_ax.set_ylabel('Time (s)')

    hv_ax.set_title('Groundtruth Hypervolume')
    hv_ax.set_xlabel('Iteration')
    hv_ax.set_ylabel('Hypervolume')

    overlay_ax.set_title('Pareto Overlay')
    overlay_ax.set_xlabel('Iteration')
    overlay_ax.set_ylabel('Fraction correct')

    fig.tight_layout()
    savepath = Path('hilo/output/action_vs_performance.pdf')
    fig.savefig(savepath)   
    print(f'Saved figure to {savepath.resolve()}')

if __name__ == '__main__':
    main()

