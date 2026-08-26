import matplotlib.pyplot as plt
import pypolar as plr
from pathlib import Path
import numpy as np

def plot_study(path: Path):
    datasets: list[plr.ExperimentDataset] = []
    for datapath in path.iterdir():
        if not datapath.is_file():
            continue
        
        datasets.append(plr.ExperimentDataset.load(datapath))
        
    regrets = {}
    for dataset in datasets:
        acqf = dataset.get_sources()[-1]
        rgts = dataset.get_aux('regret')
        if acqf in regrets:
            regrets[acqf] = np.vstack([regrets[acqf], rgts])
        else:
            regrets[acqf] = np.array([rgts])
            
    fig, ax = plt.subplots(figsize=(12,8))
    colors = ['red', 'blue', 'green', 'grey']
    for func, c in zip(regrets, colors):
        data = regrets[func]
        idxs = np.arange(len(data[0]))
        mean = data.mean(axis=0)
        std = data.std(axis=0)
        lo = data.min(axis=0)
        hi = data.max(axis=0)
        # lo = mean - std
        # hi = mean + std
        ax.plot(idxs, mean, label="mean " + func, color=c, lw=6)
        ax.fill_between(idxs, lo, hi, alpha=0.2, color=c)
        # for regret_in_trial in regrets[func]:
            # ax.plot(regret_in_trial, color=c, lw=1)
    
    fig.legend()
    fig.savefig('scripts/output/acqf_study.svg')
        

if __name__ == '__main__':
    plot_study(Path('scripts/output/experiments/20260825_144729'))