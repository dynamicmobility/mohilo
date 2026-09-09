import matplotlib.pyplot as plt
import pypolar as plr
from pathlib import Path
import numpy as np

def load_data(data_path: Path) -> list[plr.ExperimentDataset]:
    datasets = []
    # only the runs: a study directory also holds whatever was drawn from them
    for datapath in sorted(data_path.glob('*.json')):
        datasets.append(plr.ExperimentDataset.load(datapath))

    return datasets

def single_objective_study(datasets: list[plr.ExperimentDataset]):
    
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
    fig.savefig('scripts/output/so_acqf_study.svg')

def multi_objective_study(datasets: list[plr.ExperimentDataset]):
    regrets = {}
    keys = ['hv_regret', 'hv_regret_attained', 'front_alignment']
    data = {k: {} for k in keys}
    for dataset in datasets:
        acqf = dataset.get_sources()[-1]
        # rgts = dataset.get_aux('hv_regret')
        for k in keys:
            # data[k].append(dataset.get_aux(k))

            if acqf in data[k]:
                data[k][acqf] = np.vstack([data[k][acqf], dataset.get_aux(k)])
            else:
                data[k][acqf] = np.array([dataset.get_aux(k)])
    # for k in keys:
    #     data[k] = np.array(data[k])
            
    fig, axs = plt.subplots(ncols=3, figsize=(12,4))
    axs = axs.flatten()
    colors = ['C0', 'C1', 'C2', 'C3', 'C4']
    # one color per acquisition function, shared by every panel
    acqf_colors = dict(zip(data[keys[0]], colors))
    for k, ax in zip(keys, axs):
        for func, c in acqf_colors.items():
            d       = data[k][func]
            idxs    = np.arange(len(d[0]))
            mean    = d.mean(axis=0)
            lo      = d.min(axis=0)
            hi      = d.max(axis=0)
            # labelled on the first panel only, so the shared legend has one entry per acqf
            ax.plot(idxs, mean, label="mean " + func if ax is axs[0] else None, color=c, lw=2)
            ax.fill_between(idxs, lo, hi, alpha=0.2, color=c)
            plr.dress_axis(ax)
        
        ax.set_title(k)
        
    fig.legend(loc='lower center', ncols=len(acqf_colors))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig('scripts/output/mo_acqf_study.svg')



if __name__ == '__main__':
    datasets = load_data(Path('scripts/output/experiments/20260909_131158/'))
    if datasets[0].get_objectives().num_objectives > 1:
        multi_objective_study(datasets)
    else:
        single_objective_study(datasets)