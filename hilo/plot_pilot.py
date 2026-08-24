from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from botorch.models import ModelListGP
from botorch.utils.multi_objective.box_decompositions.dominated import (
    DominatedPartitioning,
)

import pypolar as plr
from pypolar.optimization.gp import DTYPE

# TODO: this needs to be remade with up-to-date code and plot metrics + LOO + recommendations etc...
# you should create an example dataset using synthetics for testing.

ACTIONS         = ['h_flex_torque_scale', 'h_ext_torque_scale', 'hip_delay_idx']
OBJS_PATH       = Path('human_data/pilot_mohilo.csv')
ACTIONS_PATH    = Path('human_data/MH01_walk.csv')

CONTOUR_RES     = 200                       # plotting only
SEED            = 95


def compute_hypervolume(Y, ref=None):
    """Finds the hypervolume dominated by Y in reference to ref"""
    Y = torch.as_tensor(np.atleast_2d(Y), dtype=DTYPE)
    ref = ref if ref is not None else torch.zeros((Y.shape[1],))
    return DominatedPartitioning(ref_point=ref, Y=Y).compute_hypervolume().item()

def posterior_mu_at(model: ModelListGP, X, chunk=2048):
    """Per-objective posterior mean at X. Vectorized evaluation in sizes of chunk"""
    X = torch.as_tensor(np.atleast_2d(X), dtype=DTYPE).unsqueeze(1)
    with torch.no_grad():
        out = [model.posterior(X[i:i+chunk]).mean.squeeze(1) for i in range(0, X.shape[0], chunk)]
    
    return torch.cat(out).numpy()

def read_data(objs_path: Path, actions_path: Path) -> tuple[plr.DecoupledObjectives]:
    objs_df = pd.read_csv(objs_path)
    actions_df = pd.read_csv(actions_path)
    
    objectives = plr.DecoupledObjectives.from_empty()
    objectives.add_objective(plr.Objective.from_data(
        actions   = actions_df[ACTIONS],
        values    = objs_df['Cost'],
        maximize  = False,
        name      = 'Metabolic Cost'
    ))
    objectives.add_objective(plr.Objective.from_data(
        actions   = actions_df[ACTIONS],
        values    = objs_df['Speed'],
        maximize  = False,
        name      = '10m walk test'
    ))
    objectives.add_objective(plr.Objective.from_data(
        actions   = actions_df[ACTIONS],
        values    = objs_df['Comfort Treadmill'],
        maximize  = True,
        name      = 'Comfort Treadmill'
    ))
    objectives.add_objective(plr.Objective.from_data(
        actions   = actions_df[ACTIONS],
        values    = objs_df['Comfort Floor'],
        maximize  = True,
        name      = 'Comfort Floor'
    ))
    
    return objectives
    

def main():
    # read in the data
    objectives = read_data(
        actions_path    = ACTIONS_PATH,
        objs_path       = OBJS_PATH
    )
    
    OBJ1 = 'Metabolic Cost'
    OBJ2 = 'Comfort Treadmill'
    
    # OBJ1 = '10m walk test'
    # OBJ2 = 'Comfort Floor'
        
    mogp = plr.DecoupledMOGP(
        objectives=objectives[[OBJ1, OBJ2]],
        fit_hyperparameters=True
    )
    
    x = np.linspace(0, 1, 50)
    X, Y, Z = np.meshgrid(x,x,x)
    points = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))

    objs, stds = mogp.posterior_at(action=points, normalized=True)
    nd_objs = objs[plr.get_nondominated(objs)]
    
    # plot them
    fig, ax = plt.subplots()
    ax.scatter(objectives[OBJ1].ytransform.inv(objs[:, 0]), objectives[OBJ2].ytransform.inv(objs[:, 1]), s=5, alpha=0.1)
    ax.scatter(objectives[OBJ1].ytransform.inv(nd_objs[:, 0]), objectives[OBJ2].ytransform.inv(nd_objs[:, 1]), s=25, c='red')
    fig.savefig('hilo/output/test.png', dpi=500)
    
    # print best of each
    actions, mu, std = mogp.recommend(raw=True)
    width = max(len(name) for name in mogp.objectives.names) + 6   # room for ' (min)'

    print('\n=== best action per objective ===')
    print(f'  {"objective":<{width}} ' + ''.join(f'{col:>22}' for col in ACTIONS)
          + f'{"predicted":>22}{"best measured":>16}')
    for i in range(len(mogp.objectives)):
        obj = mogp.objectives[i]
        label = f'{obj.name} ({"max" if obj.maximize else "min"})'
        best_seen = obj.ydata.max() if obj.maximize else obj.ydata.min()
        print(f'  {label:<{width}} ' + ''.join(f'{a:>22.4f}' for a in actions[i])
              + f'{mu[i]:>13.3f} +/- {std[i]:<5.3f}'
              + f'{best_seen:>16.3f}')


if __name__ == '__main__':
    main()