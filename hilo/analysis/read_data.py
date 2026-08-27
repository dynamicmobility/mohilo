"""Convert CSV files into Objective datastructures"""
import pandas as pd
import pypolar as plr
from pathlib import Path
import numpy as np

def read_MH01_data(
    objs_path   : Path = Path('human_data/pilot_mohilo.csv'),
    actions_path: Path = Path('human_data/MH01_walk.csv')
) -> tuple[plr.DecoupledObjectives]:
    """Function designed for pilot study completed on Friday, July 23rd, 2026."""
    objs_df = pd.read_csv(objs_path)
    actions_df = pd.read_csv(actions_path)
    ACTIONS = ['h_flex_torque_scale', 'h_ext_torque_scale', 'hip_delay_idx']

    
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


def read_MT0x_data(
    filepath: Path = Path('human_data/MT03_incline.csv'),
    seed=95,
    num_samples=None,
):
    df = pd.read_csv(filepath)
    ACTIONS = [
        'h_flex_torque_scale',
        'h_flex_power_scale',
        'h_ext_power_scale',
        'k_ext_torque_scale',
        'k_ext_power_scale',
        'k_flex_torque_scale',
        'k_flex_power_scale'
    ]
    actions = df[ACTIONS].to_numpy()
    y = df['cost'].to_numpy()
    
    if num_samples is None:
        num_samples = len(df)
    rng = np.random.default_rng(seed)
    idxs = rng.choice(np.arange(len(df)), size=num_samples, replace=False)
    obj = plr.Objective.from_data(
        actions  = actions[idxs],
        values   = y[idxs],
        maximize = False,
        name     = 'Metabolic Cost'
    )
    return obj