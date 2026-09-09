import pypolar as plr
import numpy as np
import pandas as pd
import argparse
from pathlib import Path

def compare_hypervolume(
    pareto_pts: plr.DecoupledObjectives,
    antipareto_pts: plr.DecoupledObjectives,
    ref_point: np.ndarray
):
    p_hv  = plr.hypervolume_from_objectives(pareto_pts, ref_point)
    ap_hv = plr.hypervolume_from_objectives(antipareto_pts, ref_point)
    
    print('=== Compare Hypervolume ====')
    print(f'The predicted pareto optimal points have a hypervolume of {p_hv}')
    print(f'The predicted anti-pareto points have a hypervolume of {ap_hv}')
    print('============================')


def compare_ordering(
    pareto_pts: plr.DecoupledObjectives,
    gp_prediction: dict[str, list] # each objectives ydata's indexes sorted from worst to best,
):
    class Op:
        def __init__(self, op):
            self.op = op
        def __call__(self, x, y):
            return eval(f'{x} {self.op} y')
        
    print('=== Compare Ordering ====')
    for obj in pareto_pts:
        ydata = pareto_pts[obj.name].ydata
        # idxs = np.argsort(ydata)
        idxs = gp_prediction[obj.name]
        
        left, middle, right = ydata[idxs]
        op = Op('<=') if obj.maximize else Op('>=')
        
        print(f'For objective {obj.name}')
        print(f'LEFT vs. MIDDLE:\t {left:.4f} {op.op} {middle:.4f}. {op(left, middle)}.')
        print(f'MIDDLE vs. RIGHT:\t {middle:.4f} {op.op} {right:.4f}. {op(middle, right)}.')
        print(f'LEFT vs. RIGHT: \t {left:.4f} {op.op} {right:.2f}. {op(left, right)}.')
        print()
    print('============================')

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        'dataset', 
        type    = Path,
        nargs   = '?',
        help    = 'the folder containing the dataset and evaluation.csv'
    )

    return p.parse_args()

def main():
    args = parse_args()
    d_validation = plr.ExperimentDataset.load(
        path = args.dataset / 'evaluation.json'
    )
        
    sources = d_validation.get_sources()
    
    pareto_pts = plr.DecoupledObjectives.from_empty()
    antipareto_pts = plr.DecoupledObjectives.from_empty()
    collected_objs = d_validation.get_objectives()
    for obj in collected_objs:
        for pts in (pareto_pts, antipareto_pts):
            pts.add_objective(
                plr.Objective.from_empty(
                    name     = obj.name,
                    maximize = obj.maximize
                )
            )
    
    actions = d_validation.get_actions()
    for i in range(len(actions)):
        pts = pareto_pts if sources[i] == 'pareto' else antipareto_pts
        for n in collected_objs.names:
            idxs = np.linalg.norm(collected_objs[n].xdata - actions[i], axis=1) < 1e-4            
            data = collected_objs[n].ydata[idxs]
            pts.add_point(
                objs = n,
                actions = actions[i],
                values = np.mean(data)
            )
    ref_point = plr.reference_point_from_objectives(
        objectives = (pareto_pts + antipareto_pts),
        margin     = 0.1
    )
    
    eval_df = pd.read_csv(args.dataset / 'evaluation.csv')
    pareto_df = eval_df[eval_df['type'] == 'pareto']
    gp_prediction = {}
    
    for obj in d_validation.get_objectives():
        data = np.array(pareto_df[f'{obj.name} est'].values)
        data = -data if not obj.maximize else data
        print(data)
        gp_prediction[obj.name] = np.argsort(data)
    
    print('Pareto optimal points ====')
    print(pareto_pts)
    print()
    
    print('Anti-pareto points ====')
    print(antipareto_pts)
    print()
    
    print(f'The chosen reference point is {ref_point}, {pareto_pts.names}')
    print()
    
    compare_hypervolume(
        pareto_pts     = pareto_pts,
        antipareto_pts = antipareto_pts,
        ref_point      = ref_point
    )
    
    print()
    compare_ordering(
        pareto_pts    = pareto_pts,
        gp_prediction = gp_prediction
    )

if __name__ == '__main__':
    main()