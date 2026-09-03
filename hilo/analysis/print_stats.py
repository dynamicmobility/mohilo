import pypolar as plr
import numpy as np

d = plr.ExperimentDataset.load(
    path = 'human_data/MB02/evaluation.json'
)

objs = d.get_objectives()
comfort = objs['Comfort']
met = objs['Metabolic Cost']

c = []
for i in range(0, len(comfort.ydata) - 1, 3):
    c.append(np.mean(comfort.ydata[i:i+3]))
c = np.asarray(c)
c = np.asarray(comfort.ydata)

print(c)
print()
print(met.ydata)

pareto = np.stack([-c[[1, 3, 5]], met.ydata[[1, 3, 5]] - 6]).T
antipareto = np.stack([-c[[0, 2, 4]], met.ydata[[0, 2, 4]] - 6]).T
print(pareto)

print(plr.hypervolume_from_nondominated(pareto))
print(plr.hypervolume_from_nondominated(antipareto))