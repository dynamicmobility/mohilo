import os
os.environ["JAX_PLATFORMS"] = "cpu"
import numpy as np
import matplotlib.pyplot as plt
import pypolar as plr
from pypolar import IdealPoint, MultiObjectiveIdealPoint
import time
from tqdm import tqdm

rng = np.random.default_rng(95)

pbl1 = plr.PreferenceBasedLearning(
    low              = np.array([0]),
    high             = np.array([6]),
    action_dims      = np.array([40]),
    preference_noise = 0.01,
)

pbl2 = plr.PreferenceBasedLearning(
    low              = np.array([0]),
    high             = np.array([6]),
    action_dims      = np.array([40]),
    preference_noise = 0.01,
)

f1 = IdealPoint(w=1.0, delta=1.0)
f2 = IdealPoint(w=4.0, delta=1.0)
F  = MultiObjectiveIdealPoint(ideal_point_rewards=[f1, f2])
oracle = plr.MultiObjectiveOracle(
    beta_boltzmann = 1.0,
    reward_fn      = F,
    rng            = rng
)

gp_f1   = plr.LaplaceGP(length_scale=2.5, signal_variance=2, rng=rng)
gp_f2   = plr.LaplaceGP(length_scale=2.5, signal_variance=2, rng=rng)
sampler = plr.RandomSampler(rng)
sampler = plr.DSTSampler(gps=[gp_f1, gp_f2], rng=rng, rho=0.05)

epochs = 1000  # number of pairwise comparisons to collect
def _likelihood(pbl):
    """Bind the feedback collected so far into a single-argument likelihood."""
    data = pbl.feedback_data()
    return lambda r: pbl.likelihood_from_data(r, data)

gp_f1.set_data(pbl1.action_space, _likelihood(pbl1))
gp_f2.set_data(pbl2.action_space, _likelihood(pbl2))

train_preferences = []
t = time.time()
two = sampler.sample(pbl1.action_space)
for idx in tqdm(range(epochs)):
    # Thompson sampling after first fit; random before that
    one = sampler.sample(pbl1.action_space)
    while np.all(one == two):
        two = sampler.sample(pbl1.action_space)

    p1, p2 = oracle.mo_query(one, two)
    pbl1.add_feedback((two, one, p1), None, None)
    pbl2.add_feedback((two, one, p2), None, None)
    train_preferences.append((two, one, p1, p2))
    sampler.update_posterior()

    gp_f1.set_data(pbl1.action_space, _likelihood(pbl1))
    gp_f1.fit(method='trust-constr', options={'disp': False})
    gp_f2.set_data(pbl2.action_space, _likelihood(pbl2))
    gp_f2.fit(method='trust-constr', options={'disp': False})
print(f'Total time: {time.time() - t:.2f}s')

def normalize(f):
    f_min = np.min(f)
    f_max = np.max(f)
    return (f - f_min) / (f_max - f_min)

# Plotting
std1 = gp_f1.std()
std2 = gp_f2.std()
fig, axs = plt.subplots(ncols=3)
pareto, ax_f1, ax_f2 = axs
pareto.plot(
    normalize(gp_f1.mu), 
    normalize(gp_f2.mu), 
    c='b', label='Learned Frontier')
f1_true = np.array([f1.compute(x) for x in pbl1.action_space])
f2_true = np.array([f2.compute(x) for x in pbl2.action_space])
pareto.plot(
    normalize(f1_true), 
    normalize(f2_true),
    c='g', ls='--', label='True Frontier')
# ax.fill_between(gp_f1.mu - 1 * std1, gp_f1.mu + 1 * std1,
                # gp_f2.mu - 1 * std2, gp_f2.mu + 1 * std2,
                # color='b', alpha=0.2, label='GP ±σ')
pareto.set_xlabel('f1')
pareto.set_ylabel('f2')
pareto.legend()
pareto.set_title('Learned Frontier')

ax_f1.plot(pbl1.action_space, normalize(gp_f1.mu), c='r', label='GP Mean')
ax_f1.plot(pbl1.action_space, normalize(f1_true), c='g', ls='--', label='True Function')
# ax_f1.fill_between(
#     pbl1.action_space[:,0], gp_f1.mu - 1 * std1, gp_f1.mu + 1 * std1,
#     color='r', alpha=0.2, label='GP ±σ')
ax_f1.set_xlabel('x')
ax_f1.set_ylabel('f1(x)')


ax_f2.plot(pbl2.action_space, normalize(gp_f2.mu), c='r', label='GP Mean')
ax_f2.plot(pbl2.action_space, normalize(f2_true), c='g', ls='--', label='True Function')
# ax_f2.fill_between(
#     pbl2.action_space[:,0], gp_f2.mu - 1 * std2, gp_f2.mu + 1 * std2,
#     color='r', alpha=0.2, label='GP ±σ')
ax_f2.set_xlabel('x')
ax_f2.set_ylabel('f2(x)')
name = 'output/plots/gp_pareto.pdf'
fig.set_size_inches(12, 4)
fig.tight_layout()
fig.savefig(name)
print('Saved figure to', name)
