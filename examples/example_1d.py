import os
os.environ["JAX_PLATFORMS"] = "cpu"
import numpy as np
import matplotlib.pyplot as plt
import pypolar as plr
import time

rng = np.random.default_rng(95)

# construct the PBL class. here, we define our action set.
pbl = plr.PreferenceBasedLearning(
    low              = np.array([0]),
    high             = np.array([6]),
    action_dims      = np.array([20]),
    preference_noise = 0.01,
    coactive_noise   = 0.04, # unused
    ordinal_noise    = 0.15, # unused
)

objective = plr.IdealPoint(
    w     = rng.uniform(low=0, high=6, size=(1,)),
    delta = 1.0
)

# simulates feedback according to a 1D objective function
oracle = plr.BradleyTerryOracle(
    beta_boltzmann = 1.0,
    reward_fn      = objective,
    rng            = rng
)

# setup the GP and Thompson sampler
gp                = plr.LaplaceGP(length_scale=1.5, signal_variance=1, rng=rng)
sampler           = plr.ThompsonSampler(gp, rng=rng)
# sampler           = plr.RandomSampler()

epochs = 20  # number of pairwise comparisons to collect
prev = sampler.sample(pbl.action_space)
def _likelihood():
    """Bind the feedback collected so far into a single-argument likelihood."""
    data = pbl.feedback_data()
    return lambda r: pbl.likelihood_from_data(r, data)

gp.set_data(
    action_space = pbl.action_space,
    likelihood   = _likelihood(),
)

train_preferences = []
t = time.time()
for idx in range(epochs):
    # Thompson sampling after first fit; random before that
    curr = sampler.sample(pbl.action_space)
    while np.all(curr == prev):
        curr = sampler.sample(pbl.action_space)

    preference = oracle.query(prev, curr)
    pbl.add_feedback((curr, prev, preference), None, None)
    train_preferences.append((curr, prev, preference))
    prev = curr

    gp.set_data(pbl.action_space, _likelihood())
    gp.fit(method='trust-constr', options={'disp': False})
    sampler.update_posterior()

print(f'Total time: {time.time() - t:.2f}s')

# Check consistency of the learned GP with collected preferences
consistent = 0
for curr, prev, label in train_preferences:
    pred = pbl.predict(gp.mu, curr, prev)
    consistent += int(pred == label)
print(f'Train consistency: {consistent / len(train_preferences) * 100}% '
      f'({consistent}/{len(train_preferences)} pairs)')

# Check accuracy of predicted preferred action against groundtruth objective
# TODO

actions = gp.actions.flatten()
std = gp.std()
fig, ax = plt.subplots()
ax.scatter(actions, gp.mu, c='b', s=1)
ax.plot(actions, gp.mu, c='b', ls='--', label='GP mean')
ax.fill_between(actions, gp.mu - 1 * std, gp.mu + 1 * std,
                color='b', alpha=0.2, label='GP ±σ')
ax.set_xlabel('Action')
ax.set_ylabel('GP reward', color='b')
ax.tick_params(axis='y', labelcolor='b')

ax2 = ax.twinx()
ax2.plot(actions, [objective.compute(a) for a in actions], c='r', label='Objective')
ax2.set_ylabel('Objective reward', color='r')
ax2.tick_params(axis='y', labelcolor='r')

ax.set_title('Preference fit')

plt.savefig('example_1d.pdf')
