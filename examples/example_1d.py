import numpy as np
import matplotlib.pyplot as plt
from pypolar import PreferenceBasedLearning, SimulatedFeedback, SimulatedObjective, BasicGP, ThompsonSampler, RandomSampler
import time

np.random.seed(1)

# construct the PBL class. here, we define our action set.
pbl = PreferenceBasedLearning(
    low              = np.array([0]),
    high             = np.array([6]),
    action_dims      = np.array([1000]),
    preference_noise = 0.001,
    coactive_noise   = 0.04,
    ordinal_noise    = 0.15,
)

objective = SimulatedObjective(
    smooth       = True,
    ord_lbls     = 3,
    action_space = pbl.action_space,
    function     = '1D',
)

# simulates feedback according to a 1D objective function
fb = SimulatedFeedback(
    objective    = objective,
    action_space = pbl.action_space,
)

# setup the GP and Thompson sampler
gp = BasicGP(lengthscale=1, signal_var=1)
sampler = ThompsonSampler(gp)
random_sampler = RandomSampler()

epochs = 20  # number of pairwise comparisons to collect
prev = random_sampler.sample(pbl.action_space)

t = time.time()
for idx in range(epochs):
    # Thompson sampling after first fit; random before that
    curr = sampler.sample(pbl.action_space)
    while np.all(curr == prev):
        curr = random_sampler.sample(pbl.action_space)

    preference, coactive, ordinal = fb.evaluate(
        curr         = curr,
        prev         = prev,
        get_pairwise = True,
        get_coactive = False,
        get_ordinal  = False,
    )
    pbl.add_feedback(preference, coactive, ordinal)
    prev = curr

    # refit the GP to update the posterior for Thompson sampling
    # preserve mu from the previous fit as a warm start
    old_mu = gp.mu.copy() if gp.mu is not None else None
    pbl.compile()
    gp.setup(
        action_space = pbl.action_space,
        likelihood   = pbl.overall_likelihood,
    )
    if old_mu is not None:
        gp.mu = old_mu
    gp.fit(method='trust-constr', options={'disp': False})
    sampler.update_posterior()

print(f'Total time: {time.time() - t:.2f}s')

# evaluate prediction accuracy
predictions = 1000
correct = 0
prev = random_sampler.sample(pbl.action_space)
for i in range(predictions):
    curr = random_sampler.sample(pbl.action_space)
    pred = pbl.predict(gp.mu, curr, prev)
    (_, _, label), _, _ = fb.evaluate(curr, prev, get_pairwise=True)
    correct += int(pred == label)
    prev = curr

print(f'Accuracy: {correct / predictions * 100}%')

actions = gp.actions.flatten()
fig = plt.figure()
left = fig.add_subplot(121)
right = fig.add_subplot(122)
left.scatter(actions, gp.mu, c='b', s=1)
left.plot(actions, gp.mu, c='b', ls='--')
right.plot(actions, objective(actions), c='r')
left.set_title('GP Mean')
right.set_title('Objective Function')
for ax in [left, right]:
    ax.set_xlabel('Action')
    ax.set_ylabel('Reward')

plt.show()
