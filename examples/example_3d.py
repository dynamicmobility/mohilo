import numpy as np
from pypolar import PreferenceBasedLearning, SimulatedFeedback, SimulatedObjective, BasicGP, RandomSampler
import time

np.random.seed(0)

# construct the PBL class. here, we define our action set.
pbl = PreferenceBasedLearning(
    low              = np.ones(3) * -1,
    high             = np.ones(3),
    action_dims      = np.ones(3).astype(int) * 10,
    preference_noise = 0.01,
    coactive_noise   = 0.04,
    ordinal_noise    = 0.15,
)

objective = SimulatedObjective(
    smooth       = True,
    ord_lbls     = 3,
    action_space = pbl.action_space,
    function     = 'vibrant',
)

# simulates feedback according to a 3D objective function
fb = SimulatedFeedback(
    objective    = objective,
    action_space = pbl.action_space,
)

# setup a random sampler to sample actions during data collection
sampler = RandomSampler()

epochs = 200  # number of pairwise comparisons to collect
prev = sampler.sample(pbl.action_space)
for idx in range(epochs):
    curr = sampler.sample(pbl.action_space)
    while np.all(curr == prev):
        curr = sampler.sample(pbl.action_space)

    preference, coactive, ordinal = fb.evaluate(
        curr         = curr,
        prev         = prev,
        get_pairwise = True,
        get_coactive = False,
        get_ordinal  = False,
    )
    pbl.add_feedback(preference, coactive, ordinal)
    prev = curr

# compile feedback into JAX arrays, then setup the GP
pbl.compile()
gp = BasicGP(lengthscale=1, signal_var=1)
gp.setup(
    action_space = pbl.action_space,
    likelihood   = pbl.overall_likelihood,
)

t = time.time()
mu = gp.fit(method='trust-krylov', options={'disp': False})
print(f'Time to fit: {time.time() - t:.2f}s')

predictions = 1000
correct = 0
prev = sampler.sample(pbl.action_space)
for i in range(predictions):
    curr = sampler.sample(pbl.action_space)
    pred = pbl.predict(gp.mu, curr, prev)
    (_, _, label), _, _ = fb.evaluate(curr, prev, get_pairwise=True)
    correct += int(pred == label)
    prev = curr

print(f'Accuracy: {correct / predictions * 100}%')
