import numpy as np
from pbl import PreferenceBasedLearning as PBL
from feedback import SimulatedFeedback, SimulatedObjective
from gp import BasicGP
from sampler import RandomSampler
import time
np.random.seed(0)
# construct the PBL class. here, we define our action set.
pbl = PBL(
    low              = np.ones(3) * -1,
    high             = np.ones(3),
    action_dims      = np.ones(3).astype(int) * 15,
    preference_noise = 0.01,
    coactive_noise   = 0.04,
    ordinal_noise    = 0.15
)

objective = SimulatedObjective(
    smooth       = True,
    ord_lbls     = 3,
    action_space = pbl.action_space,
    function     = '3D'
)

# simulates feedback according to a 1D objective function
fb = SimulatedFeedback(
    objective    = objective,
    action_space = pbl.action_space
) 

# setup the GP model
gp = BasicGP(lengthscale=1, signal_var=1)
gp.setup(
    action_space = pbl.action_space,
    likelihood   = pbl.overall_likelihood,
    dlikelihood  = pbl.overall_jacobian,
    d2likelihood = pbl.overall_hessian
) 

# setup a random sampler to sample actions during data collection
sampler = RandomSampler()

epochs = 20 # number of pairwise comparisons to collect
prev = sampler.sample(pbl.action_space)
for idx in range(epochs):
    # collect and add feedback to PBL object
    curr = sampler.sample(pbl.action_space)
    while np.all(curr == prev):
        curr = sampler.sample(pbl.action_space)
    
    preference, coactive, ordinal = fb.evaluate(
        curr         = curr, 
        prev         = prev,
        get_pairwise = True,
        get_coactive = False,
        get_ordinal  = False
    )
    pbl.add_feedback(preference, coactive, ordinal)
    prev = curr

N = len(pbl.action_space)

t = time.time()
mu = gp.fit(method='trust-krylov', options={'disp': False})
# mu = gp.fit(method='trust-constr', options={'disp': True})
print('Time to fit', time.time() - t)
print()

predictions = 1000
correct = 0

start = time.time()
prev = sampler.sample(pbl.action_space)
for i in range(predictions):
    # sample the next action
    curr = sampler.sample(pbl.action_space)
    pred = pbl.predict(gp.mu, curr, prev)
    (_, _, label), _, _ = fb.evaluate(curr, prev, get_pairwise=True)
    correct += int(pred == label)
    prev = curr
elapsed = time.time() - start
print(f'Time to predict: {elapsed}')

print(f'Accuracy: {correct / predictions * 100}%')