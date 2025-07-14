import numpy as np
from pbl import PreferenceBasedLearning as PBL
from feedback import SimulatedFeedback, SimulatedObjective
from gp import BasicGP
from sampler import RandomSampler
import time
np.random.seed(0)
# construct the PBL class. here, we define our action set.
pbl = PBL(
    low              = np.array([0]),
    high             = np.array([6]),
    action_dims      = np.array([40]),
    preference_noise = 0.01,
    coactive_noise   = 0.04,
    ordinal_noise    = 0.15
)

objective = SimulatedObjective(
    smooth       = True,
    ord_lbls     = 3,
    action_space = pbl.action_space,
    function     = '1D'
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
    # d2likelihood = pbl.overall_hessian
) 

# setup a random sampler to sample actions during data collection
sampler = RandomSampler()

epochs = 20 # number of pairwise comparisons to collect
prev = sampler.sample(pbl.action_space)
for idx in range(epochs):
    # collect and add feedback to PBL object
    curr = sampler.sample(pbl.action_space)
    while curr == prev:
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

if True:
    print('FBK:\t', pbl.preference_fbk)
    def check_gradient(obj, jac):
        # Check the gradient
        eps = 1e-6
        r = np.random.rand(N)
        # r = np.array([0.25, 0.5, 0.75, 1])
        print('R:\t', r)
        print('Obj:\t', obj(r))
        grad = jac(r)
        approx_grad = np.zeros(N)
        for i in range(N):
            r_plus = r.copy()
            r_plus[i] += eps
            r_minus = r.copy()
            r_minus[i] -= eps
            approx_grad[i] = ((obj(r_plus) - obj(r_minus)) / (2 * eps))
        print("Error:\t", np.linalg.norm(grad - approx_grad))
        print('Grad:\t', grad)
        print('Approx\t', approx_grad)

    # print(pbl.preference_fbk)
    check_gradient(gp.objective, gp.jacobian)
    quit()
    

# fit the model
t = time.time()
mu = gp.fit(method='trust-constr', options={'disp': True})
print(time.time() - t)
# hess = gp.hessian(gp.mu)
# hessinv = np.linalg.inv(hess)
# print('HESS')
# print(np.diagonal(hess))
# print(np.linalg.eigvals(hess))
# print('---')
# print('HESSINV')
# print(np.diagonal(hessinv))
# print(np.linalg.eigvals(hessinv))
# print('---')
# print(np.round(hess@hessinv, 2))
# std = np.sqrt(np.diag(np.linalg.inv(gp.hessian(gp.mu))))
# print(std)
# # std /= np.max(mu) - np.min(mu)
# print(std)
# print(np.max(mu) - np.min(mu))
predictions = 1000
correct = 0
prev = sampler.sample(pbl.action_space)
for i in range(predictions):
    # sample the next action
    curr = sampler.sample(pbl.action_space)
    pred = pbl.predict(gp.mu, curr, prev)
    (_, _, label), _, _ = fb.evaluate(curr, prev, get_pairwise=True)
    correct += int(pred == label[0])
    prev = curr

print(f'Accuracy: {correct / predictions * 100}%')
# quit()
import matplotlib.pyplot as plt
actions = gp.actions.flatten()
fig = plt.figure()
left = fig.add_subplot(121)
right = fig.add_subplot(122)
left.scatter(actions, gp.mu, c='b', s=1)
# left.fill_between(actions, gp.mu - std, gp.mu + std, color='b', alpha=0.25)
left.plot(actions, gp.mu, c='b', ls='--')
right.plot(actions, objective(actions), c='r')
left.set_title('GP Mean')
right.set_title('Objective Function')
for ax in [left, right]: 
    ax.set_xlabel('Action')
    ax.set_ylabel('Reward')

plt.show()