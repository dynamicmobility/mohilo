import numpy as np
import matplotlib.pyplot as plt
import time
from pbl import PreferenceBasedLearning
from feedback import SimulatedFeedback, SimulatedObjective
from gp import BasicGP
from sampler import RandomSampler
np.random.seed(20)

# construct the PBL class. here, we define our action set.
pbl = PreferenceBasedLearning(low=np.array([-1, -1]),
                              high=np.array([1, 1]),
                              action_dims=np.array([10, 10]),
                              preference_noise=0.01,
                              coactive_noise=0.01,
                              ordinal_noise=0.01)
print('here')

objective = SimulatedObjective(smooth=True, 
                               ord_lbls=3,
                               action_space=pbl.action_space,
                               function='2D')

# simulates feedback according to a 1D objective function
fb = SimulatedFeedback(objective=objective,
                       action_space=pbl.action_space) 

# setup the GP model
gp = BasicGP(lengthscale=1, signal_var=1)
gp.setup(pbl.action_space, likelihood=pbl.overall_likelihood,
                           dlikelihood=pbl.overall_jacobian,
                        #    d2likelihood=pbl.overall_hessian
                           )
# setup a random sampler to sample actions during data collection
sampler = RandomSampler()
start = time.time()
epochs = 20 # number of pairwise comparisons to collect
prev = sampler.sample(pbl.action_space)
for idx in range(epochs):
    # collect and add feedback to PBL object
    curr = sampler.sample(pbl.action_space)
    preference, coactive, ordinal = fb.evaluate(curr, prev,
                                                get_pairwise=True,
                                                get_coactive=False,
                                                get_ordinal=False)
    pbl.add_feedback(preference, coactive, ordinal)
    prev = curr
elapsed = time.time() - start
print(f'Time to collect data: {elapsed}')

start = time.time()
# fit the model
mu = gp.fit(method='trust-constr', options={'disp': False})
elapsed = time.time() - start
print(f'Time to fit: {elapsed}')

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
print(pbl.overall_likelihood(gp.mu))
print(pbl.overall_likelihood(objective(gp.actions)))

fig = plt.figure()
left = fig.add_subplot(121, projection='3d')
right = fig.add_subplot(122, projection='3d')

# Plot the surface
left.plot_trisurf(gp.actions[:,0], gp.actions[:, 1], gp.mu, cmap='viridis', alpha=0.75)
left.scatter(gp.actions[:,0], gp.actions[:, 1], gp.mu)
right.plot_trisurf(gp.actions[:,0], gp.actions[:, 1], objective(gp.actions), cmap='viridis')
left.set_title('GP Mean')
right.set_title('Objective Function')

for ax in [left, right]:
    ax.set_xlabel('Action 1')
    ax.set_ylabel('Action 2')
    ax.set_zlabel('Reward')
    
# Function to synchronize the rotation
def on_rotate(event):
    ax, elev, azim = None, 0, 0
    if event.inaxes == left:
        elev = left.elev
        azim = left.azim
        ax = right
    else:
        elev = right.elev
        azim = right.azim
        ax = left
    ax.view_init(elev=elev, azim=azim)

    fig.canvas.draw_idle()

fig.canvas.mpl_connect('motion_notify_event', on_rotate)
fig.set_size_inches((10, 6))
fig.tight_layout()
plt.show()