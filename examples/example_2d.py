import numpy as np
import matplotlib.pyplot as plt
import time
from pypolar import PreferenceBasedLearning, SimulatedFeedback, SimulatedObjective, BasicGP, RandomSampler

np.random.seed(20)

# construct the PBL class. here, we define our action set.
pbl = PreferenceBasedLearning(
    low              = np.array([-1, -1]),
    high             = np.array([1, 1]),
    action_dims      = np.array([10, 10]),
    preference_noise = 0.01,
    coactive_noise   = 0.01,
    ordinal_noise    = 0.01,
)

objective = SimulatedObjective(
    smooth       = True,
    ord_lbls     = 3,
    action_space = pbl.action_space,
    function     = '2D-circle',
)

# simulates feedback according to a 2D objective function
fb = SimulatedFeedback(
    objective    = objective,
    action_space = pbl.action_space,
)

# setup a random sampler to sample actions during data collection
sampler = RandomSampler()

start = time.time()
epochs = 20  # number of pairwise comparisons to collect
prev = sampler.sample(pbl.action_space)
for idx in range(epochs):
    curr = sampler.sample(pbl.action_space)
    preference, coactive, ordinal = fb.evaluate(
        curr, prev,
        get_pairwise=True,
        get_coactive=False,
        get_ordinal=False,
    )
    pbl.add_feedback(preference, coactive, ordinal)
    prev = curr
print(f'Time to collect data: {time.time() - start:.2f}s')

# compile feedback into JAX arrays, then setup the GP
pbl.compile()
gp = BasicGP(lengthscale=1, signal_var=1)
gp.setup(
    action_space = pbl.action_space,
    likelihood   = pbl.overall_likelihood,
)

start = time.time()
mu = gp.fit(method='trust-krylov', options={'disp': False})
print(f'Time to fit: {time.time() - start:.2f}s')

predictions = 1000
correct = 0
start = time.time()
prev = sampler.sample(pbl.action_space)
for i in range(predictions):
    curr = sampler.sample(pbl.action_space)
    pred = pbl.predict(gp.mu, curr, prev)
    (_, _, label), _, _ = fb.evaluate(curr, prev, get_pairwise=True)
    correct += int(pred == label)
    prev = curr
print(f'Time to predict: {time.time() - start:.2f}s')

print(f'Accuracy: {correct / predictions * 100}%')
print(pbl.overall_likelihood(gp.mu))
print(pbl.overall_likelihood(objective(gp.actions)))

fig = plt.figure()
left = fig.add_subplot(121, projection='3d')
right = fig.add_subplot(122, projection='3d')

left.plot_trisurf(gp.actions[:, 0], gp.actions[:, 1], gp.mu, cmap='viridis', alpha=0.75)
left.scatter(gp.actions[:, 0], gp.actions[:, 1], gp.mu)
right.plot_trisurf(gp.actions[:, 0], gp.actions[:, 1], objective(gp.actions), cmap='viridis')
left.set_title('GP Mean')
right.set_title('Objective Function')

for ax in [left, right]:
    ax.set_xlabel('Action 1')
    ax.set_ylabel('Action 2')
    ax.set_zlabel('Reward')


def on_rotate(event):
    if event.inaxes == left:
        right.view_init(elev=left.elev, azim=left.azim)
    elif event.inaxes == right:
        left.view_init(elev=right.elev, azim=right.azim)
    fig.canvas.draw_idle()


fig.canvas.mpl_connect('motion_notify_event', on_rotate)
fig.set_size_inches((10, 6))
fig.tight_layout()
plt.show()
