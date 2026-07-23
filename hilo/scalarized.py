import pypolar as plr
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt

rng = np.random.default_rng(95)

# GP Regression
optimizer = plr.LaplaceGP(
    kernel            = 'squared_exp',
    signal_variance   = 10.0,
    length_scale      = 1.0,
    x0_init_method    = 'random',
    rng               = rng
)

regression = plr.Regression(
    low           = np.array([0.0]),
    high          = np.array([4.0]),
    action_dims   = np.array([100]),
    precision     = 1e1
)

# Sampling setup
sampler = plr.RandomSampler(
    rng = rng
)

# Groundtruth oracles
mct = plr.IdealPoint(
    w     = np.array([2.0]),
    delta = 1.0,
    gamma = 1.0
)
W_MCT = 1.0
mct_oracle = plr.NoisyRegressionOracle(
    reward_fn = mct,
    rng       = rng,
    noise_std = 0.0
)

sp = plr.IdealPoint(
    w     = np.array([-1.5]),
    delta = 1.0,
    gamma = 1.0
)
W_SP = 1.0
sp_oracle = plr.NoisyRegressionOracle(
    reward_fn = sp,
    rng       = rng,
    noise_std = 0.0,
)

def mo_reward(mct_hat, sp_hat):
    return np.array([mct_hat, 1 / sp_hat])**2
scalarized_reward = lambda mct_hat, sp_hat: np.dot(np.array([W_MCT, W_SP]), mo_reward(mct_hat, sp_hat))

# Run simulation
NUM_QUERIES = 20
feedback_data = []
for i in range(NUM_QUERIES):
    # Sample an action
    sample_idx, sample_action = sampler.sample(regression.action_space)
    
    # Measure the human performance
    mct_hat = mct_oracle.query(sample_action)
    sp_hat  = sp_oracle.query(sample_action)
    
    # Optimize to the scalarized reward
    reward = scalarized_reward(mct_hat, sp_hat)
    
    regression.add_feedback(sample_action, reward)
optimizer.set_data(
    action_space = regression.action_space,
    likelihood   = regression.likelihood
)
optimizer.fit(method='trust-constr', options={'disp': False})
    
ground_truth = lambda x: scalarized_reward(mct(x), sp(x))
fig, axs = plt.subplots(ncols=3, figsize=(15,5))
gp_ax, mct_ax, sp_ax = axs

plr.plot_gp_1d(gp_ax, optimizer, regression, ground_truth=ground_truth)

x = regression.action_space
mct_ax.plot(x.ravel(), mct(x))
mct_ax.set_title('Metabolic Cost of Transport')
mct_ax.set_xlabel('Action')
mct_ax.set_ylabel('Total MCT')

sp_ax.plot(x.ravel(), 1 / sp(x))
sp_ax.set_title('Walking Time')
sp_ax.set_xlabel('Action')
sp_ax.set_ylabel('Total Time')

fig.tight_layout()
name = 'scalarized_gp_fit.pdf'
fig.savefig(name)
print(f'Saved figure to {name}')