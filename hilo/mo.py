import pypolar as plr
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt

rng = np.random.default_rng(95)

# GP Regression
mct_optimizer = plr.BasicGP(
    kernel            = 'squared_exp',
    signal_variance   = 10.0,
    length_scale      = 1.0,
    mu_init_method    = 'random',
    rng               = rng
)
sp_optimizer = plr.BasicGP(
    kernel            = 'squared_exp',
    signal_variance   = 10.0,
    length_scale      = 1.0,
    mu_init_method    = 'random',
    rng               = rng
)

mct_regression = plr.Regression(
    low           = np.array([0.0]),
    high          = np.array([4.0]),
    action_dims   = np.array([100]),
    precision     = 1e1
)
sp_regression = plr.Regression(
    low           = np.array([0.0]),
    high          = np.array([4.0]),
    action_dims   = np.array([100]),
    precision     = 1e1
)

# Sampling setup
sampler = plr.DSTSampler(gps=[mct_optimizer, sp_optimizer], rng=rng, rho=0.05)

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
    w     = np.array([1.0]),
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
    return np.array([mct_hat, sp_hat])
scalarized_reward = lambda mct_hat, sp_hat: np.dot(np.array([W_MCT, W_SP]), mo_reward(mct_hat, sp_hat))

# Run simulation
NUM_QUERIES = 10
feedback_data = []
for i in range(NUM_QUERIES):
    # Sample an action
    sample_action = sampler.sample(mct_regression.action_space)
    
    # Measure the human performance
    mct_hat = mct_oracle.query(sample_action)
    sp_hat  = sp_oracle.query(sample_action)
    
    # Optimize to the scalarized reward
    mct_reward, sp_reward = mo_reward(mct_hat, sp_hat)
    
    mct_regression.add_feedback(sample_action, mct_reward)
    sp_regression.add_feedback(sample_action, sp_reward)

mct_optimizer.setup(
    action_space = mct_regression.action_space,
    likelihood   = mct_regression.likelihood
)
mct_optimizer.fit(method='trust-constr', options={'disp': False})

sp_optimizer.setup(
    action_space = sp_regression.action_space,
    likelihood   = sp_regression.likelihood
)
sp_optimizer.fit(method='trust-constr', options={'disp': False})

fig, axs = plt.subplots(ncols=3, figsize=(15, 5))
x = mct_regression.action_space

pareto_ax, mct_ax, sp_ax = axs

pareto_ax.plot(*mo_reward(mct(x), sp(x)), label='Ground Truth Pareto Front', color='blue')
pareto_ax.scatter(
    mct_regression.feedback_data[:, 1],
    sp_regression.feedback_data[:, 1],
    color='red', label='Feedback Data'
)

pareto_ax.plot(mct_optimizer.mu, sp_optimizer.mu, label='GP Mean Pareto', color='green')

# Uncertainty band: each action is a point (mct_mu, sp_mu) with an independent
# std in each objective. Fill between the (mu + sigma) and (mu - sigma)
# parametric curves to get a ribbon around the mean Pareto front.
mct_std = mct_optimizer.std()
sp_std = sp_optimizer.std()
upper_x, upper_y = mct_optimizer.mu + mct_std, sp_optimizer.mu + sp_std
lower_x, lower_y = mct_optimizer.mu - mct_std, sp_optimizer.mu - sp_std
band_x = np.concatenate([upper_x, lower_x[::-1]])
band_y = np.concatenate([upper_y, lower_y[::-1]])
pareto_ax.fill(band_x, band_y, color='green', alpha=0.2, label='GP ±σ')

pareto_ax.set_xlabel('Obj 1: MCT Reward')
pareto_ax.set_ylabel('Obj 2: SP Reward')
pareto_ax.legend()

plr.plot_gp_1d(
    mct_ax, 
    mct_optimizer, 
    mct_regression, 
    ground_truth=lambda x: mo_reward(mct(x), sp(x))[0]
)

plr.plot_gp_1d(
    sp_ax, 
    sp_optimizer, 
    sp_regression, 
    ground_truth=lambda x: mo_reward(mct(x), sp(x))[1]
)

fig.tight_layout()
name = 'mo_gp_fit.pdf'
fig.savefig(name)   
print(f'Saved figure to {name}')