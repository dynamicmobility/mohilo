import pypolar as plr
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt

rng = np.random.default_rng(95)

# GP Regression
optimizer = plr.BasicGP(
    kernel            = 'squared_exp',
    signal_variance   = 10.0,
    length_scale      = 1.0,
    mu_init_method    = 'random',
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
mct = plr.oracles.rewards.IdealPoint(
    w     = np.array([2.0]),
    delta = 1.0,
    gamma = 1.0
)
W_MCT = 1.0
mct_oracle = plr.oracles.NoisyRegressionOracle(
    reward_fn = mct,
    rng       = rng,
    noise_std = 0.0
)

sp = plr.oracles.rewards.IdealPoint(
    w     = np.array([-1.5]),
    delta = 1.0,
    gamma = 1.0
)
W_SP = 1.0
sp_oracle = plr.oracles.NoisyRegressionOracle(
    reward_fn = sp,
    rng       = rng,
    noise_std = 0.0,
)

def mo_reward(mct_hat, sp_hat, w):
    return np.dot(w, np.array([mct_hat, 1 / sp_hat])**2)
scalarized_reward = lambda mct_hat, sp_hat: mo_reward(mct_hat, sp_hat, np.array([W_MCT, W_SP]))

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
optimizer.setup(
    action_space = regression.action_space,
    likelihood   = regression.likelihood
)
optimizer.fit(method='trust-constr', options={'disp': False})
    
fig, ax = plt.subplots()
x = regression.action_space
y = scalarized_reward(mct(x), sp(x))
ax.plot(x.ravel(), y, label='Ground Truth', color='blue')

sampled_actions = regression.action_space[regression.feedback_data[:, 0].astype(int)]
ax.scatter(
    sampled_actions, 
    scalarized_reward(mct(sampled_actions), sp(sampled_actions)), 
    color='red', label='Feedback Data'
)

ax.plot(x.ravel(), optimizer.mu, label='GP Mean', color='green')
ax.fill_between(x.ravel(), optimizer.mu - 1 * optimizer.std(), optimizer.mu + 1 * optimizer.std(), color='green', alpha=0.2, label='GP ±σ')

ax.set_xlabel('Action')
ax.set_ylabel('Reward')
ax.legend()
fig.tight_layout()
name = 'gp_fit.pdf'
fig.savefig(name)   
print(f'Saved figure to {name}')