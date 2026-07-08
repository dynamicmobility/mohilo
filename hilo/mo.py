import pypolar as plr
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def plot_gp_1d(ax, gp, regression: plr.Regression, ground_truth=None, num_std=1.0):
    x = regression.action_space
    if x.shape[1] != 1:
        raise ValueError(
            f"plot_gp_1d only supports 1D action spaces, got shape {x.shape}"
        )

    xs = x.ravel()
    norm_xs = xs / (xs.max() - xs.min())
    colors = np.array([1 - norm_xs, norm_xs, np.zeros_like(norm_xs)]).T

    # Ground truth reference curve (optional)
    if ground_truth is not None:
        ax.plot(xs, ground_truth(x), label='Ground Truth', color='grey')
        # ax.scatter(xs, ground_truth(x), c=colors, s=5)

    # Observed feedback points
    fb = regression.feedback_data
    if fb.size > 0:
        idx = fb[:, 0].astype(int)
        ax.scatter(x[idx].ravel(), fb[:, 1], color='blue', label='Feedback Data', s=40, zorder=3)

    # GP posterior mean and uncertainty band
    mu = gp.mu
    std = gp.std()
    ax.scatter(xs, mu, label='GP Mean', c=colors, s=15, zorder=2)
    ax.fill_between(
        xs, mu - num_std * std, mu + num_std * std,
        color='grey', alpha=0.1, label=f'GP ±{num_std:g}σ'
    )

    ax.set_xlabel('Action')
    ax.set_ylabel('Reward')
    ax.legend()
    return ax

def plot_pareto_2d(
    ax               : plt.Axes, 
    gps              : list[plr.Regression],
    regressions      : list[plr.Regression],
    mo_ground_truth,
):
    x = regressions[0].action_space
    xs = x.ravel()
    norm_xs = xs / (xs.max() - xs.min())
    colors = np.array([1 - norm_xs, norm_xs, np.zeros_like(norm_xs)]).T

    ax.plot(*mo_ground_truth(x), label='Ground Truth Pareto Front', color='grey')

    ax.scatter(
        mct_regression.feedback_data[:, 1],
        sp_regression.feedback_data[:, 1],
        color='blue', 
        label='Feedback Data',
        s=40,
        zorder=3
    )

    ax.scatter(
        gps[0].mu, 
        gps[1].mu, 
        label='GP Mean Pareto', 
        c=colors,
        s=15,
        zorder=2
    )
    ax.legend()


rng = np.random.default_rng(95)

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

# GP Regression
mct_optimizer = plr.BasicGP(
    kernel            = 'squared_exp',
    signal_variance   = 10.0,
    length_scale      = 1.0,
    x0_init_method    = 'random',
    rng               = rng
)
sp_optimizer = plr.BasicGP(
    kernel            = 'squared_exp',
    signal_variance   = 10.0,
    length_scale      = 1.0,
    x0_init_method    = 'random',
    rng               = rng
)

# Sampling setup
sampler = plr.DSTSampler(gps=[mct_optimizer, sp_optimizer], rng=rng, rho=0.05)

# Groundtruth oracles
mct = plr.BoundedIdealPoint(
    w     = np.array([2.0]),
    delta = 1.0,
    gamma = 0.0,
    lower_bound = 0.0,
    upper_bound = 8.0
)
W_MCT = 1.0
mct_oracle = plr.NoisyRegressionOracle(
    reward_fn = mct,
    rng       = rng,
    noise_std = 0.0
)

sp = plr.BoundedIdealPoint(
    w     = np.array([1.0]),
    delta = 1.0,
    gamma = 0.0,
    lower_bound = 0.0,
    upper_bound = 8.0
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
NUM_QUERIES = 20
feedback_data = []
for i in tqdm(range(NUM_QUERIES)):
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
    sp_optimizer.setup(
        action_space = sp_regression.action_space,
        likelihood   = sp_regression.likelihood
    )

    mct_optimizer.fit(method='trust-constr', options={'disp': False})
    sp_optimizer.fit(method='trust-constr', options={'disp': False})
    
    sampler.update_posterior()

fig, axs = plt.subplots(ncols=3, figsize=(15, 5))
x = mct_regression.action_space

pareto_ax, mct_ax, sp_ax = axs

plot_pareto_2d(
    ax                = pareto_ax,
    gps               = [mct_optimizer, sp_optimizer],
    regressions       = [mct_regression, sp_regression],
    mo_ground_truth   = lambda x: mo_reward(mct(x), sp(x))
)
pareto_ax.set_xlabel('Obj 1: MCT Reward')
pareto_ax.set_ylabel('Obj 2: SP Reward')

plot_gp_1d(
    mct_ax, 
    mct_optimizer, 
    mct_regression, 
    ground_truth=lambda x: mo_reward(mct(x), sp(x))[0]
)
mct_ax.set_title('Objective 1: MCT Reward')

plot_gp_1d(
    sp_ax, 
    sp_optimizer, 
    sp_regression, 
    ground_truth=lambda x: mo_reward(mct(x), sp(x))[1]
)
sp_ax.set_title('Objective 2: Speed Reward')

fig.tight_layout()
name = 'mo_gp_fit.pdf'
fig.savefig(name)   
print(f'Saved figure to {name}')