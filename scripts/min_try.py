# # # try function fitting with scipy minimize
# # # scipy.optimize.curve_fit()
# # import numpy as np
# # import matplotlib.pyplot as plt
# # import scipy.optimize as opt

# # N = 100
# # x = np.linspace(0, 2 * np.pi, N)
# # r = np.sin(x)

# # def objective(f):
# #     return np.linalg.norm(f - r)

# # res = opt.minimize(objective, np.random.random(N))
# # f = res.x

# # # plt.plot(np.arange(N), r, label='r')
# # plt.plot(np.arange(N), f, label='f')
# # plt.legend()
# # plt.show()

# import numpy as np
# from scipy.optimize import minimize
# from scipy.linalg import cholesky, solve_triangular

# # Define the RBF kernel (squared exponential)
# def rbf_kernel(X1, X2, length_scale, variance):
#     """Compute the RBF kernel matrix."""
#     sqdist = np.sum(X1**2, 1).reshape(-1, 1) + np.sum(X2**2, 1) - 2 * np.dot(X1, X2.T)
#     return variance * np.exp(-0.5 * sqdist / length_scale**2)

# # Define the negative log marginal likelihood (to be minimized)
# def negative_log_marginal_likelihood(params, X, y):
#     """Negative log marginal likelihood function."""
#     length_scale, variance, noise_variance = params
#     K = rbf_kernel(X, X, length_scale, variance) + noise_variance * np.eye(len(X))
#     L = cholesky(K, lower=True)  # Cholesky decomposition
    
#     # Compute the log marginal likelihood
#     alpha = solve_triangular(L.T, solve_triangular(L, y, lower=True))
#     log_likelihood = (
#         -0.5 * np.dot(y.T, alpha)
#         - np.sum(np.log(np.diagonal(L)))
#         - (len(X) / 2) * np.log(2 * np.pi)
#     )
#     return -log_likelihood  # We negate because we want to minimize

# # Generate synthetic data (for testing)
# np.random.seed(42)
# X = np.random.uniform(-5, 5, (20, 1))  # Training inputs
# y = np.sin(X).ravel() + np.random.normal(0, 0.1, X.shape[0])  # Noisy outputs

# # Initial hyperparameter guess: [length_scale, variance, noise_variance]
# initial_params = [1.0, 1.0, 1e-1]

# # Minimize the negative log marginal likelihood
# res = minimize(
#     negative_log_marginal_likelihood,
#     initial_params,
#     args=(X, y),
#     bounds=((1e-5, None), (1e-5, None), (1e-5, None)),  # Bounds to keep values positive
#     method='L-BFGS-B'
# )

# # Print optimized hyperparameters
# optimized_length_scale, optimized_variance, optimized_noise_variance = res.x
# print("Optimized Length Scale:", optimized_length_scale)
# print("Optimized Variance:", optimized_variance)
# print("Optimized Noise Variance:", optimized_noise_variance)


# import matplotlib.pyplot as plt

# # Define the predictive mean and variance functions
# def gp_predict(X_train, y_train, X_test, length_scale, variance, noise_variance):
#     """Compute the GP predictive mean and variance for test inputs."""
#     K = rbf_kernel(X_train, X_train, length_scale, variance) + noise_variance * np.eye(len(X_train))
#     K_s = rbf_kernel(X_train, X_test, length_scale, variance)
#     K_ss = rbf_kernel(X_test, X_test, length_scale, variance)

#     L = cholesky(K, lower=True)  # Cholesky decomposition
#     alpha = solve_triangular(L.T, solve_triangular(L, y_train, lower=True))

#     # Predictive mean
#     mu_s = np.dot(K_s.T, alpha)

#     # Predictive variance
#     v = solve_triangular(L, K_s, lower=True)
#     cov_s = K_ss - np.dot(v.T, v)
#     std_s = np.sqrt(np.diag(cov_s))

#     return mu_s, std_s

# # Generate test points for prediction
# X_test = np.linspace(-5, 5, 100).reshape(-1, 1)

# # Predictive mean and standard deviation
# mu_s, std_s = gp_predict(X, y, X_test, optimized_length_scale, optimized_variance, optimized_noise_variance)

# # Plotting
# plt.figure(figsize=(10, 6))
# plt.plot(X, y, 'o', label="Training Data", markersize=8, color='navy')
# plt.plot(X_test, mu_s, label="Mean Prediction", color='darkorange')
# plt.fill_between(
#     X_test.ravel(),
#     mu_s - 1.96 * std_s,
#     mu_s + 1.96 * std_s,
#     color='lightblue',
#     alpha=0.5,
#     label="95% Confidence Interval"
# )
# plt.title("Gaussian Process Regression")
# plt.xlabel("Input (X)")
# plt.ylabel("Output (y)")
# plt.legend()
# plt.grid(True)
# plt.show()


import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

def fit(X, y, degree):
    poly = PolynomialFeatures(degree=degree)
    X_poly = poly.fit_transform(X)
    model = LinearRegression()
    model.fit(X_poly, y)
    return model, poly

# Generate some sample data
np.random.seed(0)
x = np.random.rand(100, 2)
y = 3 * np.sin(6 * x[:, 0]) + x[:, 1]**2 + np.random.randn(100) * 0.1

# Create polynomial features
poly = PolynomialFeatures(degree=8)
X_poly = poly.fit_transform(x)

# Fit the model
model = LinearRegression()
model.fit(X_poly, y)

# Predict on a grid
x1, x2 = np.meshgrid(np.linspace(0, 1, 10), np.linspace(0, 1, 10))
X_grid = np.c_[x1.ravel(), x2.ravel()]
y_pred = model.predict(poly.transform(X_grid))

# Plot the results
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x[:, 0], x[:, 1], y)
ax.plot_surface(x1, x2, y_pred.reshape(x1.shape), alpha=0.5)
plt.show()