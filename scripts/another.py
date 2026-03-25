import numpy as np
from scipy.optimize import minimize

def squared_exponential_kernel(x1, x2, length_scale=1.0):
    """
    Squared Exponential (RBF) Kernel.
    """
    x1 = np.atleast_2d(x1)
    x2 = np.atleast_2d(x2)
    sq_dist = np.sum((x1[:, np.newaxis, :] - x2[np.newaxis, :, :]) ** 2, axis=-1)
    return np.exp(-sq_dist / (2 * length_scale ** 2))

def negative_log_marginal_likelihood(params, X, y):
    """
    Compute the negative log marginal likelihood for the Gaussian Process.
    """
    length_scale = params[0]
    noise_variance = params[1]

    # Kernel matrix with noise term added to the diagonal
    K = squared_exponential_kernel(X, X, length_scale) + noise_variance * np.eye(len(X))

    # Compute the negative log marginal likelihood
    L = np.linalg.cholesky(K)  # Cholesky decomposition of K
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))  # (K^-1) * y

    # Compute the log determinant and quadratic term
    log_det_K = 2 * np.sum(np.log(np.diag(L)))
    nll = 0.5 * y.T @ alpha + 0.5 * log_det_K + 0.5 * len(X) * np.log(2 * np.pi)
    return nll

def predict(X_train, y_train, X_test, length_scale, noise_variance):
    """
    Predict the mean and variance of the Gaussian process at test points.
    """
    K = squared_exponential_kernel(X_train, X_train, length_scale) + noise_variance * np.eye(len(X_train))
    K_s = squared_exponential_kernel(X_train, X_test, length_scale)
    K_ss = squared_exponential_kernel(X_test, X_test, length_scale)

    L = np.linalg.cholesky(K)
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_train))

    # Predictive mean
    mu = K_s.T @ alpha

    # Predictive variance
    v = np.linalg.solve(L, K_s)
    var = np.diag(K_ss - v.T @ v)

    return mu, var

# Generate synthetic data
np.random.seed(42)
X_train = np.random.uniform(-3, 3, (10, 1))
y_train = np.sin(X_train) + 0.1 * np.random.randn(10, 1)

# Optimize hyperparameters (length scale and noise variance)
initial_params = [1.0, 0.1]  # Initial guess for length_scale and noise_variance
res = minimize(negative_log_marginal_likelihood, initial_params, args=(X_train, y_train), bounds=[(1e-5, None), (1e-5, None)])

# Optimal hyperparameters
length_scale_opt, noise_variance_opt = res.x
print(f"Optimized length scale: {length_scale_opt:.3f}")
print(f"Optimized noise variance: {noise_variance_opt:.3f}")

# Predict on test points
X_test = np.linspace(-3, 3, 100).reshape(-1, 1)
mu, var = predict(X_train, y_train, X_test, length_scale_opt, noise_variance_opt)
# Plot the results
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(X_train, y_train, 'ro', label='Training data')
plt.plot(X_test, np.sin(X_test), 'g--', label='True function')
plt.plot(X_test, mu, 'b-', label='GP mean prediction')
# plt.fill_between(X_test.ravel(), mu - 2 * np.sqrt(var), mu + 2 * np.sqrt(var), color='blue', alpha=0.2, label='Confidence interval (2 std)')
plt.legend()
plt.title('Gaussian Process Regression with Squared Exponential Kernel')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
