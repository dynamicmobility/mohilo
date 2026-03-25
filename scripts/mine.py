import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# Given
N = 100
np.random.seed(0)
t = np.linspace(0, 7, N)
f = np.sin(t)
A = np.eye(N)  # Identity matrix

# Sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def dsigmoid(x):
    sig = sigmoid(x)
    return sig * (1 - sig)

def g(e):
    return 1 / sigmoid(e.T @ e)

# Objective function
def objective(r):
    e = (f - r)
    return -np.log(g(e))
    # return sigmoid(e.T @ e)

# Gradient of the objective function
def gradient(r):
    e = f - r
    return -1 / g(e) * -g(e)**2 * dsigmoid(e.T @ e) * 2 * e * -1 
    # return dsigmoid(e.T @ e) * 2 * e * -1

# Hessian of the objective function
def hessian(r):
    return 2 * A  # Simplifies to 2 * I when A = I

# Initial guess for r
r0 = np.random.rand(N)

def h(x):
    return x.T @ A @ x

def dh(x):
    return 2 * A @ x


def check_gradient(obj, jac):
    # Check the gradient
    eps = 1e-6
    r = np.random.rand(N)
    grad = jac(r)
    approx_grad = np.zeros(N)
    for i in range(N):
        r_plus = r.copy()
        r_plus[i] += eps
        r_minus = r.copy()
        r_minus[i] -= eps
        approx_grad[i] = (obj(r_plus) - obj(r_minus)) / (2 * eps)
    print("Gradient error:", np.linalg.norm(grad - approx_grad))
    
check_gradient(objective, gradient)
check_gradient(h, dh)

# Minimization
result = minimize(
    objective,
    r0,
    method='BFGS',  # Choose a method that supports Hessians
    jac=gradient,          # Provide the gradient
    options={'disp': True}
)

print("Optimal r:", objective(result.x))
print("Optimal value:", objective(f))
fig, ax = plt.subplots(nrows=1, ncols=2)
# Plotting the results
ax[0].plot(t, f, label='Original f')
ax[0].set_title('Original Function f')
ax[0].legend()

ax[1].plot(t, result.x, label='Optimized r')
ax[1].set_title('Optimized Function r')
ax[1].legend()

plt.show()

"""
v = [x; y]
obj = sig((y - x) / o)
jac = ?

obj = sig(1/o * [-1 1]@v)
jac = dsig(1 / o * [-1 1]@v) * 1 / o * [-1 1]
"""

