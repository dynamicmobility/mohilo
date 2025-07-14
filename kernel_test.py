import numpy as np
from gp import BasicGP

def squared_exponential_kernel(x1, x2, length_scale=1.0):
    """
    Compute the squared exponential kernel between two input vectors or matrices.

    Parameters:
        x1 (ndarray): First input array of shape (n, d), where n is the number of samples and d is the dimensionality.
        x2 (ndarray): Second input array of shape (m, d), where m is the number of samples and d is the dimensionality.
        length_scale (float): The length scale parameter of the kernel. Default is 1.0.

    Returns:
        ndarray: Kernel matrix of shape (n, m).
    """
    # Ensure input arrays are numpy arrays
    x1 = np.atleast_2d(x1)
    x2 = np.atleast_2d(x2)

    # Compute the squared Euclidean distance between each pair of points
    sq_dist = np.sum((x1[:, np.newaxis, :] - x2[np.newaxis, :, :]) ** 2, axis=-1)

    # Compute the squared exponential kernel
    kernel_matrix = np.exp(-sq_dist / (2 * length_scale ** 2))

    return kernel_matrix


# Example usage:
x1 = np.array([[1.0, 2.0], [3.0, 4.0]])  # 2 samples, 2 dimensions
x1 = np.array([[1.0, 2.5], [3.5, 4.0]])  # 2 samples, 2 dimensions

kernel_matrix = squared_exponential_kernel(x1, x1, length_scale=1.0)
print("Kernel matrix:\n", kernel_matrix)

gp = BasicGP(lengthscale=1.0, signal_var=1)
gp.setup(x1, None) # assign the action space
                                                   # and likelihood function 
                                                   # from the PBL object
my_matrix = gp.squared_exp_kernel(x1)
print("My matrix:\n", my_matrix)
