import numpy as np


class SquaredExponential:
    """Squared exponential (RBF) covariance function.

    k(a, b) = signal_variance^2 * exp(-||a - b||^2 / (2 * length_scale^2))
    """

    def __init__(self, signal_variance=1.0, length_scale=1.0):
        """
        Args:
            signal_variance: dictates the expected variation in the latent reward
            length_scale: dictates the smoothness of the latent reward
        """
        self.signal_variance = signal_variance
        self.length_scale = length_scale

    def __call__(self, A, B):
        """Cross-covariance k(A, B) for an (n, d) and (m, d) set of points.

        Args:
            A: (n, d) array of points
            B: (m, d) array of points

        Returns:
            (n, m) covariance matrix.
        """
        sqdist = (
            np.sum(A**2, axis=1)[:, None]
            + np.sum(B**2, axis=1)
            - 2 * np.dot(A, B.T)
        )
        return self.signal_variance**2 * np.exp(
            -0.5 * sqdist / self.length_scale**2
        )

    def diag_var(self):
        """Prior variance k(x, x), identical for every point."""
        return self.signal_variance**2

    def key(self):
        """Hashable identity of the hyperparameters, for cache invalidation."""
        return ("squared_exp", self.signal_variance, self.length_scale)


KERNELS = {
    "squared_exp": SquaredExponential,
}


def make_kernel(kernel, signal_variance=1.0, length_scale=1.0):
    """Build a kernel object from a name, or pass an existing one through.

    Args:
        kernel: a kernel name (e.g. ``'squared_exp'``) or a kernel instance
        signal_variance: signal variance, when building from a name
        length_scale: length scale, when building from a name

    Returns:
        A kernel object implementing ``__call__``, ``diag_var`` and ``key``.
    """
    if not isinstance(kernel, str):
        return kernel
    if kernel not in KERNELS:
        raise ValueError(
            f"Unknown kernel: {kernel!r}. Available: {sorted(KERNELS)}"
        )
    return KERNELS[kernel](
        signal_variance=signal_variance, length_scale=length_scale
    )
