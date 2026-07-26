
def derive_precision(noise_var: float):
    """Assumes the feedback data is Gaussian with variance ``var`` and returns 
    the corresponding precision ``lambda = 1/(2*var)`` for the likelihood term 
    ``lambda * ||Sr - y||^2``."""
    return 1.0 / (2.0 * noise_var)

def derive_prior_variance(expected_range: float):
    """Guesses the signal (prior) variance from the expected range of the 
    feedback data. Returns 1/2 * range"""
    return 0.5 * expected_range

def derive_lengthscale(domain_size: float):
    """Guesses the kernel lengthscale from the data domain. Returns 
    1/5 * range"""
    return 0.2 * domain_size

def derive_gp_hyperparams(domain_size, expected_range, noise_var):
    """Derives the GP hyperparameters from the data domain and expected range of 
    the feedback data. Returns a tuple of (lengthscale, prior variance, 
    precision)."""
    lengthscale    = derive_lengthscale(domain_size)
    prior_variance = derive_prior_variance(expected_range)
    precision      = derive_precision(noise_var)
    return lengthscale, prior_variance, precision