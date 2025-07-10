import numpy as np
from scipy.stats import qmc

def generate_lhs_samples(bounds: dict, n_samples: int) -> np.ndarray:
    """
    Generate Latin Hypercube Samples as a NumPy array.

    Parameters:
    - bounds: dict with parameter names as keys and (min, max) tuples as values.
    - n_samples: int, number of samples to generate.
    - seed: int, optional seed for reproducibility.

    Returns:
    - samples: (n_samples x n_parameters) NumPy array with scaled samples.
    """
    param_names = list(bounds.keys())
    lower_bounds = np.array([bounds[p][0] for p in param_names])
    upper_bounds = np.array([bounds[p][1] for p in param_names])
    
    # Create sampler without seed parameter
    sampler = qmc.LatinHypercube(d=len(bounds))
    unit_samples = sampler.random(n=n_samples)
    scaled_samples = qmc.scale(unit_samples, lower_bounds, upper_bounds)

    return scaled_samples  # shape: (n_samples, n_parameters)




