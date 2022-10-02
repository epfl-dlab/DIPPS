"""
This script implements Laplace mechianism
"""

import torch
from copy import deepcopy

# Define Laplace mechanism
def laplace_mechanism(x, delta_x, epsilon = 1):
    """
    Add laplacian noise with scale = delta_x / epsilon to input
    @return noisy_x
    """
    
    noisy_x = torch.FloatTensor(deepcopy(x))
    scale = delta_x / epsilon
           
    laplace_dist = torch.distributions.Laplace(0, scale)
    noisy_x += laplace_dist.sample(noisy_x.size())

    return noisy_x
