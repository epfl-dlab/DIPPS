"""
This is the implementation of the hybrid mechanism proposed in paper `Collectingand analyzing multidimensional data with local differential privacy.` which is compared with our proposed DiPPS method
"""

import numpy as np
import random
import math

EPS_STAR = np.log((-5 + 2 * np.power(6353 - 405 * np.sqrt(241), 1 / 3) + 2 * np.power(6353 + 405 * np.sqrt(241), 1 / 3)) / 27)

def piecewise_mechanism(t, epsilon) -> float:
    """
    Conduct piecewise mechanism
    @param t(float): input
    @param epsilon(float): LDP parameter
    @return perturbed output
    """
    
    x = random.uniform(0, 1)
    
    # Calculate a helper variable e^(eps / 2)
    exp_half_eps = np.exp(epsilon / 2)
    
    # Calculate the output bound
    C = (exp_half_eps + 1) / (exp_half_eps - 1) 
    l = (C + 1) / 2 * t - (C - 1) / 2
    r = l + C - 1

    # Calculate the perturbed output
    if x < exp_half_eps / (exp_half_eps + 1):
        t_star = random.uniform(l, r)
    else:
        _range = (l + C + C - r)
        t_star = np.random.choice([np.random.uniform(-C, l), -np.random.uniform(-C, -r)],
                                    p = [(l + C) / _range, (C - r) / _range])
        
    return t_star

def duchi_mechanism(t, epsilon) -> float:
    """
    Conduct duchi's mechanism
    @param t(float): input
    @param epsilon(float): LDP parameter
    @return perturbed output
    """
    
    # Calculate the probability of u=1
    proba_positive = (np.exp(epsilon) - 1) / (2 * np.exp(epsilon) + 2) * t + 0.5
    
    # Sample u
    u = np.random.choice([1, -1], p = [proba_positive, 1 - proba_positive])
    
    # Get perturbed output
    t_star = (np.exp(epsilon) + 1) / (np.exp(epsilon) - 1)

    if u == 1:
        return t_star
    else:
        return -t_star

def hybrid_mechanism(t, epsilon):
    """
    Conduct hybrid mechanism combinging duchi's method and pm
    @param t(float): input
    @param epsilon(float): LDP parameter
    @return perturbed output
    """
    
    alpha = 1 - np.exp(-epsilon / 2) if epsilon > EPS_STAR else 0
    
    coin = np.random.choice([1, -1], p = [alpha, 1 - alpha])
    
    if coin == 1:
        return piecewise_mechanism(t, epsilon)
    else:
        return duchi_mechanism(t, epsilon)
        

def multi_dim_hybrid_mechanism(t, epsilon):
    """
    Conduct multi dimensional version of hybrid mechanism
    @param t([float]): input
    @param epsilon(float): LDP parameter
    @return perturbed outputs
    """

    d = len(t)
    t_star = [0] * d
    
    # Compute number of values to return with non-zero perturbed results
    k = max(1, min(d, math.floor(epsilon / 2.5 ) ) )
    
    # Sample k entries to perturb
    perturbed_indices = np.random.permutation(d)[: k]
    
    # Perturb selected entries
    for i in perturbed_indices:
        t_star[i] = hybrid_mechanism(t[i], epsilon / k) * d / k
    
    return t_star
