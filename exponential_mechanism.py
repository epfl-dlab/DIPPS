import numpy as np
import matplotlib.pyplot as plt

def exponential_mechanism(utility, epsilon = 1):
    """
    This function apply exponential mechanism to utility with specified epsilon and return the choice
    @param utility: (N, F) array with N vector with F dimension each
    @param epsilon: Diffrential privacy parameter
    @return chocie: N dimension choices
    """
    
    # Get number of classes
    C = utility.shape[1]
    
    # Apply exponential mechanism to utility
    utility = np.exp(epsilon * utility / 2)
    
    # Normalize utiltiy
    utility /= utility.sum(axis = 1).reshape(-1, 1)
    
    # Select according to the utility
    results = np.asarray([np.random.choice(np.arange(C), size = 1, p = dist) for dist in utility])
    
    return results

   