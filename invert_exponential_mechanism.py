"""
This script implement inverting the exponential mechanism
"""

import numpy as np

def invert_exponential_output(group_counts, epsilon):
    """
    This function take the output class counts of exponential mechanism and produce the reweighted distribution
    @param group_counts: count of each group
    @param epsilon: the differential privacy parameter used in the mechanism
    @return reweighted distribution
    """
    
    # Get number of class
    num_class = len(group_counts)
    
    # Get number of class with positive count
    num_pos_class = np.sum(group_counts > 0)
    
    # Get index of class with no samples
    absent_classes = group_counts <= 0
    
    # Get number of samples classified to the first class with non zero entry
    n_seed = group_counts[group_counts > 0][0]

    # Obtain u_i - u_0 for each i
    delta_ui_u0 = 2 / epsilon * np.log(group_counts / n_seed)
    delta_ui_u0[np.isneginf(delta_ui_u0)] = 0

    # Calculate probability of class seed
    p_seed = (1 - np.sum(delta_ui_u0)) / num_pos_class
    
    # Evaluate other classes' proba
    p = p_seed + delta_ui_u0
    
    # Handle absent class
    p[absent_classes] = 0
    
    # print(f"Before: {p}")
    p = np.where(p >= 0, p, 0)
    p /= np.sum(p)

    return p
