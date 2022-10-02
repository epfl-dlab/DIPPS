"""
This script illustrate the preprocessing of web-visits dataset from http://archive.ics.uci.edu/ml/datasets/msnbc.com+anonymous+web+data
The website visit patterns are categorized into three types: no visit, single visit and multiple visit.
The bbs visit count is used in the experiment since we assume the usage of bbs might be a good indicator of whether a person is 
willing to participate into data collections or not.
"""

import os
import torch
import numpy as np
import pandas as pd

DATASET_FN = "./msnbc990928.seq"
DIV_COL = "bbs"
OPT_FN = f"../data/web_visits/opt.pt"
NON_OPT_FN= f"../data/web_visits/non_opt.pt"

def subsample(features, page_types, col_name = "tech", opt_size = 10000, non_opt_size = 2082, batch_size = 256):
    """
    Subsample in the dataset to keep the opt and non-opt more balanced
    """

    col_idx = page_types.index(col_name)
    used_cols = [i for i in range(len(page_types)) if i != col_idx]
    
    # Get opt and non_opt indices
    opt_indices = np.asarray(torch.where(features[:, col_idx] >= 0)[0])
    non_opt_indices = np.asarray(torch.where(features[:, col_idx] < 0)[0])
    
    # Conduct random subsampling
    opt_indices = opt_indices[np.random.permutation(len(opt_indices))[: opt_size]]
    non_opt_indices = non_opt_indices[np.random.permutation(len(non_opt_indices))[: non_opt_size]]
    
    # Get the opt and non_opt data
    opt_data = features[opt_indices][:, used_cols]
    non_opt_data = features[non_opt_indices][:, used_cols]
    
    # Truncate the data size to multiples of batch size
    opt_size = opt_data.size(0) 
    non_opt_size = non_opt_data.size(0)
    
    opt_data = opt_data[: opt_size]
    non_opt_data = non_opt_data[: non_opt_size]
    
    return opt_data, non_opt_data

if __name__ == "__main__":

    with open(DATASET_FN, "r") as f:
        data = f.readlines()

    # Get page types
    page_types = data[2].rstrip().split(" ")

    # Get page visits data
    page_visits = list(map(lambda r: r.rstrip().split(" "), data[7: ]))

    # Get unique page visits
    unique_page_visits = list(map(lambda l: list(map(int, l)), page_visits))

    # Get count of users and count of features
    all_count = len(unique_page_visits)
    feat_count = len(page_types)

    unique_page_visits = unique_page_visits[: all_count]

    # Build feature vector by dividing visit patterns into three types
    hi = 2
    features = torch.zeros(all_count, feat_count)
        
    for i in range(all_count):
        freq = dict(zip(*np.unique(unique_page_visits[i], return_counts = True)))
        
        for key, value in freq.items():
            features[i][key - 1] = min(float(value), hi)

    # Obtain opt and non-opt with similar size
    opt, non_opt = subsample(features = features, page_types = page_types, col_name = DIV_COL)

    # Save the opt and non-opt
    torch.save(opt, OPT_FN)
    torch.save(non_opt, NON_OPT_FN)