"""
This script illustrate the process to preprocess the weather dataset from https://archive.ics.uci.edu/ml/datasets/Beijing+Multi-Site+Air-Quality+Data
"""

import os
import numpy as np
import pandas as pd
import torch

# Here we demonstrate the preprocessing procedure starting from the raw datasets to the final opt and non-opt files
# The sample opt and non-opt pairs are (Huairou, Nongzhanguan)
DATASET_DIR = "./PRSA_Data_20130301-20170228/"
NORMALIZED_DATASET_DIR = "./PRSA_Normalized"
OPT_LOCATION = "Huairou"
NONOPT_LOCATION = "Nongzhanguan"
TARGET_DIR = "../data/weather_1_inv"

selected_cols = ["PM2.5", "PM10", "SO2", "NO2", "CO", "O3", "TEMP", "PRES", "DEWP", "RAIN", "WSPM"]

if __name__ == "__main__":

    # Normalize the data
    max_rows = []
    min_rows = []

    # Get the maximum and minimum of all the data
    for i, file in enumerate(os.listdir(DATASET_DIR)):
        
        # Read in data
        data_df = pd.read_csv(f"./PRSA_Data_20130301-20170228/{file}").dropna()[selected_cols].copy()
        
        # Get range of the data
        max_rows.append(data_df.max(axis = 0))
        min_rows.append(data_df.min(axis = 0))
        
    global_min = pd.DataFrame(min_rows).transpose().min(axis = 1)
    global_max = pd.DataFrame(max_rows).transpose().max(axis = 1)

    # Store the normalized dataset
    for i, file in enumerate(os.listdir(DATASET_DIR)):

        # Read in data
        data_df = pd.read_csv(f"./PRSA_Data_20130301-20170228/{file}").dropna()[selected_cols].copy()
    
        # Normalize the data
        normalized_df = data_df.apply(lambda row: (row - global_min) / (global_max - global_min) * 2 - 1, axis = 1)

        # Save the normalized data
        normalized_df.to_csv(f"{NORMALIZED_DATASET_DIR}/{file}")

    # Load the selected opt and non-opt location and convert the normalized dataset to opt and non-opt 
    # dataset which will be fed to the experiment
    opt_file = os.path.join(NORMALIZED_DATASET_DIR, f"PRSA_Data_{OPT_LOCATION}_20130301-20170228.csv")
    non_opt_file = os.path.join(NORMALIZED_DATASET_DIR, f"PRSA_Data_{NONOPT_LOCATION}_20130301-20170228.csv")
    opt_df = pd.read_csv(opt_file)
    non_opt_df = pd.read_csv(non_opt_file)
    opt_count = opt_df.shape[0]
    non_opt_count = non_opt_df.shape[0] 
    opt = torch.FloatTensor(opt_df[selected_cols].iloc[: opt_count].values)
    non_opt = torch.FloatTensor(non_opt_df[selected_cols].iloc[: non_opt_count].values)

    # Store the opt and non-opt data to the target dir
    torch.save(opt, f"{TARGET_DIR}/opt.pt")
    torch.save(non_opt, f"{TARGET_DIR}/non_opt.pt")