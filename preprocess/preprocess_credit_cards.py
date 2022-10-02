"""
This script preprocess the credit datasets from https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients
"""

import pandas as pd
import numpy as np
import torch
import torchvision

if __name__ == "__main__":

    # Read original datasets
    data_df = pd.read_csv("./UCI_Credit_Card.csv").dropna()

    # Select cols for experiments
    selected_cols = ["LIMIT_BAL", "SEX", "AGE", 'PAY_0',
        'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6', 'BILL_AMT1', 'BILL_AMT2',
        'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1',
        'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']

    # Use selected columns only
    reduced_data_df = data_df[selected_cols]

    # Preprocess the data df so that its range is between -1 and 1
    normalized_df = reduced_data_df.apply(lambda col: (col - col.min()) / (col.max() - col.min()) * 2 - 1, axis = 0)

    # Select opt and non-opt portion
    opt_data_df = normalized_df.merge(data_df[["default.payment.next.month"]], left_index = True, right_index = True)
    opt_data_df = opt_data_df[opt_data_df["default.payment.next.month"] == 0]

    nonopt_data_df = normalized_df.merge(data_df[["default.payment.next.month"]], left_index = True, right_index = True)
    nonopt_data_df = nonopt_data_df[nonopt_data_df["default.payment.next.month"] == 1]

    # Save the data
    opt = torch.FloatTensor(opt_data_df.values)
    torch.save(opt, "../data/credit_cards/opt.pt")
    non_opt = torch.FloatTensor(nonopt_data_df.values)
    torch.save(non_opt, "../data/credit_cards/non_opt.pt")
