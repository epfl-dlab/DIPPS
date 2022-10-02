import pandas as pd 
import numpy as np
from scipy.linalg import norm
def computeWassersteinDist(src_file, tgt_file, src_weight_file, tgt_weight_file, plan_file, p = 1):

    src = pd.read_csv(src_file)
    tgt = pd.read_csv(tgt_file)
    src_weight = pd.read_csv(src_weight_file)
    tgt_weight = pd.read_csv(tgt_weight_file)
    plan = pd.read_csv(plan_file)

    # src *= src_weight.values
    # tgt *= tgt_weight.values
    assert(tgt_weight.values.sum() == src_weight.values.sum())
    assert(tgt_weight.values.sum() == plan["mass"].sum())
    total_weight = src_weight.values.sum()
    print(total_weight)

    # Clean the columns
    src["index"] = range(1, src.shape[0] + 1)
    tgt["index"] = range(1, tgt.shape[0] + 1)

    # Merge plan with src and tgt
    result = plan.merge(src, left_on = "from", right_on = "index")
    result = result.merge(tgt, left_on = "to", right_on = "index")

    # Compute the distance
    l_values = result[[col for col in result.columns if col[-1] == 'x' and "index" not in col]].values
    r_values = result[[col for col in result.columns if col[-1] == 'y' and "index" not in col]].values

    res = np.power(np.sum(np.sum(np.power(np.abs((r_values - l_values)), p), axis = 1) * result["mass"].values / total_weight), 1 / p)
   
    return res

if __name__ == "__main__":

    src_file = "src.csv"
    tgt_file = "tgt.csv"
    src_weight_file = "src_weights.csv"
    tgt_weight_file = "tgt_weights.csv"
    plan_file = "plan.csv"

    res = computeWassersteinDist(src_file, tgt_file, src_weight_file, tgt_weight_file, plan_file)    
    print(res)