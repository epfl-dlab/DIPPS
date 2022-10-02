"""
Some utilization fuctions
"""

from pathlib import Path
from shutil import rmtree
import numpy as np

def l1_loss(x, y):
    return np.sum(np.abs(x - y))

def l2_loss(x, y):
    return np.sqrt(np.sum(np.square(x - y)))

def construct_experiment_dir(args, config):
    """
    Set appropriate name of experiment directory
    It is structured as /experiments/{feat_dim}_{n_component}/random_seed/
    """

    feat_dim = args.feat_dim
    n_component = args.n_component
    latent_dim = args.latent_dim
    exp_type = args.exp_type
    gmm_metric = args.gmm_metric
    pca_dim = args.pca_dim
    random_seed = args.random_seed
    experiment_prefix = args.data_dir.split("/")[-1]

    eval_range = config["EVALUATION"]["eval_range"]
    damping = "damping" if config.getboolean("MODEL", "damping") else "nodamping"
    return f"./experiments_wasserstein_{eval_range}/{exp_type}/{experiment_prefix}/PCA_{pca_dim}/metric_{gmm_metric}/{random_seed}"

l2_loss = lambda x, y: np.sqrt(np.sum(np.square(x - y)))

l1_loss = lambda x, y: np.sum(np.abs(x - y))

def initialize_dir(args, config):
    """
    Initialize output directory \n
    @param args: arguments \n
    @param config: configuration \n
    """
    
    # Construct experiment directory
    experiment_dir = construct_experiment_dir(args, config)
    rmtree(experiment_dir, ignore_errors = True)
    Path(experiment_dir).mkdir(parents = True, exist_ok = True)

    # Construct model directory
    model_dir = f"{experiment_dir}/model"
    Path(model_dir).mkdir(parents = True, exist_ok = True)

    # Construct figure directory
    figure_dir = f"{experiment_dir}/figure"
    Path(figure_dir).mkdir(parents = True, exist_ok = True)

def record_metric_results(stats: dict, n_component: int, pca_dim: int):
    """
    Record results for metric
    """

    results = dict()
    results["naive"] = stats["mean"][1]["naive"].item()
    results["LCIW"] = stats["mean"][1]["LCIW"].item()
    results["DPLCIW"] = stats["mean"][1]["DPLCIW mean"].item()
    results["n_component"] = n_component
    results["pca_dim"] = pca_dim
    return results
