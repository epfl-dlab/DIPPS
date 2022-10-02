import configparser
from argparse import ArgumentParser
from util import *
import numpy as np
from generator.gmm_generator import GMM
from generator.gmm_generator import visualize_generator
from generator.common_generator import Generator
from autoencoder import AutoEncoder
from autoencoder import GaussianMixtureEncoder
import torch
from torch import nn, optim
from time import time
from collections import defaultdict
from estimation import estimate
import pickle
import pathlib
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from copy import copy

import json
import matplotlib.pyplot as plt 

def parse():
    """
    Read command line arguments
    Load configuration
    """

    # Read command line arguments
    parser = ArgumentParser(description='Estimation of target variables of non-opt data using opt data')
    parser.add_argument("--random_seed", default = 1111, type = int, help = "random seed")
    parser.add_argument("--feat_dim", default = 10, type = int, help = "data feature dimension")
    parser.add_argument("--n_component", default = 5, type = int, help = "number of gaussian mixture components")
    parser.add_argument("--latent_dim", default = 5, type = int, help = "latent feature dimension")
    parser.add_argument("--config_file", default = "config/config.cfg", type = str, help = "configuration file")
    parser.add_argument("--epsilon", default = 1.0, type = float, help = "differential private parameter")
    parser.add_argument("--gmm_metric", default = "default", type = str, help = "Metric to select gaussian mixture")
    parser.add_argument("--exp_type", default = "trial", type = str, help = "Target of experiment")
    parser.add_argument("--pca_dim", default = 2, type = int, help = "pca dimension")
    parser.add_argument("--data_dir", default = "./data/credit_cards", type = str, help = "data_dir")
    parser.add_argument("--n_latent_component", default = 0, type = int, help = "number of latent component")

    parser.add_argument("--diff_thres", default = 1.0, type = float, help = "difference threshold in gaussian mixture synthetic dataset")
    args = parser.parse_args()

    # Read configuration file
    print(f"The configuration file is {args.config_file}")
    config = configparser.ConfigParser()
    config.read(args.config_file)

    # Log the settings
    print(f"The random seed is {args.random_seed}")
    print(f"The feature dim is {args.feat_dim}")
    print(f"The number of component of GMM is {args.n_component}")
    print(f"The latent dimension is {args.n_component}")
    print(f"The metric to select gaussian mixture is {args.gmm_metric}")
    print(f"The experiment type is {args.exp_type}")
    # Initialize output folder
    initialize_dir(args, config)

    return args, config

def normalize_data(x, data_max, data_min):
    """
    Normalize data to [-1, 1]
    """
    
    x -= data_min
    x /= (data_max - data_min)
    x *= 2
    x -= 1
    
    return x

def read_data_generator(args, config):
    """
    Read data generator from existing source
    @param args
    @param config
    @return train_generator
    @return test_generator
    """

    data_dir = args.data_dir
    opt_data = torch.load(f"{data_dir}/opt.pt")
    non_opt_data = torch.load(f"{data_dir}/non_opt.pt")

    # Randomly select some trivial columns as labels
    opt_generator = Generator(opt_data, opt_data[:, 0], unnormalized_features = opt_data)
    non_opt_generator = Generator(non_opt_data, non_opt_data[:, 0], unnormalized_features = non_opt_data)
    return opt_generator, non_opt_generator

def construct_data_generator(args, config):
    """
    Construct data generator for train/valid/test data
    @param args
    @param config
    @return train_generator(GMM)
    @return valid_generator(GMM)
    @return test_generator(GMM)
    """

    # Obtain settings for data generation
    feat_dim = args.feat_dim
    n_component = args.n_component
    random_seed = args.random_seed
    diff_thres = config.getfloat("DATA", "diff_thres")
    diff_thres = args.diff_thres        # Temporary modification to speed up the experiments
    gmm_mean_range = config["DATA"]["mean"]
    gmm_mean_range = [float(i) for i in gmm_mean_range.split(",")]
    batch_size = config.getint("TRAINING", "batch_size")
    n_train_batch = config.getint("TRAINING", "n_train_batch")
    n_valid_batch = config.getint("TRAINING", "n_valid_batch")
    n_test_batch = config.getint("TRAINING", "n_test_batch")

    # Set common mean for all gmm
    means = [np.random.choice(gmm_mean_range, size = feat_dim) for i in range(n_component)]

    # Get the sample sizes
    sample_sizes = {"train": batch_size * n_train_batch, "valid": batch_size * n_valid_batch,
                    "test": batch_size * n_test_batch}

    # Set weights over components of train/valid/test data
    np.random.seed(random_seed)
    while True:
        while True:
            
            train_weights = np.random.uniform(low = 0.05, high = 1.0, size = n_component)
            train_weights /= np.sum(train_weights)

            valid_weights = np.random.uniform(low = 0.05, high = 1.0, size = n_component)
            valid_weights /= np.sum(valid_weights)

            if (diff_thres > 0):
                test_weights = np.random.uniform(low = 0.05, high = 1.0, size = n_component)
                test_weights /= np.sum(test_weights)
            else:
                test_weights = train_weights

            if l1_loss(train_weights, test_weights) >= diff_thres and l1_loss(train_weights, test_weights) < diff_thres + 0.02:
                break

        # Define data generator
        train_source = GMM(K = n_component, D = feat_dim, mean = means, weights = train_weights)

        valid_source = GMM(K = n_component, D = feat_dim, mean = means,
                            covariance = train_source.covariance, weights = valid_weights)

        test_source = GMM(K = n_component, D = feat_dim, mean = means,
                            covariance = train_source.covariance, weights = test_weights)
                
        # Get data
        train_data, train_cluster = train_source.sample(sample_sizes["train"])

        valid_data, valid_cluster = valid_source.sample(sample_sizes["valid"])

        test_data, test_cluster = test_source.sample(sample_sizes["test"])

        # Get data max and min
        data_max = np.asarray([train_data.max(axis = 0), valid_data.max(axis = 0), test_data.max(axis = 0)]).max(axis = 0)
        data_min = np.asarray([train_data.min(axis = 0), valid_data.min(axis = 0), test_data.min(axis = 0)]).min(axis = 0)

        # Normalize data to [-1, 1]
        train_data = normalize_data(train_data, data_max, data_min)
        valid_data = normalize_data(valid_data, data_max, data_min)
        test_data = normalize_data(test_data, data_max, data_min)

        # Craete generator
        train_generator = Generator(train_data, train_cluster, unnormalized_features = train_data)
        valid_generator = Generator(valid_data, valid_cluster, unnormalized_features = valid_data)
        test_generator = Generator(test_data, test_cluster, unnormalized_features = test_data)

        # Enfoce big l1 norm difference between train and test data
        train_generator.reset_randomness()
        test_generator.reset_randomness()

        break
    
    # Save the data to datadir
    data_dir = args.data_dir
    pathlib.Path(data_dir).mkdir(parents = True, exist_ok = True)
    torch.save(train_generator.features, f"{data_dir}/opt.pt")
    torch.save(test_generator.features, f"{data_dir}/non_opt.pt")

    return train_generator, valid_generator, test_generator

def train(model, criterion, train_generator, valid_generator, args, config):
    """
    Train the autoencoder
    @param model(AutoEncoder)
    @param criterion(nn.loss)
    @param args
    @param config
    """

    # Read the settings
    random_seed = args.random_seed
    epsilon = args.epsilon
    epochs = config.getint("TRAINING", "epochs")
    batch_size = config.getint("TRAINING", "batch_size")
    n_train_batch = config.getint("TRAINING", "n_train_batch")
    n_valid_batch = config.getint("TRAINING", "n_valid_batch")
    damping = config.getboolean("MODEL", "damping")
    lr = config.getfloat("MODEL", "lr")

    # Define the optimizer 
    current_time = time()
        
    for e in range(epochs):

        # Ugly learning rate transition
        if e > 50:
            # Decrease weights in late training procedure
            for g in model.solver.param_groups:
                g["lr"] = 0.001
                
        # Reset randomness of data generator and loss
        train_generator.reset_randomness()
        # valid_generator.reset_randomness()
        running_loss = 0
        
        # Train through batch
        model.train()
        for i in range(n_train_batch):
            
            model.solver.zero_grad()
            
            # Obtain next batch of data
            x_train, y_train = train_generator.next_batch()

            # Convert input to expected output
            y_train = x_train.clone()
            
            # Forward pass
            if damping:
                y_train_pred = model(x_train, epsilon = epsilon)
            else:
                y_train_pred = model(x_train)

            # Calculate loss

            loss = criterion(y_train_pred, y_train)
            running_loss += loss
            
            # Back propagation
            loss.backward()
            model.solver.step()
        
        # # Validation
        # model.eval()
        # with torch.no_grad():
            
        #     valid_running_loss = 0
        #     for _ in range(n_valid_batch):
                
        #         # Obtain next batch of valid data
        #         x_valid, y_valid = valid_generator.next_batch()
                
        #         y_valid = x_valid.data.clone()
                
        #         # Forward pass
        #         if damping:
        #             y_valid_pred = model(x_valid, epsilon = epsilon)
        #         else:
        #             y_valid_pred = model(x_valid)
                
        #         # Calculate loss
        #         valid_loss = criterion(y_valid_pred, y_valid)
        #         valid_running_loss += valid_loss
                
        # Log result
        if e % 50 == 0:
            print(f"Epoch {e}:")
            print(f"Training loss is {running_loss / n_train_batch}")
        # print(f"Validation loss is {valid_running_loss / n_valid_batch}")
    
def store_scores(scores, experiment_dir):
    with open("scores", "a+") as f:
        f.write(f"{experiment_dir}: {scores}\n")

def select_gaussian_mixture_encoder(train_generator, latent_sizes, metric = "default", pca_dim_prop = 2, experiment_dir = ""):
    """
    Select a proper latent size to fit train generator using ELBOW method
    @param train_generator
    @return model
    """

    # Decide pca dimension
    if pca_dim_prop != 0:
        pca_dim = pca_dim_prop
    else:
        pca_dim = decide_pca_dimension(train_generator)

    print(f"The selected PCA dimension is {pca_dim}")
    # Conduct ELBOW method to select the most proper n_component
    
    gmes = []
    gme_scores = []
    # indices = np.random.permutation(train_generator.size)
    # train_data = train_generator.unnormalized_features[: int(train_generator.size * .8)]
    # valid_data = train_generator.unnormalized_features[int(train_generator.size * .8): ]
    train_data = train_generator.unnormalized_features
    for latent_size in latent_sizes:

        # Initialize Gaussian Mixture Encoder
        gme = GaussianMixtureEncoder(0, 0, 0, latent_size, cov_type = "tied", pca_dim = pca_dim)
        
        # Fit gaussian mixture encoder
        gme.fit(train_data)
        
        # Obtain intra class variance and inter
        gme_scores.append(gme.compute_scores(train_data, metric = metric))
        
        gmes.append(gme)

    print(f"The scores are {gme_scores}")

    if experiment_dir != "":
        store_scores(gme_scores, experiment_dir)
    
    print(f"ELBOW select {latent_sizes[ELBOW(gme_scores)]}")
    # exit()
    gme_score_diffs = [gme_scores[i] - gme_scores[i - 1] if i >= 1 else 0 for i in range(len(gme_scores))]

    # Plot and save the gme scores
    # plot_save_gme_scores(gme_scores, experiment_dir)

    latent_size = latent_sizes[ELBOW(gme_scores)]
    # latent_size = latent_sizes[np.asarray(gme_scores).argmax()]
    # print(f"The gme scores are {gme_scores}")
    # print(f"The gme score diffs are {gme_score_diffs}")
    print(f"The selected latent size is {latent_size}")

    gme = GaussianMixtureEncoder(0, 0, 0, latent_size, cov_type = "tied", pca_dim = pca_dim)
    gme.fit(train_generator.unnormalized_features)
    return gme

def plot_save_gme_scores(gme_scores, experiment_dir):

    # Get the dataset
    dataset = Path(experiment_dir).parent.parent.parent.stem
    random_seed = Path(experiment_dir).stem

    fig = plt.figure(figsize = (8, 8))
    ax = fig.add_subplot(111)
    ax.scatter(range(3, 11), gme_scores)
    ax.plot(range(3, 11), gme_scores)
    ax.set_xlabel("n_component")
    ax.set_ylabel("scores")
    ax.set_title(dataset)

    tgt_dir = Path("./figures", dataset)
    tgt_dir.mkdir(parents = True, exist_ok = True)
    tgt_fn = Path(tgt_dir, random_seed + ".png")

    fig.savefig(tgt_fn)
    exit()

def ELBOW(gme_scores):
    """
    Use approximated second order derivatives to select the latent size
    @param gme_scores(np.array)
    """
    scores = copy(gme_scores)
    scores += [-10, -10, -10]
    print(scores)
    for i in range(len(gme_scores)):
        if (max(scores[i + 1], max(scores[i + 2], scores[i + 3])) - scores[i]) < .01:
            break

    return i
    
def decide_pca_dimension(train_generator):

    pca =  PCA(n_components = train_generator.unnormalized_features.size(1))
    pca.fit_transform(train_generator.unnormalized_features)
    scores = pca.explained_variance_ratio_.cumsum()
    print(scores)
    index = [i > .8 for i in scores].index(True) + 1
    return index
    
def evaluate(model, criterion, test_generator, args, config):
    """
    Evaluate the model performance on ptest set 
    @return test_acc(float): testing accuracy
    """

    # Read the settings
    random_seed = args.random_seed
    batch_size = config.getint("TRAINING", "batch_size")
    n_test_batch = config.getint("TRAINING", "n_test_batch")

    model.eval()
    with torch.no_grad():
        
        # Reset randomness of test generator
        test_generator.reset_randomness()
        
        running_test_loss = 0
        hits = torch.FloatTensor([0])
        
        for _ in range(n_test_batch):
            
            # Obtain next batch of test data
            x_test, y_test = test_generator.next_batch()
            
            # Forward pass
            y_test_pred = model(x_test)
            
            # Calculate loss
            loss = criterion(y_test_pred, y_test)
            running_test_loss += loss
            
            # Get number of hits
            hits += (y_test_pred.argmax(axis = 1) == y_test).sum()
            print(hits)

    # Log test loss and acc
    test_acc = hits / (n_test_batch * batch_size)
    print(f"The test loss is {running_test_loss / n_test_batch}")
    print(f"The test accuracy is {test_acc}")

    return test_acc

if __name__ == "__main__":

    # Read arguments
    args, config = parse()
    experiment_dir = construct_experiment_dir(args, config)
    config["EXPERIMENTS"]["experiment_dir"] = experiment_dir
    print(f"The experiment dir is {experiment_dir}")

    # Initialize randomness
    random_seed = args.random_seed
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    # Prepare data generator
    data_src = config["DATA"]["src"]
    if data_src != "local":
        # Generate the data on the fly
        train_generator, valid_generator, test_generator = construct_data_generator(args, config)
    else:
        train_generator, test_generator = read_data_generator(args, config)

    # Visualize data generator 
    # visualize_generator(train_generator, valid_generator, test_generator, dir_name = f"{experiment_dir}/figure")

    # Initialize model
    feat_dim = train_generator.unnormalized_features.size(1)
    latent_dim = args.latent_dim
    enc_hidden_sizes = config["MODEL"]["enc_hidden_sizes"]
    enc_hidden_sizes = [int(i) for i in enc_hidden_sizes.split(",")]
    dec_hidden_sizes = config["MODEL"]["dec_hidden_sizes"]
    model_type = config["MODEL"]["model_type"]

    if dec_hidden_sizes != "":
        dec_hidden_sizes = [int(i) for i in dec_hidden_sizes.split(",")]
    else:
        dec_hidden_sizes = []
    lr = config.getfloat("MODEL", "lr")

    results = defaultdict(dict)

    # Load the n_component from dict
    """
    latent_component_fn = "./latent_components"
    with open(latent_component_fn, "r") as f:
        latent_components_dict = json.load(f)
    dataset = Path(experiment_dir).parent.parent.parent.stem
    n_latent_component = latent_components_dict[dataset][str(random_seed)]
    latent_sizes = [n_latent_component]
    """
    if args.n_latent_component == 0:
        latent_sizes = range(3, 16)
    else:
        latent_sizes = [args.n_latent_component]

    if model_type == "nn":
        model = AutoEncoder(feat_dim, enc_hidden_sizes, dec_hidden_sizes, latent_dim)
        model.set_solver(lr)
        print(model)

        # Conduct training
        criterion = nn.MSELoss()
        valid_generator = None
        train(model, criterion, train_generator, valid_generator, args, config)

    else:
        # Select the model by ELBOW method
        model = select_gaussian_mixture_encoder(train_generator, latent_sizes, args.gmm_metric, args.pca_dim, config["EXPERIMENTS"]["experiment_dir"])


    # Conduct evaluation
    epsilons = [1, 0.5, 2, 4]

    stats = dict()
    targets = dict()
    wasserstein = False
    for f in ["mean", "var", "median"]:
        if wasserstein:
            stats[f], dist_dict, targets[f] = estimate(model, train_generator, test_generator, epsilons, args, config, f, wasserstein)
            wasserstein = False
        else:
            stats[f], _, targets[f] = estimate(model, train_generator, test_generator, epsilons, args, config, f, wasserstein)
    results = record_metric_results(stats, model.gmm.n_components, model.pca.n_components)

    # Save the results
    pathlib.Path(f"{experiment_dir}/results").mkdir(parents=True, exist_ok=True)

    dtype = args.data_dir.split("/")[-1]
    non_opt_size = test_generator.size

    if args.exp_type == "n_component":
        target_dir = f"n_components_individual/{args.n_latent_component}"
        pathlib.Path(target_dir).mkdir(parents = True, exist_ok = True)
        with open(f"{target_dir}/{dtype}_{non_opt_size}.pkl", "wb") as f:
            pickle.dump(results, f)
    elif args.exp_type == "metric":
        target_dir = f"metrics/{args.gmm_metric}"
        pathlib.Path(target_dir).mkdir(parents = True, exist_ok = True)
        with open(f"{target_dir}/{dtype}_{non_opt_size}.pkl", "wb") as f:
            pickle.dump(results, f)
    elif args.exp_type == "pca_dim":
        target_dir = f"pca_dim/{args.pca_dim}"
        pathlib.Path(target_dir).mkdir(parents = True, exist_ok = True)
        with open(f"{target_dir}/{dtype}_{non_opt_size}.pkl", "wb") as f:
            pickle.dump(results, f)

    with open(f"{experiment_dir}/results/results.pkl", "wb") as g:
        pickle.dump(stats, g)
    with open(f"{experiment_dir}/results/all_results.pkl", "wb") as g:
        pickle.dump(targets, g)
    
    if wasserstein:
        with open(f"{experiment_dir}/results/dist.pkl", "wb") as g:
            pickle.dump(dist_dict, g)

