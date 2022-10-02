import torch
import numpy as np
from collections import defaultdict
from exponential_mechanism import exponential_mechanism

from invert_exponential_mechanism import invert_exponential_output
from laplace_mechanism import laplace_mechanism
from util import l1_loss
from util import l2_loss
import sys
# sys.path.append("../src/")
from hybrid_mechanism import *
from LCIW import *
import pprint
from time import time

def get_latent_count(generator, model):
    """
    Get the latent cluster count of data
    @param generator
    @param model
    @return (dict): latent_count
    """

    generator.reset_randomness()
    backup_batch_size = generator.batch_size
    generator.batch_size = generator.size
    n_batch = generator.size // generator.batch_size
    batch_size = generator.batch_size
    latent_count = defaultdict(int)

    model.eval()
    with torch.no_grad():
        
        for _ in range(n_batch):
            
            # Get next batch's target and latent features
            x, y = generator.next_batch()
            _, latent = model.forward(x, return_latent = True)
            latent_size = latent.shape[1]

            # Record latent cluster for each test sample
            if model.model_type == "nn":
                latent = latent.data.numpy()

            # Record latent clusters
            latent_classes = torch.LongTensor([np.random.choice(range(latent_size), p = latent[i]) for i in range(latent.shape[0])])
            # latent_classes = latent.argmax(axis = 1)
            
            for i in range(batch_size):
                _class = latent_classes[i].item()
                latent_count[_class] += 1
    
    for c in range(latent_size):
        latent_count[c] = max(latent_count[c], 0)

    generator.batch_size = backup_batch_size
    return latent_count

def get_noisy_latent_count(generator, model, eps = 1):

    latent_counts = defaultdict(int)
    backup_batch_size = generator.batch_size
    generator.batch_size = generator.size
    n_batch = generator.size // generator.batch_size
    generator.reset_randomness()
    model.eval()

    with torch.no_grad():

        for _ in range(n_batch):
            
            # Get next batch's target and latent features
            x, y = generator.next_batch()
            _, latent = model.forward(x, return_latent = True)
            latent_size = latent.shape[1]

            # Record latent cluster for each test sample
            if model.model_type == "nn":
                latent = latent.data.numpy()

            # Record noisy targets for each latent class using exponential mechanism
            latent_classes = exponential_mechanism(latent, eps).reshape(-1)
            for _class in latent_classes:
                latent_counts[_class] += 1

        # Conduct post-processing to obtain noisy latent class weight
        latent_weights = invert_exponential_output(np.asarray([latent_counts[_class] for _class in range(latent_size)]),
                                                    epsilon = eps)
        latent_weights *= generator.size
        latent_counts = dict(zip(range(latent_size), latent_weights))
    # pprint.pprint(latent_counts)

    generator.batch_size = backup_batch_size
    return latent_counts

def compute_loss(target_dict, eval_range = "all"):
    """
    Compute the naive, laplace, hybrid, lciw, dplciw loss
    @param target_dict(dict)
    @return (dict)
    """

    # Compute loss 
    l1_loss = dict() 
    target_type = eval_range
 
    l1_loss["naive"] = torch.mean(torch.abs(target_dict["train_gt"] - target_dict[f"{target_type}_gt"])) 
    
    l1_loss["Laplace mean"] = torch.mean(torch.mean(torch.abs(target_dict[f"{target_type}_gt"].unsqueeze(dim = 0) - target_dict["lap"]), axis = 1)) 
    
    l1_loss["Laplace std"] = torch.std(torch.mean(torch.abs(target_dict[f"{target_type}_gt"].unsqueeze(dim = 0) - target_dict["lap"]), axis = 1))
    
    l1_loss["LCIW"] = torch.mean(torch.abs(target_dict["LCIW"] - target_dict[f"{target_type}_gt"]))
    
    l1_loss["DPLCIW mean"] = torch.mean(torch.mean(torch.abs(target_dict[f"{target_type}_gt"].unsqueeze(dim = 0) - target_dict["DPLCIW"]), axis = 1))
    
    l1_loss["DPLCIW std"] = torch.std(torch.mean(torch.abs(target_dict[f"{target_type}_gt"].unsqueeze(dim = 0) - target_dict["DPLCIW"]), axis = 1))
    
    l1_loss["Hybrid mean"] = torch.mean(torch.mean(torch.abs(target_dict[f"{target_type}_gt"].unsqueeze(dim = 0) - target_dict["hybrid"]), axis = 1))
    
    l1_loss["Hybrid std"] = torch.std(torch.mean(torch.abs(target_dict[f"{target_type}_gt"].unsqueeze(dim = 0) - target_dict["hybrid"]), axis = 1))

    return l1_loss

def estimate(model, train_generator, test_generator, epsilons, args, config, f = "mean", wasserstein = False):
    """
    Estimate target variable of test data using reweighint on train data
    @param model(AutoEncoder)
    @param train_generator(GMM)
    @param test_generator(GMM)
    @param epsilon(list(float))
    @param args
    @param config
    @param f: target function
    """

    # Read configuration
    random_seed = args.random_seed
    latent_dim = train_generator.unnormalized_features.shape[1]

    batch_size = config.getint("TRAINING", "batch_size")
    eval_range = config["EVALUATION"]["eval_range"]
    n_train_batch = train_generator.size // train_generator.batch_size
    n_test_batch = test_generator.size // test_generator.batch_size
    n_trial = config.getint("EVALUATION", "n_trial")
    train_count = train_generator.size
    test_count = test_generator.size
    print(f"Train count is {train_count}, test count is {test_count}")
    print(f"The feature dim is {latent_dim}")
    all_count = train_count + test_count

    # Initialize target holder
    target_dict = {}
    dist_dict = {}
    latent_2_tgt_dict = {}

    """ Obtain train groundtruth """
    train_generator.reset_randomness()
    if f == "mean":
        train_gt = train_generator.unnormalized_features.mean(axis = 0)
    elif f == "var":
        train_gt = train_generator.unnormalized_features.var(axis = 0)
    elif f == "median":
        train_gt = train_generator.unnormalized_features.median(axis = 0)[0]
    else:
        train_gt = f(train_generator.unnormalized_features)
    target_dict["train_gt"] = train_gt

    """ Get test groundtruth """
    test_generator.reset_randomness()
    if f == "mean":
        test_gt = test_generator.unnormalized_features.mean(axis = 0)
    elif f == "var":
        test_gt = test_generator.unnormalized_features.var(axis = 0)
    elif f == "median":
        test_gt = test_generator.unnormalized_features.median(axis = 0)[0]
    else:
        test_gt = f(test_generator.unnormalized_features)
    target_dict["test_gt"] = test_gt

    """ Get all groundtruth """
    if f == "mean":
        all_gt = train_gt * train_count / all_count + test_gt * test_count / all_count
    elif f == "var":
        all_gt = torch.cat([train_generator.unnormalized_features, test_generator.unnormalized_features]).var(axis = 0)
    elif f == "median":
        all_gt = torch.cat([train_generator.unnormalized_features, test_generator.unnormalized_features]).median(axis = 0)[0]
    else:
        all_gt = f(torch.cat([train_generator.unnormalized_features, test_generator.unnormalized_features]))
    target_dict["all_gt"] = all_gt

    losses = dict()
    wasserstein_computed = False
    for epsilon in epsilons:

        """ 1. Laplace mechanism estimation """
        sensitivity = latent_dim * 2
        laplace_results = []
        for _ in range(n_trial):
            lap_data = laplace_mechanism(test_generator.unnormalized_features, sensitivity, epsilon)
            if eval_range == "all":
                lap_data = torch.cat([lap_data, train_generator.unnormalized_features])
            
            if f == "mean":
                result = lap_data.mean(dim = 0)
            elif f == "var":
                if eval_range == "all":
                    balance_term = (1 - train_count / (train_count + test_count)) ** 2 * 2 * (sensitivity / epsilon) ** 2
                else:
                    balance_term = 2 * (sensitivity / epsilon) ** 2     # Need to specify the balance term for evaluation on test
                result = lap_data.var(dim = 0) - balance_term
            elif f == "median":
                result = lap_data.median(dim = 0)[0]
            else:
                result = f(lap_data)
            laplace_results.append(result)
        target_dict["lap"] = torch.stack(laplace_results)
        
        """ 2. Hybrid mechanism estimation """
        hybrid_results = []

        for _ in range(n_trial):

            hybrid_data = []
            for participant in test_generator.unnormalized_features.data.numpy():
                hybrid_data.append(multi_dim_hybrid_mechanism(participant, epsilon))

            hybrid_data = torch.FloatTensor(np.asarray(hybrid_data))
            # hybrid_data = torch.FloatTensor([multi_dim_hybrid_mechanism(participant, epsilon)
            #                                     for participant in test_generator.unnormalized_features])
            if eval_range == "all": 
                hybrid_data = torch.cat([hybrid_data, train_generator.unnormalized_features])

            if f == "mean":
                result = hybrid_data.mean(dim = 0)
            elif f == "var":
                result = hybrid_data.var(dim = 0)
            elif f == "median": 
                result = hybrid_data.median(dim = 0)[0]
            else:
                result =f(hybrid_data)
            hybrid_results.append(result)
        target_dict["hybrid"] = torch.stack(hybrid_results)

        """ 3. Propensity Score based estimation """
        lciw = LCIW(train_generator, model, args)
        test_latent_count = get_latent_count(test_generator, model)
        lciw_result = lciw.estimate_direct(test_latent_count, f, eval_range = eval_range)    # TO(DO: Might need to reshape it a little bit here
        start = time()
        if not wasserstein_computed:
            lciw_dist = lciw.compute_wasserstein_dist(test_generator, test_latent_count, eval_range = eval_range, experiment_dir = config["EXPERIMENTS"]["experiment_dir"]) if wasserstein else [0, 0, 0, 0]
        print(f"Time spent on calculating wasserstein dist is {(int)(time() - start)}")
        dist_dict["naive"] = lciw_dist[0]
        dist_dict["LCIW"] = lciw_dist[1]
        dist_dict["r_naive"] = lciw_dist[2]
        dist_dict["r_LCIW"] = lciw_dist[3]
        target_dict["LCIW"] = lciw_result


        """ 4. DiPPS estimation """
        dplciw_results = []
        if not wasserstein_computed:
            dplciw_dists = []
            r_dplciw_dists = []
        dplciw_latent_counts = []
        for _ in range(n_trial):
            test_latent_count = get_noisy_latent_count(test_generator, model, epsilon)
            dplciw_latent_counts.append(test_latent_count)                 # TODO: Remove this sanity checking line of code
            dplciw_results.append(lciw.estimate_direct(test_latent_count, f, eval_range = eval_range))
            if not wasserstein_computed:
                if wasserstein:
                    _, reweighted, _, r_reweighted = lciw.compute_wasserstein_dist(test_generator, test_latent_count, eval_range = eval_range, experiment_dir = config["EXPERIMENTS"]["experiment_dir"], eval_naive = False)
                else:
                    reweighted, r_reweighted = 0, 0
                dplciw_dists.append(reweighted)
                r_dplciw_dists.append(r_reweighted)
                wasserstein_computed = True
        dist_dict["DPLCIW"] = torch.Tensor(dplciw_dists)
        dist_dict["r_DPLCIW"] = torch.Tensor(r_dplciw_dists)
        target_dict["DPLCIW"] = torch.stack(dplciw_results)

        """ Compute loss """
        loss = compute_loss(target_dict, eval_range = eval_range)
        losses[epsilon] = loss

        wasserstein = False

    # Record results
    print(f"Evaluating {f}")
    pprint.pprint(losses)
    pprint.pprint(dist_dict)
    return losses, dist_dict, target_dict
