"""
Implementation of Propensity score based estimation
Here we refer LCIW(Latent Cluster Importance Weighting) to the Propensity Score method in the paper.
And similary DPLCIW(Differentially Private Latent Cluster Importance Weighting) represents the proposed DiPPS method.
"""

import numpy as np
import torch
import subprocess
import pandas as pd
from collections import defaultdict 
from computeWassersteinDist import computeWassersteinDist
from math import gcd
from time import time
# from test_ortools import ortools

USE_CUDA = False
if USE_CUDA:
   torch.set_default_tensor_type('torch.cuda.FloatTensor')

class LCIW:
    def __init__(self, opt_generator, clustering_mech, args):

        self.opt_generator = opt_generator
        self.clustering_mech = clustering_mech
        self.opt_cluster_count = defaultdict(int)
        self.opt_cluster_2_idx = defaultdict(list)
        self.cluster_opt_proba = []
        self.n_cluster = 0
        self.build_opt_clusters()
        self.config = args

    def build_opt_clusters(self):
        """
        Build opt cluster count and cluster2idx
        """

        self.opt_cluster_count = defaultdict(int)
        self.opt_cluster_2_idx = defaultdict(list)
        self.cluster_opt_proba = []                             # Probability to select opt given cluster P(x | c)

        self.opt_generator.reset_randomness()
        backup_batch_size = self.opt_generator.batch_size
        self.opt_generator.batch_size = self.opt_generator.size
        batch_size = self.opt_generator.batch_size
        n_train_batch = self.opt_generator.size // batch_size

        if self.clustering_mech.model_type != "prop_score":
            for batch_id in range(n_train_batch):

                # Obtain the latent class unnormalized probabilities for current batch
                x, _ = self.opt_generator.next_batch()
                _, latent = self.clustering_mech.forward(x, return_latent = True)

                # Obtain latent classes for current batch
                if self.clustering_mech.model_type == "nn":
                    latent = latent.data.numpy()

                self.cluster_opt_proba.append(latent)

                latent_classes = torch.LongTensor([np.random.choice(range(latent.shape[1]), p = latent[i]) for i in range(latent.shape[0])])
                for i in range(batch_size):
                    self.opt_cluster_count[latent_classes[i].item()] += 1
                    self.opt_cluster_2_idx[latent_classes[i].item()].append(self.opt_generator.order[batch_id * batch_size + i])

            # Convert proba to tensor to store
            
            self.cluster_opt_proba = torch.FloatTensor(np.concatenate(self.cluster_opt_proba))
            self.cluster_opt_proba = self.cluster_opt_proba[np.argsort(self.opt_generator.order)]
            self.cluster_opt_proba /= self.cluster_opt_proba.sum(dim = 1).unsqueeze(dim = 1)

            # Get number of clusters
            self.n_cluster = self.cluster_opt_proba.size(1)
        else:
            self.n_cluster = 2
            self.opt_cluster_count[1] = self.opt_generator.size
            self.opt_cluster_2_idx[1] = list(range(self.opt_generator.size))
            self.cluster_opt_proba = self.clustering_mech.get_opt_scores()

        # Restore batch size
        self.opt_generator.batch_size = backup_batch_size
        
    def sample(self, k, v):
        """
        Sample cluster k for v times
        @param k(int): cluster id
        @param v(int): sample size
        @return torch.Tensor(v, feature_size)
        """

        selected_ids = np.random.choice(self.opt_cluster_2_idx[k], size = v)
        candidates = self.opt_generator.unnormalized_features[selected_ids]
        return candidates

    def estimate(self, non_opt_cluster_count, f, c = -1, hard = True):
        """
        Estimate f on non-opt + opt data using latent cluster importance weighting
        @param non_opt_cluster_count(dict): non opt data's latent cluster count
        @param f(function): function on a distribution to be evaluate such as mean/var/percentile
        @param c(int): sample size, default to size(non_opt) + size(opt)
        @param hard(bool): hard estimation or not
        @return f(opt + non_opt) 
        """

        if hard:
            return self.estimate_hard(non_opt_cluster_count, f)
        else:
            return self.estimate_soft(non_opt_cluster_count, f, c)

    def estimate_hard(self, non_opt_cluster_count, f):
        """
        Conduct hard estimation on f on non-opt + opt data
        @param non_opt_cluster_count(dict): non opt data's latent cluster count
        @param f(function): function on a distribution to be evaluate such as mean/var/percentile
        @return f(opt + non_opt) 
        """
        # Obtain the all latent count
        all_cluster_count = self.opt_cluster_count.copy()
        for k, v in non_opt_cluster_count.items():
            all_cluster_count[k] += v

        # Randomly sample candidates for each clusters from train data
        # This might be modified to use a generative network which learn the 
        # distribution of a cluster and genearate from it
        candidates = []
        for k, v in non_opt_cluster_count.items():
            candidates.append(self.sample(k, v))

        # Append the results with train samples
        candidates.append(torch.Tensor(self.opt_generator.unnormalized_features))
        candidates = torch.cat(candidates)

        # Apply target functions to candidates
        return f(candidates)

    def estimate_soft(self, non_opt_cluster_count, f, c = -1):
        """
        Conduct soft estimation of f on non-opt + opt using propensity score method with sample size=c
        @param non_opt_cluster_count(dict)
        @param f(func)
        @param c(int)
        """

        non_opt_count = sum(non_opt_cluster_count.values())
        opt_count = self.opt_generator.size
        c = max(c, self.opt_generator.size + sum(non_opt_cluster_count.values()))

        # Decide number of samples for opt and non-opt data separately 
        category_prop = [non_opt_count / (non_opt_count + opt_count), opt_count / (non_opt_count + opt_count)]
        sample_count = dict( zip( *np.unique( np.random.choice([0, 1], size = c, p = category_prop),
                                             return_counts = True) ) )
        sample_opt_count, sample_non_opt_count = sample_count[1], sample_count[0]
        print(f"Sample opt: {sample_opt_count}, Sample non-opt: {sample_non_opt_count}")

        # Sample opt
        sample_opt_indices = np.random.choice(range(opt_count), size = sample_opt_count)

        # Sample non-opt representatives in opt
        proba_opt_for_non_opt = self.get_proba_opt_for_non_opt(non_opt_cluster_count)
        sample_non_opt_indices = np.random.choice(range(opt_count), size = sample_non_opt_count, p = proba_opt_for_non_opt.data.numpy())
        candidates = self.opt_generator.unnormalized_features[np.concatenate([sample_opt_indices, sample_non_opt_indices])]

        return f(candidates)
    
    def get_proba_opt_for_non_opt(self, non_opt_cluster_count): 
        """
        Get probability of opt to be sampled to represent non-opt given their cluster distribution
        @param non_opt_cluster_count(dict)
        @return proba(torch.Tensor)
        """

        non_opt_count = sum(non_opt_cluster_count.values())
        non_opt_weight = torch.FloatTensor([non_opt_cluster_count[i] / non_opt_count for i in range(self.n_cluster)])

        opt_weight = (self.cluster_opt_proba * non_opt_weight).sum(dim = 1)

        return opt_weight

    def estimate_direct(self, non_opt_cluster_count, ftype = "mean", eval_range = "all"):
        """
        Estimate some special target function using closed form
        @param non_opt_cluster_count(dict)
        @return f's return value
        """

        if (ftype == "mean"):
            return self.estimate_mean_direct(non_opt_cluster_count, eval_range)
        elif (ftype == "var"):
            return self.estimate_var_direct(non_opt_cluster_count, eval_range)
        elif (ftype == "median"):
            return self.estimate_percentile_direct(non_opt_cluster_count, [0.5], eval_range)
            
    def estimate_mean_direct(self, non_opt_cluster_count, eval_range = "all"):

        sampling_proba, unique_opt_features = self.compute_soft_weights(non_opt_cluster_count, eval_range)
        return torch.sum(sampling_proba.unsqueeze(dim = 1) * unique_opt_features, dim = 0)
    
    def estimate_var_direct(self, non_opt_cluster_count, eval_range = "all"):
        
        sampling_proba, unique_opt_features = self.compute_soft_weights(non_opt_cluster_count, eval_range)
        all_count = self.opt_generator.size + sum(non_opt_cluster_count.values())

        sampling_proba = sampling_proba.unsqueeze(dim = 1)
        mean_est = torch.sum(sampling_proba * unique_opt_features, dim = 0)
        return all_count / (all_count - 1) * torch.sum((sampling_proba * (unique_opt_features - mean_est) ** 2), dim = 0)

    def estimate_percentile_direct(self, non_opt_cluster_count, p = [0.5], eval_range = "all"):

        sampling_proba, unique_opt_features = self.compute_soft_weights(non_opt_cluster_count, eval_range)
        quantiles = self.weighted_quantile(unique_opt_features, p, sampling_proba)
        return quantiles[0]

    def compute_soft_weights(self, non_opt_cluster_count, eval_range = "all"):
        """
        Compute the soft weights of opt-samples based on propensity score estimation
        @param non_opt_cluster_count(dict)
        @return (list)
        """

        opt_count = self.opt_generator.size
        non_opt_count = sum(non_opt_cluster_count.values())

        opt_cluster_weights = self.cluster_opt_proba.sum(dim = 0)

        # Convert non_opt_cluster_count into a n_cluster-dim tensor
        non_opt_cluster_count = torch.FloatTensor([non_opt_cluster_count[i] for i in range(self.n_cluster)])

        # Compute cluster propensity scores
        cluster_prop_scores = opt_cluster_weights / (opt_cluster_weights + non_opt_cluster_count)

        # Compute the unique cluster weights
        unique_opt_features, first_apperance_indices, opt_freq = np.unique(self.opt_generator.unnormalized_features,
                                                                        return_index = True, return_counts = True,
                                                                        axis = 0)
        opt_freq = torch.FloatTensor(opt_freq) / opt_count  # P(D = d | Z = 1)
        unique_opt_features = torch.FloatTensor(unique_opt_features)
        unique_cluster_opt_proba = self.cluster_opt_proba[first_apperance_indices]


        # Compute the propensity score for each opt sample
        if self.clustering_mech.model_type != "prop_score":
            prop_scores = torch.sum(unique_cluster_opt_proba * cluster_prop_scores, dim = 1)
            # print(f"Mean of propensity score is {prop_scores.mean()}")
        else:
            # print("Estimating using propensity score directly")
            prop_scores = self.clustering_mech.get_opt_scores()[:, 0]
            # print(f"Mean of propensity score is {prop_scores.mean()}")
            opt_freq = 1 / opt_count
            unique_opt_features = self.opt_generator.unnormalized_features
        # print(f"Std of propensity score is {prop_scores.std()}")

        # Compute opt weights (P(Z = 1))
        opt_weights = opt_count / (opt_count + non_opt_count)

        # Compute estimation of feature sampling probability
        sampling_proba = opt_freq * opt_weights / prop_scores
        sampling_proba /= sampling_proba.sum()

        # Compute non-opt sampling probability if evaluation range is test
        if eval_range == "test":
            # print(f"The evaluation range is {eval_range}")
            sampling_proba = (sampling_proba - opt_freq * (opt_weights)) / (1 - opt_weights)

        return sampling_proba, unique_opt_features

    def weighted_quantile(self, values, quantiles, sample_weight=None, 
                      values_sorted=False, old_style=False):
        """ Very close to numpy.percentile, but supports weights.
        NOTE: quantiles should be in [0, 1]!
        :param values: numpy.array with data
        :param quantiles: array-like with many quantiles needed
        :param sample_weight: array-like of the same length as `array`
        :param values_sorted: bool, if True, then will avoid sorting of
            initial array
        :param old_style: if True, will correct output to be consistent
            with numpy.percentile.
        :return: numpy.array with computed quantiles.
        """
        
        values = np.array(values).T
        quantiles = np.array(quantiles)
        if sample_weight is None:
            sample_weight = np.ones(len(values))
        sample_weight = np.array(sample_weight)
        assert np.all(quantiles >= 0) and np.all(quantiles <= 1), \
            'quantiles should be in [0, 1]'

        if not values_sorted:
            sorter = np.argsort(values, axis = 1)
            sorter = sorter.flatten()
            row_idx = np.zeros(values.shape[1])
            row_idx = np.tile(row_idx, (values.shape[0], 1)) + np.asarray(list(range(values.shape[0]))).reshape(-1, 1)
            row_idx = row_idx.flatten().astype(int)
            
            sorted_values = values[row_idx, sorter]
            sample_weight = sample_weight[sorter]
            
            # Reshape the values and sample weights
            sorted_values = sorted_values.reshape(values.shape)
            sample_weight = sample_weight.reshape(values.shape)
    #         print(sample_weight)
    #         print(sorted_values
        weighted_quantiles = np.cumsum(sample_weight, axis = 1) - 0.5 * sample_weight

        if old_style:
            # To be convenient with numpy.percentile
            weighted_quantiles -= weighted_quantiles[0]
            weighted_quantiles /= weighted_quantiles[-1]
        else:
            weighted_quantiles /= np.sum(sample_weight, axis = 1).reshape(-1, 1)
        
        # print(weighted_quantiles)
        # print(sorted_values)
        
        result = np.array([np.interp(quantiles, weighted_quantiles[i], sorted_values[i])
                            for i in range(values.shape[0])])
        
        return torch.FloatTensor(result.T)

    def compute_wasserstein_dist(self, non_opt_generator, non_opt_cluster_count, eval_range = "test", experiment_dir = "", eval_naive = True):
        """
        Call external r script to compute the wasserstein distance
        """
        
        if eval_range == "test":
            target_df = pd.DataFrame(non_opt_generator.unnormalized_features.data.numpy())
        elif eval_range == "all":
            target_df = pd.DataFrame(torch.cat([non_opt_generator.unnormalized_features,
                                                self.opt_generator.unnormalized_features]).data.numpy())

        # Compute naive distance
        if eval_naive:
            opt_df = pd.DataFrame(self.opt_generator.unnormalized_features.data.numpy())
            opt_df.to_csv(f"./{experiment_dir}/naive_src.csv", index = False)
            target_df.to_csv(f"./{experiment_dir}/naive_tgt.csv", index = False)

            common_multiplier = (opt_df.shape[0] * target_df.shape[0]) // gcd(opt_df.shape[0], target_df.shape[0])
            opt_weights = pd.Series((np.ones(opt_df.shape[0]) * common_multiplier // opt_df.shape[0]).astype(np.int))
            opt_weights.to_csv(f"./{experiment_dir}/naive_src_weights.csv", index = False)
            tgt_weights = pd.Series((np.ones(target_df.shape[0]) * common_multiplier // target_df.shape[0]).astype(np.int))
            tgt_weights.to_csv(f"./{experiment_dir}/naive_tgt_weights.csv", index = False)
            assert(opt_weights.sum() == tgt_weights.sum())
            print(f"Naive weights sum {opt_weights.sum()}")

            command = f"Rscript --vanilla computeWassersteinDist.r naive_src.csv naive_tgt.csv naive_src_weights.csv naive_tgt_weights.csv {experiment_dir}"
            process = subprocess.run(command.split(), stdout=subprocess.PIPE)
            out = process.stdout
            print(out.decode("utf-8").split(" ")[-1].rstrip())
            res = float(out.decode("utf-8").split(" ")[-1].rstrip())
            print(f"Naive {res}")
            r_naive_dist = res
            naive_dist = computeWassersteinDist(f"./{experiment_dir}/naive_src.csv", f"./{experiment_dir}/naive_tgt.csv",
                                                f"./{experiment_dir}/naive_src_weights.csv", f"./{experiment_dir}/naive_src_weights.csv", 
                                                f"./{experiment_dir}/naive_plan.csv", p = 1)
        else:
            r_naive_dist = 0
            naive_dist = 0

        # Compute reweighted distance
        sampling_proba, unique_opt_features = self.compute_soft_weights(non_opt_cluster_count, eval_range)
        boosted_sampling_proba = boost_sampling_proba(sampling_proba)

        # Compute the sum of boosted sampling proba and evaluate the amount need to be added to the target
        boosted_sum = boosted_sampling_proba.sum()
        tgt_individual_weight = boosted_sum // target_df.shape[0]
        dummy_tgt_weight = boosted_sum % target_df.shape[0]
        tgt_weights = np.ones(target_df.shape[0]).astype(np.int) * tgt_individual_weight 
        tgt_weights = np.append(tgt_weights, [dummy_tgt_weight])
        target_df.loc[target_df.shape[0]] = np.zeros(target_df.shape[1])
        assert(tgt_weights.sum() == boosted_sum)

        # Save src, tgt and their weights
        opt_df = pd.DataFrame(unique_opt_features.data.numpy())
        opt_df.to_csv(f"./{experiment_dir}/rw_src.csv", index = False)
        opt_weights = pd.Series(boosted_sampling_proba)
        opt_weights.to_csv(f"./{experiment_dir}/rw_src_weights.csv", index = False)
        target_df.to_csv(f"./{experiment_dir}/rw_tgt.csv", index = False)
        tgt_weights = pd.Series(tgt_weights)
        tgt_weights.to_csv(f"./{experiment_dir}/rw_tgt_weights.csv", index = False)
        
        # print(f"Unique size {opt_weights.shape} All size {self.opt_generator.unnormalized_features.shape}")
        print(f"Reweight weights sum {opt_weights.sum()}")
        command = f"Rscript --vanilla computeWassersteinDist.r rw_src.csv rw_tgt.csv rw_src_weights.csv rw_tgt_weights.csv {experiment_dir}"
        process = subprocess.run(command.split(), stdout=subprocess.PIPE)
        out = process.stdout
        res = float(out.decode("utf-8").split(" ")[-1].rstrip()) 
        
        print(f"Dummy target mass is {dummy_tgt_weight}")
        print(f"Reweighted {res}")
        r_reweighted_dist = res

        reweighted_dist = computeWassersteinDist(f"./{experiment_dir}/rw_src.csv", f"./{experiment_dir}/rw_tgt.csv",
             f"./{experiment_dir}/rw_src_weights.csv", f"./{experiment_dir}/rw_tgt_weights.csv",
              f"./{experiment_dir}/rw_plan.csv", p = 1)
        # reweighted_dist = ortools()
        print(f"Naive dist:{naive_dist}\tReweighted dist:{reweighted_dist}")
        return naive_dist, reweighted_dist, r_naive_dist, r_reweighted_dist

def good_print(out):

    out = out.decode("utf-8")
    print(out)

def boost_sampling_proba(sampling_proba):

    max_proba = sampling_proba.max()
    shift = 0
    while max_proba < 1:
        max_proba *= 10
        shift += 1
    
    boosted_sampling_proba = sampling_proba.data.numpy().copy()
    boosted_sampling_proba *= 10 ** shift
    boosted_sampling_proba *= 1e8
    boosted_sampling_proba = np.where(boosted_sampling_proba >= 1, boosted_sampling_proba, 0).astype(np.int)
    return boosted_sampling_proba

def boost_feature_vec(features):

    boosted_feat = features.copy()
    boosted_feat *= 1e6
    boosted_feat = boosted_feat.astype(np.int)
    return boosted_feat

def boost_dist(src_df, tgt_df, p = 1):

    BOOST_RATIO = 1e6
    src = src_df.values.copy()
    tgt = tgt_df.values.copy()

    src = src.reshape(src.shape[0], 1, -1)
    tgt =  tgt.reshape(1, *tgt.shape)
    
    cost = np.sum(np.abs(src - tgt), axis = 2)

    max_proba = cost.max()
    shift = 0
    while max_proba < 1:
        max_proba *= 10
        shift += 1
    cost *= 10 ** shift
    cost *= BOOST_RATIO
    cost = cost.astype(np.int)

    assert(cost.shape[0] == src_df.shape[0])
    assert(cost.shape[1] == tgt_df.shape[0])

    return cost, BOOST_RATIO * np.power(10, shift).astype(np.int)


