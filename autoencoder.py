"""
This file implements the autoencoder to find the latent classes.
In general every type of autoencoder can be deployed in our mechanism,
here we provide the fully-connected neural network based and gaussian-mixture model based ones.
The experiments in paper are based one gaussian-mixture encoder and the readers are encouraged
to implement customized ones for further exploration.
"""

import torch
from torch import nn, optim
from sklearn.mixture import GaussianMixture
import pandas as pd  
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
import numpy as np
from sklearn.metrics import silhouette_score

class Identity:

    def __init__(self, n_components):
        pass

    def transform(self, x_train):
        return x_train

    def fit(self, x_train):
        return


class AutoEncoder(nn.Module):

    def __init__(self, input_size, enc_hidden_sizes, dec_hidden_sizes, latent_size):
        
        super(AutoEncoder, self).__init__()
        self.model_type = "nn"

        # Encoder
        self.encoder = nn.ModuleList()
        layer_input_size = input_size

        # Add layers to encoder
        for layer_output_size in enc_hidden_sizes:

            self.encoder.append(nn.Linear(layer_input_size, layer_output_size))
            self.encoder.append(nn.ReLU())

            # Set input size of next layer to be output size of current layer
            layer_input_size = layer_output_size
        
        # Add latent layer to encoder
        self.encoder.append(nn.Linear(enc_hidden_sizes[-1], latent_size))

        # Define damping softmax layer
        self.latent = nn.ModuleList()
        self.latent.append(nn.Softmax(dim = 1))

        # Convert encoder list to sequential module
        self.encoder = nn.Sequential(*self.encoder)
        self.latent = nn.Sequential(*self.latent)
        
        # Decoder 
        self.decoder = nn.ModuleList()
        layer_input_size = latent_size

        # Add layers to decoder 
        for layer_output_size in dec_hidden_sizes:

            self.decoder.append(nn.Linear(layer_input_size, layer_output_size))
            self.decoder.append(nn.ReLU())

            # Set input size of next layer to be output size of last layer
            layer_input_size = layer_output_size
        
        # Add log softmax layer to produce prediction
        if len(dec_hidden_sizes) > 0:
            self.decoder.append(nn.Linear(dec_hidden_sizes[-1], input_size))
        else:
            self.decoder.append(nn.Linear(latent_size, input_size))
            
        # Convert decoder list to sequential module
        self.decoder = nn.Sequential(*self.decoder)

    def set_solver(self, lr):
        self.solver = optim.Adam(self.parameters(), lr=lr)

    def forward(self, x, epsilon = 1, return_latent = False):
        
        encoded = self.encoder(x)
        latent = self.latent(epsilon * encoded)
        output = self.decoder(latent)

        if return_latent:
            return output, latent
        else:
            return output

class GaussianMixtureEncoder:
    
    def __init__(self, input_size, enc_hidden_sizes, dec_hidden_sizes, latent_size, cov_type = "full", pca_dim = 2):
        """
        Only the latent size is used here
        """

        self.model_type = "gme"
        self.cov_type = cov_type
        self.n_components = latent_size
        self.pca = PCA(n_components = pca_dim)
        # self.pca = Identity(n_components = pca_dim)
        self.dbscan = DBSCAN(eps = 0.05)

    def fit(self, x_train):
        
        # Fit pca to high dimensional data
        self.pca.fit(x_train)

        # Feed low dimensional representation to Gaussian Mixture Model
        x_train_reduced = self.pca.transform(x_train)

        # init_means = self.select_init_means(x_train_reduced, self.n_components)

        # self.gmm = GaussianMixture(n_components = self.n_components,
        #                             means_init = init_means, covariance_type = self.cov_type)
        self.gmm = GaussianMixture(n_components = self.n_components, init_params = "kmeans", covariance_type = self.cov_type)
        self.gmm.fit(x_train_reduced)
    
    def select_init_means(self, x_train_reduced, n_components):
        """
        Select initial means using DBSCAN
        @param x_train_reduced(list): features
        @param n_components(int): number of cluster
        @return (list): means
        """

        # Get clusters of individual samples and cluster counts
        clusters = self.dbscan.fit_predict(x_train_reduced)
        cluster_counts = dict(zip(*np.unique(clusters, return_counts = True)))
        del cluster_counts[-1]

        # Get selected clusters by inverse frequency
        selected_clusters = sorted(cluster_counts.keys(), key = lambda x: cluster_counts[x], reverse = True)[: n_components]

        # Compute means of selected clusters
        init_means = [x_train_reduced[clusters == i].mean(axis = 0) for i in selected_clusters]

        # Add some random trivial means if the number of cluster detected by dbscan is less than requested
        if len(init_means) < n_components:
            init_means += [np.random.randn(x_train_reduced.shape[1]) for i in range(n_components - len(init_means))]
        print(f"The selected means by DBSCAN are {init_means}")
        return init_means

    def forward(self, x, return_latent = False):
        
        # Convert high dimensional features to low dimensional ones
        x_reduced = self.pca.transform(x)

        if return_latent:
            latent_weights = self.gmm.predict_proba(x_reduced)
            return 0, latent_weights
        else:
            return 0

    def compute_scores(self, x, metric = "default"):
        """
        Compute the scores of current model
        """

        x_reduced = self.pca.transform(x)

        if metric == "default":
            scores = self.gmm.score(x_reduced)
        elif metric == "silhouette":
            scores = silhouette_score(x_reduced, self.gmm.predict(x_reduced))
        elif metric == "bic":
            scores = self.gmm.bic(x_reduced)
        else:
            raise ValueError("Invalid parameter for gmm metric")
        # scores = silhouette_score(x_reduced, self.gmm.predict(x_reduced))
    
        return scores

    def eval(self):
        """ Trivial function to adapt to the structure of neural network"""
        return
