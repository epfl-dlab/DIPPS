import torch
import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA

# Define data loader
# Define data loader
class GMM:
    
    def __init__(self, K = 10, D = 2, mean = None, covariance = None, weights = None, random_seed = 1111):
        """
        Initialize GMM with K d-dim components so that mean, covariance and weights of each component
        are specified as the parameters
        @param K: number of component
        @param D: dimension of each component
        @param mean: mean of all components, shape = (K, D)
        @param covariance: shape = (K, D, D)
        @param weights: shape = K
        @param random_seed: internal random seed to generate samples
        """
        
        self.x_range = 10
        self.K = K
        self.D = D
        self.random_seed = random_seed
        self.r = np.random.RandomState(self.random_seed)
        
        self.mean = mean if mean is not None \
                            else np.random.randn(K, D) * self.x_range
        
        self.covariance = covariance if covariance is not None \
                                        else np.tile(np.identity(D), (K, 1, 1)) * 3
        self.covariance /= 2
        
        if covariance is None:
            # Make covariance symmetric and positive definite
            self.covariance += self.covariance.swapaxes(1, 2)
            self.covariance = np.matmul(self.covariance, self.covariance)
            
        self.weights = weights if weights is not None \
                                else self.r.uniform(low = 0.2, high = 1, size = K)
        
        self.components = [lambda n, i = i: self.r.multivariate_normal(self.mean[i], self.covariance[i], n) 
                              for i in range(K)]
        
        self.component_selection = lambda : self.r.choice(range(K), 1, 
                                                             p = self.weights / np.sum(self.weights))
        
#         print(f"Weights {self.weights}")
#         print(f"Mean {self.mean}")
        
    def sample(self, n = 100):
        """
        Generate n random samples from the given GMM
        @param n: number of samples
        @return vector of shape (n, D)
        """
        
        points = list()
        clusters = list()
        
        for _ in range(n):
            
            # Decide current cluster and generate point from the corresponding cluster
            cluster = self.component_selection()[0]
            point = self.components[cluster](1)[0]
            
            # Add the generated samples to holder
            clusters.append(cluster)
            points.append(point)
        #result = np.asarray([self.components[self.component_selection()[0]](1)[0] for _ in range(n)])
        
        points = np.asarray(points)
        
        return points, clusters
    
    def next_batch(self, batch_size = 256):
        """
        Generate next batch of sample points with cluster labels
        @param batch_size(int)
        @return features(np.array)
        @return labels(list)
        """
        
        features, labels = self.sample(batch_size)
        
        # Convert features and labels to torch.Tensor
        features = torch.autograd.Variable(torch.FloatTensor(features))
        labels = torch.LongTensor(labels)
        
        return features, labels

    def reset_randomness(self):
        """
        Reset random sample generator's seed for reproduction of sampling
        """
        
        self.r = np.random.RandomState(self.random_seed)

def visualize_generator(train_generator, valid_generator, test_generator, train_size = 10000, valid_size = 10000, test_size = 10000, dir_name = "."):
    """
    Visualize the difference among different data generator
    """
    
    # Sample data from generator
    x_train, y_train = train_generator.sample(train_size)
    x_valid, y_valid = valid_generator.sample(valid_size)
    x_test, y_test = test_generator.sample(test_size)

    # Conduct dimensionality reduction
    pca = PCA(n_components = 2)
    x_reduced_train = pca.fit_transform(x_train)
    x_reduced_valid = pca.transform(x_valid)
    x_reduced_test = pca.transform(x_test)

    fig = plt.figure(figsize = (24, 8))
    ax_1 = fig.add_subplot(131)
    ax_2 = fig.add_subplot(132)
    ax_3 = fig.add_subplot(133)
    ax_1.scatter(x_reduced_train[:, 0], x_reduced_train[:, 1], s = 5, c = "blue", label = "train")
    ax_2.scatter(x_reduced_valid[:, 0], x_reduced_valid[:, 1], s = 5, c = "red", label = "valid")
    ax_3.scatter(x_reduced_test[:, 0], x_reduced_test[:, 1], s = 5, c = "green", label = "test")
    ax_1.set_title("Train")
    ax_2.set_title("Valid")
    ax_3.set_title("Test")

    # Save the figure
    if dir_name != "":
        fig.savefig(f"{dir_name}/data.png")