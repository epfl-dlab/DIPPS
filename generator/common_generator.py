import torch
import numpy as np
class Generator:
    
    def __init__(self, features, labels, batch_size = 256, random_seed = 1111, unnormalized_features = None):
        
        self.features = torch.FloatTensor(features)
        self.labels = torch.FloatTensor(labels)
        self.unnormalized_features = torch.FloatTensor(unnormalized_features)
        self.size = len(features)
        self.random_seed = random_seed
        self.random = np.random.RandomState(self.random_seed)
        self.batch_size = batch_size
        self.order = self.random.permutation(self.size)
        self.current_index = 0
        
    def reset_randomness(self):
        
        self.random = np.random.RandomState(self.random_seed)
        self.current_index = 0
        
    def reorder(self):
            
        self.order = self.random.permutation(self.size)
        self.current_index = 0
        
    def next_batch(self):
        
        # Obtain next batch features and labels
        x_next_batch = self.features[ self.order[self.current_index: self.current_index + self.batch_size] ]
        x_next_batch = torch.autograd.Variable(x_next_batch)
        y_next_batch = self.labels[ self.order[self.current_index: self.current_index + self.batch_size] ]
        y_next_batch = torch.autograd.Variable(y_next_batch)
        
        # Increment index
        self.current_index += self.batch_size
        
        return x_next_batch, y_next_batch
    
    def next_unnormalized_batch(self):
        
        # Obtain next batch of unnormalized features and labesl
        x_next_batch = self.features[self.order[self.current_index: self.current_index + self.batch_size] ]
        x_next_batch = torch.autograd.Variable(x_next_batch)
        x_unnormalized_next_batch = self.unnormalized_features[self.order[self.current_index: self.current_index + self.batch_size]]
        y_next_batch = self.labels[ self.order[self.current_index: self.current_index + self.batch_size] ]
        y_next_batch = torch.autograd.Variable(y_next_batch)    
        
        # Increment index
        self.current_index += self.batch_size
        
        return x_unnormalized_next_batch, x_next_batch, y_next_batch