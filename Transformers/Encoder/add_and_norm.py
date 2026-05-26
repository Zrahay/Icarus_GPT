"""
This file implements the Add and Norm block that we have in the Transformer Architecture
"""
"""
In this layer, we are going to add the Residual Connections and we are going to  
we are going to perform some normalization tasks
"""

import torch
import torch.nn as nn 
import numpy as np 
from .attention import Attention

class AddNorm(nn.Module):
    def __init__(self, d_model):
        """
        This is the constructor of the class
        d_model: embedding dimension (6)
        """
        # In the norm layer we add the Input Embeddings to the output from the Multi-Head Attention Block
        super().__init__()
        self.d_model = d_model
        self.epsilon = 1e-6 # A constant we are going to use in the calculation of the Norm
        # We need to initialize both Gamma and Beta as well since they are learnable parameters in the Architecture
        self.beta = nn.Parameter(torch.zeros(self.d_model))
        self.gamma = nn.Parameter(torch.ones(self.d_model)) # Using nn.Parameters, we get these params as learnable parameters

    def add_block(self, X, X_input):
        """
        The method that is responsible for calculating the Residual Connection parts of this block
        """
        self.X_attention = X
        self.X_input = X_input
        self.d_model = X_input.shape[1]
        try:
            self.output = self.X_attention + self.X_input
        except ValueError as error:
            print("Mismatch in the shapes")
            print(error)
    def normalize_block(self, X):
        """
        We use this method to normalize the output that we get from the Add block
        """
        # So we after the residual connections are added, we add the norm layer 
        # For norm, we first calculate the meand and variance of the row and then manipulate the whole row

        for i, row in enumerate(X):
            mean_val = np.mean(row)
            var_val = np.var(row)
            self.output[i] = (row - mean_val) / np.sqrt(var_val + self.epsilon)
            self.output[i] = ((self.gamma) * (self.output[i])) + (self.beta)

        return self.output
