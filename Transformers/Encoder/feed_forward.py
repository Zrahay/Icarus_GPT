import torch.nn as nn
import os
import numpy as np 

"""
This is the feed forward layer that we are going to use to process the tokens
"""

class Feedforward(nn.Module):
    def __init__(self, d_model, d_ff):
        """
        This is the constructor that we are going to use define the structure of the Feed Forward Network
        d_model: embedding dimension (6)
        d_ff: inner dimension of the FFN, typically 2x d_model (12)
        """
        super().__init__()
        self.layer1 = nn.Linear(d_model, d_ff) # We want to project this Input to d_ff space from d_model
        self.layer2 = nn.Linear(d_ff, d_model)
        self.activation = nn.GELU()
    
    def forward(self, X):
        """
        This forward layer is responsible for working with all teh attention layer
        """
        X = self.layer1(X) # The tensor gets projected to a higher dimension
        X = self.activation(X) # Activation goes between the two linear layers
        X = self.layer2(X)

        return X


