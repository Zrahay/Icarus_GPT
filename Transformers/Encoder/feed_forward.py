import torch.nn as nn
import os
import numpy as np 

"""
This is the feed forward layer that we are going to use to process the tokens
"""

class Feedforward(nn.Module):
    def __init__(self):
        """
        This is the constructor that we are going to use define the structure of the Feed Forward Network
        """
        super.__init__()
        self.layer1 = nn.Linear(6, 12) # We want to project this Input to 12 Embeddings space from 6, so, we project it from 6 to 12.
        self.layer2 = nn.Linear(12, 6)
        self.activation = nn.GeLU()
    
    def forward(self, X):
        """
        This forward layer is responsible for working with all teh attention layer
        """
        X = self.layer1(X) # The tensor gets projected to a higher dimension
        X = self.layer2(X)
        X = self.activation(X)

        return X


