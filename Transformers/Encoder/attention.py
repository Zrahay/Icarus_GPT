import numpy as np
import torch.nn as nn
from scipy.special import softmax
"""
This file implements the Attention Block of the Transformer Architecture
"""

class Attention(nn.Module):
    """
    The blueprint for the implementation of the Attention Block
    """

    def __init__(self, d_model, heads):
        """
        d_model: The embedding dimension flowing through the architecture (6)
        heads: Number of attention heads (2), d_model must be divisible by heads

        The output is going to be a Nxd_model tensor
        """
        super().__init__()
        self.heads = heads
        self.d_model = d_model
        self.d_k = d_model // heads  # Per-head dimension: 6 // 2 = 3
        # We will also need to initialize three Weight Vectors of a certain shape with random values -> W(q), W(k), W(v)
        self.W_q = np.random.randn(self.d_model, self.d_k)
        self.W_k = np.random.randn(self.d_model, self.d_k)
        self.W_v = np.random.randn(self.d_model, self.d_k)

    
    def forward(self, X):
        """
        This forward API will be used to calculate the output for the Attention Block
        """
        Q = np.dot(X, self.W_q)
        K = np.dot(X, self.W_k)
        V = np.dot(X, self.W_v)

        K_t = K.T # We take the transpose of the Key matrix since we will be using it to calculate the final Attention Score
        attention_score = softmax(np.dot(Q, K_t) / (np.sqrt(self.d_k))) # This gives us a NxN tensor (Nxd_k.dot(d_kxN))
        final_score = np.dot(attention_score, V) # This returns a Nxd_k tensor. 
        """
        After this, all the outputs from all the heads are combined to give back the Nxd Tensor
        """
        return final_score

    


    