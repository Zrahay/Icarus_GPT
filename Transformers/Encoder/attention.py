import numpy as np
from scipy.special import softmax
"""
This file implements the Attention Block of the Transformer Architecture
"""

class Attention(nn.Module):
    """
    The blueprint for the implementation of the Attention Block
    """

    def __init__(self, X, heads):
        """
        X: A Nxd tensor will be given as an Input from the Positional Embedding Block

        The output is going to be a shape shifted Nxd(k) tensor
        """
        super.__init__()
        self.X = X
        self.n = X.shape(0)
        self.heads = heads
        self.d_model = X.shape(1) # This has to be specified in the architecture first. We can not let the  user choose whatever they want to
        self.d_k = self.d_model / (self.heads)
        # We will also need to initialize three Weight Vectors of a certain shape with random values -> W(q), W(k), W(v)
        self.W_q = np.random.randn(self.d_model, self.d_k)
        self.W_k = np.random.randn(self.d_model, self.d_k)
        self.W_v = np.random.rand(self.d_model, self.d_k)

    
    def forward(self, X):
        """
        This forward API will be used to calculate the output for the Attention Block
        """
        Q = np.dot(X, W_q)
        K = np.dot(X, W_k)
        V = np.dot(X, W_v)

        K_t = K.T # We take the transpose of the Key matrix since we will be using it to calculate the final Attention Score
        attention_score = softmax(np.dot(self.X, self.W_q) / (np.sqrt(self.d_model))) # This gives us a NxN tensor (Nxd_k.dot(d_kxN))
        final_score = np.dot(attention_score, V) # This returns a Nxd_k tensor. 
        """
        After this, all the outputs from all the heads are combined to give back the Nxd Tensor
        """
        return final_score

    


    