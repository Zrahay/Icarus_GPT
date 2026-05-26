"""
This file is used for creating an Encoder and encapsulating all the other blocks under it
"""
from .attention import Attention
from .feed_forward import Feedforward
from .inputs import InputEmbeddings
from .inputs import PositionalEmbeddings
from .add_and_norm import AddNorm
import torch.nn as nn 
import numpy as np 

class Encoder(nn.Module):
    def __init__(self):
        """
        We are going to use this constructor to define the Encoder Layer
        """
        self.input_embeddings_layer = InputEmbeddings()
        self.positonal_embeddings_layer = PositionalEmbeddings()
        self.multi_head_attention = Attention()
        self.feed_forward = Feedforward()
        self.add_norm = AddNorm()
        # Forgot to implement the Add and Norm Layer

    def forward(self, X):
        """
        This API is going to be associated with feeding each layer the input and
        then the output of that layer being fed as an input to the next layer
        """
        X_1 = self.input_embeddings_layer(X)
        X_2 = self.positonal_embeddings_layer(X_1)
        X_3 = self.multi_head_attention.forward(X_2)
        X_4 = self.add_norm.add_block(X_3, X_1)
        X_5 = self.add_norm.normalize_block(X_4)
        X_6 = self.feed_forward(X_5)

        return X_6 # This is the final output of shape Nxd which is fed to the Decoder

