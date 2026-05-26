"""
This file is the inputs that we are going to send in the Transformer Block
"""
from math import cos, sin, sqrt, pow
import torch
import torch.nn as nn
import numpy as np


class InputEmbeddings(nn.Module):
    """
    This class will be used by us to create the Nxd input embeddings of the inputs
    """

    def __init__(self, vocab_size, d_model): # We are assuming a 1D Tensor as of now (1D Tensor means a row vector).
        super().__init__()

        self.embeddings = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model

    def forward(self, x):
        """
        This is the Instance method that we are going define 

        The input is a 1D Sequence.
        """
        # We need to convert this tokenized sequence into a Nxd Tensor

        self.output = self.embeddings(x) * sqrt(self.d_model)

        return self.output

class PositionalEmbeddings(nn.Module):
    """
    This class implements the Positional Embeddings block. This block helps us find us the Positional Embeddings
    """

    def __init__(self, d_model, max_seq_len):
        """
        This is the constructor for the Positional Embedding class that we are going to work with
        """
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        # We need to output a Nxd tensor which will be added to the original Nxd tensor.
    def forward(self, X):
        # We need to generate a Nxd tensor using the sin and consine formulas that we have to respect the slow and fast tokens strategy
        # We need to iterate through self.n and self.d_model to to calculate the embeddings using sine and cosine
        n = X.shape[0]

        matrix = torch.tensor([[self.calculate(r, c) for c in range(self.d_model)] for r in range(n)], dtype=torch.float32)

        try:
            layer_output = matrix + X
        except ValueError as error:
            print("Shape mismatch between the PE Matrix shape and the Layer shape")
            print(error)


        return layer_output


    def calculate(self, row_number, column_number):
        # I will use this helper function to calculate the sine embeddings.
        # Sine only works for the even dimensions. Its formula is -> sin(pos / (10000 ^ (2*i / d_model)))
        denominator = pow(10000, (2 * column_number) / self.d_model)
        if column_number % 2 == 0:
            return sin(row_number / denominator)
        else:
            return cos(row_number / denominator)






    
