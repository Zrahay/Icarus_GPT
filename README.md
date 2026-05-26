# Icarus GPT

A from-scratch implementation of the Transformer Encoder architecture in PyTorch, built as a learning project to understand the internals of models like BERT.

## Overview

This project implements each component of the Transformer Encoder block individually, following the original architecture described in "Attention Is All You Need" (Vaswani et al., 2017). The goal is not to use high-level abstractions but to build each piece manually to understand exactly what is happening at every step.

The encoder stack processes a tokenized input sequence and produces a context-aware representation of each token. This is the same core architecture used in BERT and other encoder-only language models.

## Architecture

The encoder is composed of the following blocks, applied in order:

**Input Embeddings**
Converts a sequence of token IDs into dense vectors of dimension d_model. Uses PyTorch's nn.Embedding lookup table and scales the output by the square root of d_model, following the original paper.

**Positional Embeddings**
Since the attention mechanism has no inherent notion of sequence order, positional information is injected by adding a positional encoding matrix to the input embeddings. Even-indexed dimensions use a sine function and odd-indexed dimensions use a cosine function, both computed using the standard formula with a base of 10000.

**Multi-Head Attention**
Implements scaled dot-product attention. The input is projected into Query, Key, and Value matrices using learned weight vectors. The attention score is computed as the softmax of QK^T divided by the square root of d_k, which is then used to weight the Value matrix. Currently implements single-head attention, with the multi-head concatenation step planned.

**Add and Norm**
After the attention block, a residual connection is added between the attention output and the original input embeddings. The result is then normalized using a manual implementation of Layer Normalization, which computes the mean and variance per token vector and applies learnable scale (gamma) and shift (beta) parameters.

**Feed Forward Network**
A two-layer fully connected network applied independently to each token. The input is projected from d_model to d_ff (a larger intermediate dimension), passed through a GELU activation, and then projected back to d_model.

## Dimensions

The architecture is currently configured with the following small dimensions for ease of debugging and testing:

    vocab_size  = 100
    d_model     = 6
    heads       = 2
    d_k         = 3  (d_model / heads)
    d_ff        = 12
    max_seq_len = 50

## Project Structure

    transformers/
    |
    |__ Transformers/
    |   |__ Encoder/
    |       |__ inputs.py          Input Embeddings and Positional Embeddings
    |       |__ attention.py       Multi-Head Attention
    |       |__ feed_forward.py    Feed Forward Network
    |       |__ add_and_norm.py    Residual Connection and Layer Normalization
    |       |__ encoder.py         Encoder class composing all blocks
    |
    |__ test_encoder.py            Standalone test script for the encoder
    |__ README.md

## Running the Encoder

A test script is provided that runs a dummy tokenized sequence through the full encoder pipeline.

    python test_encoder.py

The script creates a sequence of 5 token IDs, passes them through the encoder, and prints the output tensor. The expected output shape is (5, 6), representing 5 tokens each with a 6-dimensional embedding.

Make sure to run the script using a Python environment that has PyTorch and NumPy installed.

## Status

The encoder is partially functional. Input Embeddings, Positional Embeddings, and Attention are working end to end. The multi-head concatenation step in Attention and the full Add and Norm integration are the immediate next steps. The Decoder block has not been implemented yet.
