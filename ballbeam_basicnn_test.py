# -*- coding: utf-8 -*-
"""
Created on Sat Oct 27 14:55:06 2018

@author: yiyuezhuo
"""

import numpy as np
import scipy.io
import matplotlib.pyplot as plt

matfile = scipy.io.loadmat("ballbeam.mat")
u = matfile["U"]
y = matfile["Y"]

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class System(nn.Module):
    # https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html
    def __init__(self, input_dim, hidden_dim, output_dim):
        # input_dim = control_dim + exact_lag_dim(option)
        super(System, self).__init__()
        self.hidden_dim = hidden_dim

        #self.word_embeddings = nn.Embedding(vocab_size, input_dim)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(input_dim, hidden_dim)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2out = nn.Linear(hidden_dim, output_dim)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (torch.zeros(1, 1, self.hidden_dim),
                torch.zeros(1, 1, self.hidden_dim))

    def forward(self, inputs):
        lstm_out, self.hidden = self.lstm(
            inputs.view(len(inputs), 1, -1), self.hidden)
        tag_space = self.hidden2out(lstm_out.view(len(inputs), -1))
        return tag_space
