# -*- coding: utf-8 -*-
"""
Created on Sat Oct 27 16:48:07 2018

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


class LSTMStackSystem(nn.Module):
    # https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html
    def __init__(self, input_dim, hidden_dim, num_layers):
        # input_dim = control_dim + exact_lag_dim(option)
        super(LSTMStackSystem, self).__init__()
        self.hidden_dim = hidden_dim

        #self.word_embeddings = nn.Embedding(vocab_size, input_dim)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers = num_layers)

        # The linear layer that maps from hidden state space to tag space
        #self.hidden2out = nn.Linear(hidden_dim, output_dim)
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
        
        return lstm_out

y_lag_size = 5
u_lag_size = 5

def lag_design(x,lags):
    return np.transpose(np.array([np.roll(x,i) for i in range(lags+1)]))[lags:,:]
def design(y,u,lag_y,lag_u):
    Y = lag_design(y,lag_y)
    U = lag_design(u,lag_u)
    m = min(Y.shape[0],U.shape[0])
    return Y[:m,0],np.c_[Y[:m,1:],U[:m,:]]

Y,U = design(y.reshape(-1), u.reshape(-1), y_lag_size, u_lag_size)

inputs = torch.tensor(U, dtype=torch.float)
targets = torch.tensor(Y.reshape((Y.shape[0],1)), dtype=torch.float)

# define model
input_dim = 1 + y_lag_size + u_lag_size
hidden_dim = 5
output_dim = 1
fc_hidden_dim = 5
num_layers = 1
model = LSTMStackSystem(input_dim, hidden_dim, num_layers)

loss_function = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)


cuda = False
if cuda:
    inputs = inputs.cuda()
    targets = targets.cuda()
    model.cuda()


# See what the scores are before training
# Note that element i,j of the output is the score for tag j for word i.
# Here we don't need to train, so the code is wrapped in torch.no_grad()
with torch.no_grad():
    #inputs = torch.tensor([[1.],[2.],[3.],[4.]])
    output = model(inputs[:5])
    print(output)


    
#%time
num_iter = 100
for epoch in range(num_iter):  # again, normally you would NOT do 300 epochs, it is toy data
    #for inputs, tags in training_data:
        # Step 1. Remember that Pytorch accumulates gradients.
        # We need to clear them out before each instance
        model.zero_grad()

        # Also, we need to clear out the hidden state of the LSTM,
        # detaching it from its history on the last instance.
        model.hidden = model.init_hidden()

        # Step 2. Get our inputs ready for the network, that is, turn them into
        # Tensors of word indices.
        #sentence_in = prepare_sequence(sentence, word_to_ix)
        #targets = prepare_sequence(tags, tag_to_ix)
        

        # Step 3. Run our forward pass.
        output = model(inputs)

        # Step 4. Compute the loss, gradients, and update the parameters by
        #  calling optimizer.step()
        loss = loss_function(output, targets)
        print(f"{epoch}/{num_iter} loss: {loss.item()}")
        
        loss.backward()
        optimizer.step()

with torch.no_grad():
    #inputs = torch.tensor([[1.],[2.],[3.],[4.]])
    output = model(inputs)
    #print(output)

plt.plot(u,label='u')
plt.plot(y,label='y')
plt.plot(output.numpy(),label='out')
plt.legend()
plt.show()


