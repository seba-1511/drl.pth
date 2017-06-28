#!/usr/bin/env python

import torch as th
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable as V

from functools import reduce

class BasePolicyModel(nn.Module):

    """
    This class provides the architecure for models to be used by policies.
    """
    def __init__(self):
        super(BasePolicyModel, self).__init__()
        self.critic_state = None # The state that should be fed to the critic
        self.critic_input = 0 # size of the input to the critic

    def _track_params(self, layers):
        """
        Optional helper function: given a list of layers, adds their parameters
        to the current model.
        """
        self.params = []
        for l in layers:
            self.params.extend(list(l.parameters()))
        self.params = nn.ParameterList(self.params)


    def forward(self, x):
        """
        The standard forward pass of a nn.Module.

        Returns: the final activation.
        """
        pass

    def forgetful_forward(self, x):
        """
        A forward pass that uses but doesn't modify the state of the model. 
        E.g: forward of a RNN, but the hidden states stay untouched.

        Returns: (final activation, new state of the model)
        """
        pass

    def reset_state(self):
        """
        Resets the state of the model. E.g: re-initialize the hidden layers of
        an RNN.
        """
        pass

    def set_state(self, state):
        """

        """
        pass


class FC(BasePolicyModel):

    """ Policy implemented as a fully-connected network. """

    def __init__(self, num_in, num_out, layer_sizes=(16, 16), activation=None, dropout=0.0):
        super(FC, self).__init__()
        layers = [nn.Linear(num_in, layer_sizes[0])]
        for i, l in enumerate(layer_sizes[1:]):
            layer = nn.Linear(layer_sizes[i - 1], l)
            layers.append(layer)
        layer = nn.Linear(layer_sizes[-1], num_out)
        layers.append(layer)
        if activation is None:
            activation = F.tanh
        self.activation = activation
        self.num_in = num_in
        self.num_out = num_out
        self.dropout = dropout
        self.layers = layers
        self._track_params(layers)
        self.critic_state_size = layer_sizes[-1]
        print('Optimizing ', len(list(self.parameters())), ' parameters')

    def forward(self, x):
        for l in self.layers[:-1]:
            x = self.activation(l(x))
            self.critic_state = x
            if self.dropout > 0.0:
                x = F.dropout(x, p=self.dropout, training=True)
        x = self.layers[-1](x)
        return x

    def forgetful_forward(self, x):
        return self.forward(x), [[0.0, ], ]


class LSTM(BasePolicyModel):

    def __init__(self, num_in, num_out, layer_sizes=(16, 16), dropout=0.0):
        super(LSTM, self).__init__()
        self.num_in = num_in
        self.num_out = num_out
        self.layer_sizes = layer_sizes
        self.lstms = [nn.LSTM(num_in, layer_sizes[0])]
        self.hiddens = [(V(th.rand(1, 1, layer_sizes[0])), V(th.rand(1, 1, layer_sizes[0])))]

        # Create layers
        for i, l in enumerate(layer_sizes[1:]):
            self.lstms.append(nn.LSTM(layer_sizes[i - 1], layer_sizes[i]))
            self.hiddens.append((V(th.rand(1, 1, layer_sizes[i])), 
                                 V(th.rand(1, 1, layer_sizes[i]))))
        self.lstms.append(nn.LSTM(layer_sizes[-1], num_out))
        self.hiddens.append((V(th.rand(1, 1, num_out)),
                             V(th.rand(1, 1, num_out))))
        self.dropout = dropout
        self._track_params(self.lstms)
        self.critic_state_size = layer_sizes[-1]
        print('Optimizing ', len(list(self.parameters())), ' parameters')

    def forward(self, x):
        for lstm, hidden in zip(self.lstms[:-1], self.hiddens[:-1]):
            x, new_hidden = lstm(x.view(1, 1, -1), hidden)
            x = F.tanh(x)
            self.critic_state = x
            if self.dropout > 0.0:
                x = F.dropout(x, p=self.dropout, training=True)
            hidden[0].data.copy_(new_hidden[0].data)
            hidden[1].data.copy_(new_hidden[1].data)
        x, new_hidden = self.lstms[-1](x.view(1, 1, -1), self.hiddens[-1])
        self.hiddens[-1][0].data.copy_(new_hidden[0].data)
        self.hiddens[-1][1].data.copy_(new_hidden[1].data)
        return x

    def forgetful_forward(self, x):
        hiddens = []
        for lstm, hidden in zip(self.lstms[:-1], self.hiddens[:-1]):
            x, new_hidden = lstm(x.view(1, 1, -1), hidden)
            x = F.tanh(x)
            self.critic_state = x
            if self.dropout > 0.0:
                x = F.dropout(x, p=self.dropout, training=True)
            hiddens.append(new_hidden)
        x, new_hidden = self.lstms[-1](x.view(1, 1, -1), self.hiddens[-1])
        hiddens.append(new_hidden)
        return x, hiddens

    def set_state(self, state):
        for h, s in zip(self.hiddens, state):
            h[0].data.copy_(s[0].data)
            h[1].data.copy_(s[1].data)

    def reset_state(self):
        self.hiddens = [(V(th.rand(h[0].size())), 
                         V(th.rand(h[1].size()))) for h in self.hiddens]
