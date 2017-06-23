#!/usr/bin/env python

import torch as th
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable as V

from functools import reduce


class StochasticPolicy(nn.Module):

    """
    Transforms a model into a stochastic continuous policy, with sampling
    from the logstd.

    `forward` returns a tuple, the mean and the logstd.
    """

    def __init__(self, model):
        super(StochasticContinuousPolicy, self).__init__()
        self.model = model
        self.params = nn.ParameterList(list(model.parameters()))
        self.logstd = nn.Parameter(0.01 * th.rand(model.num_out))
        self.params.extend([self.logstd, ])

    def forward(self, x):
        x = self.model(x)
        return x, self.logstd

    def reset(self):
        """ Resets the state of the current policy. """
        self.model.reset_state()


class DropoutPolicy(nn.Module):

    """
    Transforms a model into a stochastic continuous policy, with sampling
    from the logstd.

    `forward` returns a tuple, the mean and the logstd which are obtained by 
    forwarding through sampled dropout masks.
    """

    def __init__(self, model, num_samples=10):
        super(DropoutStochasticPolicy, self).__init__()
        self.model = model
        self.params = nn.ParameterList(list(model.parameters()))
        self.num_samples = num_samples

    def forward(self, x):
        samples = []
        states = []
        # TODO: avoid the for loop by creating a batch
        for _ in range(self.num_samples):
            out, state = self.model.forgetful_forward(x)
            samples.append(out)
            states.append(state)
        samples = th.cat(samples, 0)
        state = reduce(lambda x, y: [a + b for a, b in zip(x, y)], states[1:], states[0])
        state = [[val / self.num_samples for val in s] for s in state]
        self.model.set_state(state)
        mu, std = th.mean(samples, 0), th.std(samples, 0)
        return mu, th.log(std)
    
    def reset(self):
        self.model.reset_state()


class FC(nn.Module):

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
        print('Optimizing ', len(list(self.parameters())), ' parameters')

    def _track_params(self, layers):
        self.params = []
        for l in layers:
            self.params.extend(list(l.parameters()))
        self.params = nn.ParameterList(self.params)

    def forward(self, x):
        for l in self.layers[:-1]:
            x = self.activation(l(x))
            if self.dropout > 0.0:
                x = F.dropout(x, p=self.dropout, training=True)
        x = self.layers[-1](x)
        return x

    def forgetful_forward(self, x):
        return self.forward(x), [[0.0, ], ]

    def set_state(self, state):
        pass

    def reset_state(self):
        pass


class LSTM(nn.Module):

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
        print('Optimizing ', len(list(self.parameters())), ' parameters')

    def _track_params(self, layers):
        self.params = []
        for l in layers:
            self.params.extend(list(l.parameters()))
        self.params = nn.ParameterList(self.params)

    def forward(self, x):
        for lstm, hidden in zip(self.lstms[:-1], self.hiddens[:-1]):
            x, new_hidden = lstm(x.view(1, 1, -1), hidden)
            x = F.tanh(x)
            if self.dropout > 0.0:
                x = F.dropout(x, p=self.dropout, training=True)
            hidden[0].data.copy_(new_hidden[0].data)
            hidden[1].data.copy_(new_hidden[1].data)
        x, new_hidden = self.lstms[-1](x.view(1, 1, -1), self.hiddens[-1])
        self.hiddens[0].data.copy_(new_hidden[0].data)
        self.hiddens[1].data.copy_(new_hidden[1].data)
        return x

    def forgetful_forward(self, x):
        hiddens = []
        for lstm, hidden in zip(self.lstms[:-1], self.hiddens[:-1]):
            x, new_hidden = lstm(x.view(1, 1, -1), hidden)
            x = F.tanh(x)
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
