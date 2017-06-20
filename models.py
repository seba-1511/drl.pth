#!/usr/bin/env python

import torch as th
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable as V


class StochasticContinuousPolicy(nn.Module):

    """
    Transforms a model into a stochastic continuous policy, with sampling
    from the logstd.

    `forward` returns a tuple, the mean and the logstd.
    """

    def __init__(self, net):
        super(StochasticContinuousPolicy, self).__init__()
        self.net = net
        self.logstd = nn.Parameter(0.1 * th.rand(net.num_out))

    def forward(self, x):
        x = self.net(x)
        return x, self.logstd


class FC(nn.Module):

    """ Policy implemented as a fully-connected network. """

    def __init__(self, num_in, num_out, layers=(16, 16), activation=None):
        super(FC, self).__init__()
        params = [nn.Linear(num_in, layers[0])]
        for i, l in enumerate(layers[1:]):
            layer = nn.Linear(layers[i - 1], l)
            params.append(layer)
            setattr(self, 'l' + str(i), layer)
        layer = nn.Linear(layers[-1], num_out)
        params.append(layer)
        setattr(self, 'last', layer)
        self.params = params
        if activation is None:
            activation = F.tanh
        self.activation = activation
        self.num_in = num_in
        self.num_out = num_out

    def forward(self, x):
        for l in self.params[:-1]:
            x = self.activation(l(x))
        x = self.params[-1](x)
        # x = F.softmax(self.params[-1](x))
        return x


class LSTM(nn.Module):

    def __init__(self, num_in, num_out, layers=(16, 16)):
        super(LSTM, self).__init__()
        self.num_in = num_in
        self.num_out = num_out
        self.layers = layers

    def forward(self, x):
        return x

class Atari(nn.Module):

    def __init__(self, num_in, num_out):
        super(Atari, self).__init__()
        self.num_in = num_in
        self.num_out = num_out

    def forward(self, x):
        return x
