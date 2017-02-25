#!/usr/bin/env python

import torch as th
from torch import nn
from torch.nn import functional as F


class FCPolicy(nn.Module):

    """ Policy implemented as a fully-connected network. """

    def __init__(self, num_in, num_out, layers=(16, 16), activation=None):
        super(FCPolicy, self).__init__()
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
            activation = F.relu
        self.activation = activation
        self.num_in = num_in
        self.num_out = num_out

    def forward(self, x):
        for l in self.params[:-1]:
            x = self.activation(l(x))
        x = self.params[-1](x)
        # x = F.softmax(self.params[-1](x))
        return x
