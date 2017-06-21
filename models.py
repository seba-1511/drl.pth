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

    def __init__(self, model):
        super(StochasticContinuousPolicy, self).__init__()
        self.model = model
        self.logstd = nn.Parameter(0.01 * th.rand(model.num_out))

    def forward(self, x):
        x = self.model(x)
        return x, self.logstd

    def reset(self):
        """ Resets the state of the current policy. """
        self.model.reset()


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
        # x = F.softmax(x)
        return x

    def reset(self):
        pass


class LSTM(nn.Module):

    def __init__(self, num_in, num_out, layers=(16, 16)):
        super(LSTM, self).__init__()
        self.num_in = num_in
        self.num_out = num_out
        self.layers = layers
        self.lstms = [nn.LSTM(num_in, layers[0])]
        self.hiddens = [(V(th.rand(1, 1, layers[0])), V(th.rand(1, 1, layers[0])))]
        for i, l in enumerate(layers[1:]):
            self.lstms.append(nn.LSTM(layers[i-1], layers[i]))
            self.hiddens.append(
                    (V(th.rand(1, 1, layers[i])),
                     V(th.rand(1, 1, layers[i])))
                    )
        self.lstms.append(nn.LSTM(layers[-1], num_out))
        self.hiddens.append(
                (V(th.rand(1, 1, num_out)),
                 V(th.rand(1, 1, num_out)))
                )

    def forward(self, x):
        for lstm, hidden in zip(self.lstms, self.hiddens):
            x, new_hidden = lstm(x.view(1, 1, -1), hidden)
            hidden[0].data.copy_(new_hidden[0].data)
            hidden[1].data.copy_(new_hidden[1].data)
        return x

    def reset(self):
        print('asdf')
        self.hiddens = [(V(th.rand(h[0].size())), 
                         V(th.rand(h[1].size()))) for h in self.hiddens]

class Atari(nn.Module):

    def __init__(self, num_in, num_out):
        super(Atari, self).__init__()
        self.num_in = num_in
        self.num_out = num_out

    def forward(self, x):
        return x
