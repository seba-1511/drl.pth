#!/usr/bin/env python

import numpy as np
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
        self.critic_state_size = 0 # size of the input to the critic

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
        Sets the state of the model to the provided one. E.g: the hidden layers 
        of an RNN.
        """
        pass

    def get_state(self):
        """
        Gets the state of the model to the provided one. E.g: the hidden layers 
        of an RNN.
        """
        return []


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
        return self.forward(x), []


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
        self.reset_state()
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
        state = []
        for lstm, hidden in zip(self.lstms[:-1], self.hiddens[:-1]):
            x, new_hidden = lstm(x.view(1, 1, -1), hidden)
            x = F.tanh(x)
            self.critic_state = x
            if self.dropout > 0.0:
                x = F.dropout(x, p=self.dropout, training=True)
            state.append(new_hidden[0])
            state.append(new_hidden[1])
        x, new_hidden = self.lstms[-1](x.view(1, 1, -1), self.hiddens[-1])
        state.append(new_hidden[0])
        state.append(new_hidden[1])
        return x, state

    def set_state(self, state):
        hiddens = []
        for i in range(0, len(state), 2):
            hiddens.append((state[i], state[i+1]))
        self.hiddens = hiddens

    def get_state(self):
        state = []
        for h in self.hiddens:
            state.append(h[0])
            state.append(h[1])
        return state

    def reset_state(self):
        self.hiddens = [(V(th.zeros(h[0].size())), 
                         V(th.zeros(h[1].size()))) for h in self.hiddens]




        """ Atari stuff is below: """


def normalized_columns_initializer(weights, std=1.0):
    out = th.randn(weights.size())
    out *= std / th.sqrt(out.pow(2).sum(1).expand_as(out))
    return out


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = np.prod(weight_shape[1:4])
        fan_out = np.prod(weight_shape[2:4]) * weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = weight_shape[1]
        fan_out = weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)

class Atari(BasePolicyModel):

    def __init__(self, num_in, num_out, layer_sizes=(16, 16), activation=None, dropout=0.0):
        super(Atari, self).__init__()
        self.activation = activation
        self.dropout = dropout
        self.num_out = num_out
        self.num_in = 210

        self.conv1 = nn.Conv2d(self.num_in, 32, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(32, 32, 3, stride=2, padding=1)

        self.lstm = nn.LSTMCell(320, 256)

        num_outputs = num_out
        self.actor_linear = nn.Linear(256, num_outputs)

        self.apply(weights_init)
        self.actor_linear.weight.data = normalized_columns_initializer(
            self.actor_linear.weight.data, 0.01
        )
        self.actor_linear.bias.data.fill_(0)

        self.lstm.bias_ih.data.fill_(0)
        self.lstm.bias_hh.data.fill_(0)

        self.critic_state = None
        self.critic_state_size = 256
        self.reset_state()

    def forward(self, x):
        x = F.elu(self.conv1(x))
        x = F.elu(self.conv2(x))
        x = F.elu(self.conv3(x))
        x = F.elu(self.conv4(x))
        x = x.view(-1, 320)
        hx, cx = self.lstm(x, (self.hx, self.cx))
        x = hx

        self.hx = hx
        self.cx = cx

        self.critic_state = x
        if self.dropout > 0.0:
            x = F.dropout(x, self.dropout, training=True)

        return self.actor_linear(x)

    def forgetful_forward(self, x):
        x = F.elu(self.conv1(x))
        x = F.elu(self.conv2(x))
        x = F.elu(self.conv3(x))
        x = F.elu(self.conv4(x))
        x = x.view(-1, 320)
        hx, cx = self.lstm(x, (self.hx, self.cx))
        x = hx

        self.critic_state = x
        if self.dropout > 0.0:
            x = F.dropout(x, self.dropout, training=True)

        return self.actor_linear(x), [hx, cx]

    def set_state(self, state):
        hx, cx = state
        self.hx = hx
        self.cx = cx

    def reset_state(self):
        self.cx = V(th.zeros(1, 256))
        self.hx = V(th.zeros(1, 256))

    def get_state(self):
        return [self.cx, self.hx]
