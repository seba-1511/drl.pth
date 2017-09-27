#!/usr/bin/env python

import numpy as np
import torch as th
from torch import nn
from torch import Tensor as T
from torch.nn import functional as F
from torch.autograd import Variable as V

from functools import reduce


class DiscreteFeatures(nn.Module):

    def __init__(self, state_size, layer_sizes=[128, 128], dropout=0.0):
        super(DiscreteFeatures, self).__init__()
        if dropout != 0.0:
            raise Exception('Dropout not supported yet.')
        self.affine1 = nn.Linear(state_size, layer_sizes[0])

    def forward(self, x):
        x = F.relu(self.affine1(x))
        return x


class DiscreteActor(nn.Module):

    def __init__(self, feature_extractor=None, action_size=None, features_size=None):
        super(DiscreteActor, self).__init__()
        self.feature_extractor = feature_extractor
        self.linear = nn.Linear(features_size, action_size)

    def forward(self, x):
        features = self.feature_extractor(x)
        return self.linear(features)


class ContinuousFeatures(nn.Module):

    def __init__(self, state_size, layer_sizes=[128, 128], dropout=0.0):
        super(ContinuousFeatures, self).__init__()
        if dropout != 0.0:
            raise Exception('Dropout not supported yet.')
        self.affine1 = nn.Linear(state_size, layer_sizes[0])

    def forward(self, x):
        x = F.tanh(self.affine1(x))
        return x


class ContinuousActor(nn.Module):

    def __init__(self, feature_extractor=None, action_size=None, features_size=None):
        super(ContinuousActor, self).__init__()
        self.feature_extractor = feature_extractor
        self.linear = nn.Linear(features_size, action_size, bias=False)
        self.linear.weight.data *= 0.1

    def forward(self, x):
        features = self.feature_extractor(x)
        return self.linear(features)




class Critic(nn.Module):

    def __init__(self, feature_extractor, state_size):
        super(Critic, self).__init__()
        self.feature_extractor = feature_extractor
        self.linear = nn.Linear(state_size, 1)

    def forward(self, x):
        features = self.feature_extractor(x)
        return self.linear(features)


class ConstantCritic(nn.Module):

    def __init__(self, value=0.0):
        super(ConstantCritic, self).__init__()
        self.value = V(th.zeros(1, 1) + value) 

    def forward(self, x):
        return self.value


class LSTMFeatures(nn.Module):

    def __init__(self, state_size, layer_sizes=[128, 128], dropout=0.0):
        super(LSTMFeatures, self).__init__()
        if dropout != 0.0:
            raise Exception('Dropout not supported yet.')
        self.lstm = nn.LSTMCell(state_size, layer_sizes[0])

    def forward(self, x, states):
        x = F.relu(self.lstm(x, states))
        return x


def FC2(state_size, action_size, layer_sizes=[128, 128], dropout=0.0, discrete=True):
    if discrete:
        features = DiscreteFeatures(state_size, layer_sizes, dropout)
        actor = DiscreteActor(feature_extractor=features,
                      features_size=layer_sizes[-1],
                      action_size=action_size)
    else:
        features = ContinuousFeatures(state_size, layer_sizes, dropout)
        actor = ContinuousActor(feature_extractor=features,
                      features_size=layer_sizes[-1],
                      action_size=action_size)
    critic = Critic(feature_extractor=features,
                    state_size=layer_sizes[-1])
    return (actor, critic)

def LSTM2(state_size, action_size, layer_sizes=[128, 128], dropout=0.0):
    features = LSTMFeatures(state_size, layer_sizes, dropout)
    actor = Actor(feature_extractor=features,
                  features_size=layer_sizes[-1],
                  action_size=action_size)
    critic = Critic(feature_extractor=features,
                    state_size=layer_sizes[-1])
    return (actor, critic)
