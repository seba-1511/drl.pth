#!/usr/bin/env python

import numpy as np
import torch as th
from torch import nn
from torch import Tensor as T
from torch.nn import functional as F
from torch.autograd import Variable as V

from functools import reduce


class Features(nn.Module):

    def __init__(self, state_size, layer_sizes=[128, 128], dropout=0.0):
        super(Features, self).__init__()
        if dropout != 0.0:
            raise Exception('Dropout not supported yet.')
        self.affine1 = nn.Linear(state_size, layer_sizes[0])

    def forward(self, x):
        x = F.relu(self.affine1(x))
        return x
        action_scores = self.action_head(x)
        state_values = self.value_head(x)
        return F.softmax(action_scores), state_values


class Actor(nn.Module):

    def __init__(self, feature_extractor=None, action_size=None, features_size=None):
        super(Actor, self).__init__()
        self.feature_extractor = feature_extractor
        self.linear = nn.Linear(features_size, action_size)

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
        self.affine1 = nn.Linear(state_size, layer_sizes[0])

    def forward(self, x):
        x = F.relu(self.affine1(x))
        return x
        action_scores = self.action_head(x)
        state_values = self.value_head(x)
        return F.softmax(action_scores), state_values


def FC2(state_size, action_size, layer_sizes=[128, 128], dropout=0.0):
    features = Features(state_size, layer_sizes, dropout)
    actor = Actor(feature_extractor=features,
                  features_size=layer_sizes[-1],
                  action_size=action_size)
    critic = Critic(feature_extractor=features,
                    state_size=layer_sizes[-1])
    return (actor, critic)
