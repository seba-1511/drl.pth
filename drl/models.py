#!/usr/bin/env python

import numpy as np
import torch as th
from torch import nn
from torch import Tensor as T
from torch.nn import functional as F
from torch.autograd import Variable as V

from functools import reduce


class NoOp(nn.Module):
    def __init__(self):
        super(NoOp, self).__init__()

    def forward(self, x):
        return x


class DiscreteFeatures(nn.Module):

    def __init__(self, state_size, layer_sizes=[128, 128], dropout=0.0):
        super(DiscreteFeatures, self).__init__()
        if dropout != 0.0:
            raise Exception('Dropout not supported yet.')
        self.affine1 = nn.Linear(state_size, layer_sizes[0])

    def forward(self, x, *args, **kwargs):
        x = F.relu(self.affine1(x))
        return x


class ContinuousFeatures(nn.Module):

    def __init__(self, state_size, layer_sizes=[128, 128], dropout=0.0):
        super(ContinuousFeatures, self).__init__()
        if dropout != 0.0:
            raise Exception('Dropout not supported yet.')
        self.affine1 = nn.Linear(state_size, layer_sizes[0])

    def forward(self, x, *args, **kwargs):
        x = F.tanh(self.affine1(x))
        return x


class DiscreteActor(nn.Module):

    def __init__(self, feature_extractor=None, action_size=None, features_size=None, recurrent=False):
        super(DiscreteActor, self).__init__()
        self.feature_extractor = feature_extractor
        self.linear = nn.Linear(features_size, action_size)
        self.recurrent = recurrent

    def forward(self, x, *args, **kwargs):
        if self.recurrent:
            features, states = self.feature_extractor(x, *args, **kwargs)
            return self.linear(features), states
        else:
            features = self.feature_extractor(x, *args, **kwargs)
            return self.linear(features)


class ContinuousActor(nn.Module):

    def __init__(self, feature_extractor=None, action_size=None, features_size=None, recurrent=False):
        super(ContinuousActor, self).__init__()
        self.feature_extractor = feature_extractor
        self.linear = nn.Linear(features_size, action_size, bias=False)
        self.recurrent = recurrent
        self.linear.weight.data *= 0.1

    def forward(self, x, *args, **kwargs):
        if self.recurrent:
            features, states = self.feature_extractor(x, *args, **kwargs)
            return self.linear(features), states
        else:
            features = self.feature_extractor(x, *args, **kwargs)
            return self.linear(features)


class Critic(nn.Module):

    def __init__(self, feature_extractor, state_size, recurrent=False):
        super(Critic, self).__init__()
        self.feature_extractor = feature_extractor
        self.linear = nn.Linear(state_size, 1)
        self.recurrent = recurrent

    def forward(self, x, *args, **kwargs):
        if self.recurrent:
            features, states = self.feature_extractor(x, *args, **kwargs)
            return self.linear(features)
        else:
            features = self.feature_extractor(x)
            return self.linear(features)


class ConstantCritic(nn.Module):

    def __init__(self, value=0.0):
        super(ConstantCritic, self).__init__()
        self.value = V(th.zeros(1, 1) + value) 

    def forward(self, x, *args, **kwargs):
        return self.value


class LSTMDiscreteFeatures(nn.Module):

    def __init__(self, state_size, layer_sizes=[128, 128], dropout=0.0):
        super(LSTMDiscreteFeatures, self).__init__()
        if dropout != 0.0:
            raise Exception('Dropout not supported yet.')
        self.lstm = nn.LSTMCell(state_size, layer_sizes[0], bias=False)

    def forward(self, x, hiddens):
        hx, cx = self.lstm(x, hiddens)
        return hx, (hx, cx) 


class LSTMContinuousFeatures(nn.Module):

    def __init__(self, state_size, layer_sizes=[128, 128], dropout=0.0):
        super(LSTMContinuousFeatures, self).__init__()
        if dropout != 0.0:
            raise Exception('Dropout not supported yet.')
        self.lstm = nn.LSTMCell(state_size, layer_sizes[0], bias=False)
        self.lstm.weight_hh.data *= 0.1
        self.lstm.weight_ih.data *= 0.1

    def forward(self, x, hiddens):
        hx, cx = self.lstm(x, hiddens)
        return hx, (hx, cx) 

class BaselineActor(nn.Module):

    def __init__(self, state_size, action_size):
        super(BaselineActor, self).__init__()
        self.fc1 = nn.Linear(state_size, 64, bias=True)
        self.fc2 = nn.Linear(64, 64, bias=True)
        self.mean = nn.Linear(64, action_size, bias=True)

        # Init
        for p in [self.fc1, self.fc2, self.mean]:
            p.weight.data.normal_(0, 1)
            p.weight.data *= 1.0 / th.sqrt(p.weight.data.pow(2).sum(1, keepdim=True))
            p.bias.data.mul_(0.0)

        self.mean.weight.data.mul_(0.01)

    def forward(self, x, *args, **kwargs):
        x = F.tanh(self.fc1(x)) 
        x = F.tanh(self.fc2(x))
        return self.mean(x)



class BaselineCritic(nn.Module):

    def __init__(self, state_size):
        super(BaselineCritic, self).__init__()
        self.fc1 = nn.Linear(state_size, 64, bias=True)
        self.fc2 = nn.Linear(64, 64, bias=True)
        self.value = nn.Linear(64, 1, bias=True)

        # Init
        for p in [self.fc1, self.fc2, self.value]:
            p.weight.data.normal_(0, 1)
            p.weight.data *= 1.0 / th.sqrt(p.weight.data.pow(2).sum(1, keepdim=True))
            p.bias.data.mul_(0.0)

    def forward(self, x, *args, **kwargs):
        x = F.tanh(self.fc1(x)) 
        x = F.tanh(self.fc2(x))
        return self.value(x)


def Baseline(state_size, action_size, layer_sizes=[128, 128], dropout=0.0, discrete=True):
    actor = BaselineActor(state_size, action_size)
    critic = BaselineCritic(state_size)
    return actor, critic

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


def LSTM2(state_size, action_size, layer_sizes=[128, 128], dropout=0.0, discrete=True):
    if discrete:
        features = LSTMDiscreteFeatures(state_size, layer_sizes, dropout)
        actor = DiscreteActor(feature_extractor=features,
                              features_size=layer_sizes[-1],
                              action_size=action_size,
                              recurrent=True)
    else:
        features = LSTMContinuousFeatures(state_size, layer_sizes, dropout)
        actor = ContinuousActor(feature_extractor=features,
                                features_size=layer_sizes[-1],
                                action_size=action_size,
                                recurrent=True)
    critic = Critic(feature_extractor=features,
                    state_size=layer_sizes[-1], recurrent=True)
    return (actor, critic)
