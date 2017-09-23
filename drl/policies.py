#!/usr/bin/env python

import torch as th
import torch.nn as nn
import torch.nn.functional as F

from math import exp, log, pi

from torch import Tensor as T
from torch.nn import Parameter as P
from torch.autograd import Variable as V


class Action(object):

    """ An Action() is essentially a dictionary structure 
        to pass around information about actions taken by agent.
        
        It may contain the following properties:
        
        raw: the output, straight from the model.
        value: the action to be returned to OpenAI Gym.
        probs: the probability of this action.
        log_prob: the log probability of the action.
        entropy: the entropy of the action.
    """

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class Policy(nn.Module):
    
    """ Transforms a nn.Module into a Policy that returns an Action(). """

    def __init__(self, model):
        super(Policy, self).__init__()
        self.model = model

    def forward(self, x):
        activations = self.model(x)
        return Action(raw=activations)


class DiscretePolicy(nn.Module):

    """ Converts a policy with continuous outputs to a discrete one. """

    def __init__(self, policy):
        super(DiscretePolicy, self).__init__()
        self.policy = policy

    def forward(self, x):
        action = self.policy(x)
        probs = F.softmax(action.raw)
        action.value = probs.multinomial().data[:, 0].tolist()
        action.prob = probs[:, action.value][0].unsqueeze(0)
        action.log_prob = F.log_softmax(action.raw)[:, action.value][0].unsqueeze(0)
        action.entropy = -(action.prob * action.log_prob)
        return action


class DiagonalGaussianPolicy(nn.Module):

    """ Similar to the ones in Schulman. """

    def __init__(self, policy, action_size, init_value=-3.0):
        super(DiagonalGaussianPolicy, self).__init__()
        self.policy = policy
        self.init_value = init_value
        self.logstd = th.randn((1, action_size)) + self.init_value
        self.logstd = P(self.logstd)
        self.halflog2pie = V(T([2 * pi * exp(1)])) * 0.5
        self.pi = V(T([pi]))

    def _normal(self, x, mean, logstd):
        std = logstd.exp()
        std_sq = std.pow(2)
        a = (-(x - mean).pow(2) / (2 * std_sq)).exp()
        b = (2 * std_sq * self.pi.expand_as(std_sq)).sqrt()
        return a / b

    def forward(self, x):
        action = self.policy(x)
        size = action.raw.size()
        value = action.raw + self.logstd.exp().expand_as(action.raw) * V(th.randn(size))
        value = value.detach()
        action.value = value.data.tolist()
        action.prob = self._normal(value, action.raw, self.logstd)
        action.log_prob = action.prob.log1p() 
        action.entropy = self.logstd + self.halflog2pie
        return action


class ContinuousPolicy(DiagonalGaussianPolicy):
    pass
