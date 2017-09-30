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
        args: additional arguments returned from model.forward(s)
    """

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class Policy(nn.Module):
    
    """ Transforms a nn.Module into a Policy that returns an Action(). """

    def __init__(self, model, returns_args=False, *args, **kwargs):
        super(Policy, self).__init__()
        self.model = model
        self.returns_args = returns_args

    def forward(self, x, *args, **kwargs):
        out = self.model(x, *args, **kwargs)
        returns =[None, ] 
        if self.returns_args:
            returns = out[1:]
            out = out[0]
        return Action(raw=out, returns=returns, args=args, kwargs=kwargs)


class DiscretePolicy(Policy):

    """ Converts a policy with continuous outputs to a discrete one. """

    def __init__(self, *args, **kwargs):
        super(DiscretePolicy, self).__init__(*args, **kwargs)

    def forward(self, x, *args, **kwargs):
        action = super(DiscretePolicy, self).forward(x, *args, **kwargs)
        probs = F.softmax(action.raw)
        action.value = probs.multinomial().data[:, 0].tolist()
        action.prob = probs[:, action.value][0].unsqueeze(0)
        action.log_prob = F.log_softmax(action.raw)[:, action.value][0].unsqueeze(0)
        action.entropy = -(action.prob * action.log_prob)
        return action


class DiagonalGaussianPolicy(Policy):

    """ Similar to the ones in Schulman. """

    def __init__(self, model, action_size=1, init_value=-0.0, *args, **kwargs):
        super(DiagonalGaussianPolicy, self).__init__(model, *args, **kwargs)
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

    def forward(self, x, *args, **kwargs):
        action = super(DiagonalGaussianPolicy, self).forward(x, *args, **kwargs)
        size = action.raw.size()
        std = self.logstd.exp().expand_as(action.raw)
        value = action.raw + std * V(th.randn(size))
        value = value.detach()
        action.value = value
        action.prob = self._normal(value, action.raw, self.logstd)
        action.log_prob = action.prob.log1p() 
        action.entropy = self.logstd + self.halflog2pie
        action.logstd = self.logstd.clone()
        action.compute_log_prob = lambda a: -0.5 * ((a - action.raw) / std).pow(2) - 0.5 * log(2*pi) - action.logstd
        action.log_prob = action.compute_log_prob(value)
        return action


class ContinuousPolicy(DiagonalGaussianPolicy):
    pass
