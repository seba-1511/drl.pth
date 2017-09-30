#!/usr/bin/env python
from __future__ import print_function

"""
    Some algorithmic utility functions.
"""

import torch as th
from torch import Tensor as T
from torch.autograd import Variable as V
from math import log

EPSILON = 1e-5
PI = 3.141592654
LOG2PI = log(2.0 * PI)


class DiscountedAdvantage(object):

    def __init__(self, gamma=0.99, normalize=True):
        self.gamma = gamma
        self.normalize = normalize

    def __call__(self, rewards, critics, *args, **kwargs):
        discounted = discount(rewards, self.gamma)
        discounted = normalize(discounted)
        advantage = discounted - critics
        if self.normalize:
            advantage = normalize(advantage)
        return discounted, advantage


class GeneralizedAdvantageEstimation(object):

    def __init__(self, gamma=0.99, tau=0.97, normalize=True):
        self.gamma = gamma
        self.tau = tau
        self.normalize = normalize

    def __call__(self, rewards, values, *args, **kwargs):
        advantage = generalized_advantage_estimations(rewards,
                                                values,
                                                self.gamma,
                                                self.tau)
        discounted = discount(rewards, self.gamma)
        if self.normalize:
            advantage = normalize(advantage)
            discounted = normalize(discounted)
        return discounted, advantage


def discount(rewards, gamma):
    tensor = False
    if not isinstance(rewards, list):
        tensor = True
        rewards = rewards.split(1)
    R = 0.0
    discounted = []
    for r in rewards[::-1]:
        R = r + gamma * R
        discounted.insert(0, R)
    if tensor:
        return th.cat(discounted).view(-1)
    return T(discounted)


def generalized_advantage_estimations(rewards, values, gamma, tau):
    rewards = discount(rewards, gamma)
    values = th.cat([values, V(T([0.0]))])
    deltas = rewards + gamma * values[1:] - values[:-1]
    advantage = discount(deltas, gamma * tau)
    return advantage


def normalize(tensor):
    if tensor.size(0) == 1:
        return tensor
    mean = tensor.mean()
    std = tensor.std()
    if isinstance(mean, float):
        return (tensor - mean) / (std + EPSILON)
    else:
        return (tensor - mean.expand_as(tensor)) / (std.expand_as(tensor) + EPSILON)


def gauss_log_prob(means, logstds, x):
    var = th.exp(2 * logstds)
    top = (-(x - means)**2)
    bottom = (2 * var) - 0.5 * LOG2PI - logstds
    gp = top / bottom 
    return th.sum(gp, dim=1)


def logp(x, mean, std):
    out = 0.5 * ((x - mean) / (std))**2 + 0.5 * LOG2PI + th.log(std)
    return -out


def dot_not_flat(A, B):
    """Equivalent of flattening matrices A, B and doing a vector product."""
    return sum([th.sum(a * b) for a, b in zip(A, B)])
