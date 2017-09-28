#!/usr/bin/env python
from __future__ import print_function

"""
    Some algorithmic utility functions.
"""

import torch as th
from torch.autograd import Variable as V
from math import log

EPSILON = 1e-6
PI = 3.141592654
LOG2PI = log(2.0 * PI)


class DiscountedAdvantage(object):

    def __init__(self, gamma=0.99, normalize=True):
        self.gamma = gamma
        self.normalize = normalize

    def __call__(self, rewards, critics, *args, **kwargs):
        discounted = discount(rewards, self.gamma)
        if self.normalize:
            discounted = normalize(discounted)
        critics = th.cat(critics, 1)[0]
        discounted = V(discounted)
        advantage = discounted - critics
        return advantage


class GeneralizedAdvantageEstimation(object):

    def __init__(self, gamma=0.99, tau=0.97, normalize=True):
        self.gamma = gamma
        self.tau = tau
        self.normalize = normalize

    def __call__(self, rewards, critics, *args, **kwargs):
        discounted = generalized_advantage_estimations(rewards,
                                                       critics,
                                                       self.gamma,
                                                       self.tau)
        if self.normalize:
            discounted = normalize(discounted)
        return discounted


def discount(rewards, gamma):
    R = 0.0
    discounted = []
    for r in rewards[::-1]:
        R = r + gamma * R
        discounted.insert(0, R)
    return th.Tensor(discounted)


def generalized_advantage_estimations(rewards, critics, gamma, tau):
    gaes = [V(th.zeros(1)), ]
    gae = V(th.zeros(1))
    prev_c = V(th.zeros(1))
    for r, c in zip(rewards, critics):
        delta = r + gamma * prev_c - c
        gae = gae * gamma * tau + delta
        gaes.insert(0, gae)
        prev_c = c
    gaes = th.cat(gaes).view(-1)
    return gaes


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
