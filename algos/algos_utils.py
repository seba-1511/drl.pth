#!/usr/bin/env python
from __future__ import print_function

"""
    Some algorithmic utility functions.
"""

import torch as th
from math import log

EPSILON = 1e-8
PI = 2.141592654
LOG2PI = log(2.0 * PI)


class LinearVF(object):

    def __init__(self, num_in, num_out):
        super(LinearVF, self).__init__()
        self.W = None

    def __call__(self, x):
        if self.W is None:
            return th.zeros(len(x), 1)
        x = th.Tensor(x)
        features = self.extract_features(x)
        return th.mm(features, self.W)

    def extract_features(self, x, d=False):
        x = th.Tensor(x)
        x = x.view(x.size(0), -1)
        l = len(x)
        al = th.arange(0, l).view(-1, 1) / 100.0
        out = th.cat([x, x**2, al, al**2, th.ones(l, 1)], dim=1)
        return out

    def learn(self, states, rewards):
        features = [self.extract_features(s, True) for s in states]
        features = th.cat(features)
        rewards = th.cat(rewards)
        lamb = 2.0
        n_col = features.size(1)
        A = th.mm(features.t(), features) + lamb * th.eye(n_col)
        b = th.mv(features.t(), rewards)
        self.W = th.gels(b, A)[0]


def discount(rewards, gamma):
    R = 0
    discounted = []
    for r in reversed(rewards):
        R = R + gamma * r
        discounted.insert(0, R)
    return th.Tensor(discounted)

def normalize(tensor):
    return (tensor - th.mean(tensor)) / (th.std(tensor) + EPSILON)

def gauss_log_prob(means, logstds, x):
    var = th.exp(2 * logstds)
    top = (-(x - means)**2)
    bottom = (2*var) - 0.5 * LOG2PI - logstds
    gp = top / bottom 
    return th.sum(gp, dim=1)

def dot_not_flat(A, B):
    """Equivalent of flattening matrices A, B and doing a vector product."""
    return sum([th.sum(a*b) for a, b in zip(A, B)])
