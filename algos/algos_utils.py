#!/usr/bin/env python
from __future__ import print_function

"""
    Some algorithmic utility functions.
"""

import torch as th


class LinearVF(object):

    def __init__(self, num_in, num_out):
        super(LinearVF, self).__init__()
        self.W = th.zeros(num_in, num_out)

    def __call__(self, x):
        x = th.Tensor(x)
        features = self.extract_features(x)
        return th.mm(features, self.W)

    def extract_features(self, x):
        return x

    def learn(self, states, rewards):
        features = [self.extract_features(s) for s in states]
        features = th.cat(features).t()
        rewards = th.cat(rewards)
        lamb = 2.0
        n_col = features.size(1)
        self.W = th.gels(th.mm(features, features) + lamb * th.eye(n_col),
                         th.mm(features, rewards))


def discount(rewards, gamma):
    R = 0
    discounted = []
    for r in reversed(rewards):
        R = R + gamma * r
        discounted.insert(0, R)
    return th.Tensor(discounted)



