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

    """ 
    Functions that compute the advantage return two arguments:

    rewards: the targets of the value function
    advantages: the values to use for the policy loss
    """

    def __init__(self, gamma=0.99, normalize=True):
        self.gamma = gamma
        self.normalize = normalize

    def __call__(self, all_rewards, all_critics, all_terminals=None, *args, **kwargs):
        discounted = []
        advantage = []
        discounted = [V(discount(r, self.gamma)) for r in all_rewards if len(r) > 0]
        all_discounted = th.cat(discounted).view(-1)
        d_mean, d_std = all_discounted.mean(), all_discounted.std()
        discounted = [(d-d_mean) / (d_std+EPSILON) for d in discounted]
        advantage = [d - th.cat(c).view(-1) for d, c in zip(discounted, all_critics) if len(c) > 0]
        all_advantage = th.cat(advantage).view(-1)
        a_mean, a_std = all_advantage.mean(), all_advantage.std()
        advantage = [(a - a_mean) / (a_std + EPSILON) for a in advantage]
        return discounted, advantage


#class DiscountedAdvantage(object):
#
#    """ 
#    Functions that compute the advantage return two arguments:
#
#    rewards: the targets of the value function
#    advantages: the values to use for the policy loss
#    """
#
#    def __init__(self, gamma=0.99, normalize=True):
#        self.gamma = gamma
#        self.normalize = normalize
#
#    def __call__(self, rewards, critics, terminals=None, *args, **kwargs):
#        if terminals is not None:
#            start = 0
#            discounted = T()
#            while terminals[start:].index(1) != -1:
#                end = terminals[start:].index(1)
#                discounted = th.cat([discounted, discount(rewards[start:end+1], self.gamma)])
#                start = end
#        else:
#            discounted = discount(rewards, self.gamma)
#        discounted = normalize(discounted)
#        advantage = discounted - critics
#        if self.normalize:
#            advantage = normalize(advantage)
#        return discounted, advantage


class GeneralizedAdvantageEstimation(object):

    """ 
    Functions that compute the advantage return two arguments:

    rewards: the targets of the value function
    advantages: the values to use for the policy loss
    """

    def __init__(self, gamma=0.99, tau=0.95, normalize=False, discount=False):
        self.gamma = gamma
        self.tau = tau
        self.normalize = normalize
        self.discounted = discount

    def __call__(self, all_rewards, all_values, terminal=None, *args, **kwargs):
        rew_vec = []
        val_vec = []
        terminals = []
        for r, v, t in zip(all_rewards, all_values, terminal):
            if len(r) > 0:
                rew_vec += r
                val_vec += v
                terminals += t
        rew_vec = V(T(rew_vec)).view(-1)
        val_vec = th.cat(val_vec).view(-1)
        advantage = generalized_advantage_estimations(rew_vec, val_vec, terminals, self.gamma, self.tau).view(-1)
        start = 0
        adv_list = []
        for l in all_rewards:
            l = len(l)
            if l > 0:
                adv_list.append(advantage[start:start+l])
                start += l
        mean = advantage.mean()
        std = advantage.std()
        return adv_list, [(a - mean) / (std + EPSILON) for a in adv_list]



        if self.discounted:
            rewards = discount(rewards, self.gamma)
        if self.normalize:
            advantage = normalize(advantage)
            rewards = normalize(discounted)
        return rewards, advantage


#class GeneralizedAdvantageEstimation(object):
#
#    """ 
#    Functions that compute the advantage return two arguments:
#
#    rewards: the targets of the value function
#    advantages: the values to use for the policy loss
#    """
#
#    def __init__(self, gamma=0.99, tau=0.95, normalize=False, discount=False):
#        self.gamma = gamma
#        self.tau = tau
#        self.normalize = normalize
#        self.discounted = discount
#
#    def __call__(self, rewards, values, terminal=None, *args, **kwargs):
#        advantage = generalized_advantage_estimations(rewards, values, terminal, self.gamma, self.tau)
#        if self.discounted:
#            rewards = discount(rewards, self.gamma)
#        if self.normalize:
#            advantage = normalize(advantage)
#            rewards = normalize(discounted)
#        return rewards, advantage


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


def generalized_advantage_estimations(rewards, values, terminal=None, gamma=0.99, tau=0.95):
    gae = 0.0
    advantages = []
    values = th.cat([values, V(T([0.0077]))])
    for i in reversed(range(len(rewards))):
        nonterminal = 1.0 - terminal[i]
        delta = rewards[i] + gamma * values[i+1] * nonterminal - values[i]
        gae = delta + gamma * tau * gae * nonterminal
        advantages.insert(0, gae + values[i])
    return th.cat(advantages) 


def vectorized_generalized_advantage_estimations(rewards, values, gamma, tau):
#    rewards = discount(rewards, gamma)
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
