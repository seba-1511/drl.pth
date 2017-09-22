#!/usr/bin/env python

import torch as th
import torch.nn as nn
import torch.nn.functional as F


class Action(object):

    """ An Action() is essentially a dictionary structure 
        to pass around information about actions taken by agent.
        
        It may contain the following properties:
        
        raw: the output, straight from the model.
        value: the action to be returned to OpenAI Gym.
        logstd: the logstd used on the raw actions.
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


class ContinuousPolicy(nn.Module):

    """ Converts a policy to a Continuous one. """

    def __init__(self, policy):
        super(ContinuousPolicy, self).__init__()
        self.policy = policy

    def forward(self, x):
        action = self.policy(x)
        action.value = action.raw.data.tolist()
        action.sampled = action.raw
        action.log_sampled = action.sampled.log()
        return action


class DiscretePolicy(nn.Module):

    """ Converts a policy with continuous outputs to a discrete one. """

    def __init__(self, policy):
        super(DiscretePolicy, self).__init__()
        self.policy = policy

    def forward(self, x):
        action = self.policy(x)
        pre_sample = F.softmax(action.raw)
        action.value = pre_sample.multinomial().data[:, 0].tolist()
        action.sampled = pre_sample[:, action.value].mean(0)
        action.log_sampled = F.log_softmax(action.raw)[:, action.value].mean(0)
        return action


class DiagonalGaussianPolicy(nn.Module):

    """ Similar to the ones in Schulman. """

    pass
