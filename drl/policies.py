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
        action = Action(raw=activations)
        return action


class DiscretePolicy(nn.Module):

    """ Converts a policy with continuous outputs to a discrete one. """

    def __init__(self, model):
        super(DiscretePolicy, self).__init__()
        self.model = model

    def forward(self, x):
        action = self.model(x)
        action.value = F.softmax(action.raw).multinomial().data
        action.log_raw = F.log_softmax(action.raw)
        return action


