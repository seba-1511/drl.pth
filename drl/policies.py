#!/usr/bin/env python

import torch as th
import torch.nn as nn
import torch.nn.functional as F


class StochasticPolicy(nn.Module):

    """
    Transforms a model into a stochastic continuous policy, with sampling
    from the logstd.

    `forward` returns a tuple, the mean and the logstd.
    """

    def __init__(self, model):
        super(StochasticPolicy, self).__init__()
        self.model = model
        self.params = nn.ParameterList(list(model.parameters()))
        # self.logstd = nn.Parameter(th.zeros(model.num_out))
        self.logstd = nn.Parameter(-5.0 + th.zeros(model.num_out))
        self.params.extend([self.logstd, ])
        print('Optimizing ', len(list(self.parameters())), ' parameters')

    def forward(self, x):
        x = self.model(x)
        return x, self.logstd

    def reset(self):
        """ Resets the state of the current policy. """
        self.model.reset_state()


class DropoutPolicy(nn.Module):

    """
    Transforms a model into a stochastic continuous policy, with sampling
    from the logstd.

    `forward` returns a tuple, the mean and the logstd which are obtained by 
    forwarding through sampled dropout masks.
    """

    def __init__(self, model, num_samples=10):
        super(DropoutPolicy, self).__init__()
        self.model = model
        self.params = nn.ParameterList(list(model.parameters()))
        self.num_samples = num_samples
        print('Optimizing ', len(list(self.parameters())), ' parameters')

    def forward(self, x):
        samples = []
        state = [0.0 for _ in self.model.get_state()]
        for _ in range(self.num_samples):
            out, new_state = self.model.forgetful_forward(x)
            samples.append(out)
            state = [s + u for s, u in zip(state, new_state)]
        samples = th.cat(samples, 0)
        state = [s / self.num_samples for s in state]
        self.model.set_state(state)
        mu = th.mean(samples, 0)
        std = th.std(samples, 0)
        return mu, th.log(std)
    
    def reset(self):
        self.model.reset_state()


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
    
    """ Transforms a nn.Module into a Policy that returns an Action()."""

    def __init__(self, model):
        super(Policy, self).__init__()
        self.model = model

    def forward(self, x):
        activations = self.model(x)
        action = Action(raw=activations)
        return action


class DiscretePolicy(nn.Module):

    def __init__(self, model):
        super(DiscretePolicy, self).__init__()
        self.model = model

    def forward(self, x):
        action = self.model(x)
        action.value = F.softmax(action.raw).multinomial().data
        return action


