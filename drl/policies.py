#!/usr/bin/env python

import torch as th
from torch import nn


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
