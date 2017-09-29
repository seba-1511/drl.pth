#!/usr/bin/env python

from torch import nn


class BaseAgent(nn.Module):

    """
    The base of all agents.
    """

    def __init__(self):
        super(BaseAgent, self).__init__()

    def reset_stats(self, stats):
        """ Resets all statistics or sets them to `stats`. """
        raise NotImplementedError()

    def get_stats(self, stats):
        """ Returns all statistics. """
        return self.stats

    def forward(self, state, *args, **kwargs):
        """ Returns an action to be taken. """
        raise NotImplementedError()

    def learn(self, state=None, action=None, reward=None, next_state=None, done=None, info=None):
        """ Given (s, a, r, s') tuples, does the necessary to compute the update. """
        pass

    def new_episode(self, terminated=False):
        """ Indicates to the agent that a new episode is about to start."""
        pass

    def done(self):
        """ Tells whether the agents needs to continue training. """
        return False

    def updatable(self):
        """ Returns whether the agent is ready to be updated. """
        return False

    def update(self, update):
        """ Applies the update to the parameters. """
        for param, updt in zip(self.parameters, update):
            param.data.add_(updt)

    def get_update(self):
        """ Returns the parameter update from local experience."""
        return None

    def set_gradients(self, gradients):
        """ Sets gradients of the parameters. """
        for param, grad in zip(self.parameters(), gradients):
            param.grad.data[:] = grad.data
