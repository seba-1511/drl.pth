#!/usr/bin/env python

class BaseAgent(object):

    def act(self, state):
        """ Returns an action to be taken. """
        raise NotImplementedError()

    def learn(self, state=None, action=None, reward=None, next_state=None, done=None):
        """ Given (s, a, r, s') tuples, oes the necessary to compute the update. """
        raise NotImplementedError()

    def done(self):
        """ Tells whether the agents needs to continue training. """
        return False

    def update(self, update):
        """ Applies the update to the parameters. """
        raise NotImplementedError()

    def get_update(self):
        """ Returns the parameter update from local experience."""
        raise NotImplementedError()
