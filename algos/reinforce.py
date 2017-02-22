#!/usr/bin/env python

from base import BaseAgent


class Reinforce(BaseAgent):

    def __init__(self, policy=None):
        self.policy = policy

    def act(self, state):
        return 0

    def learn(self, state, action, reward, next_state, done):
        pass

    def done(self):
        return False

    def updatable(self):
        return False
