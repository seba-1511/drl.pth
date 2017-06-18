#!/usr/bin/env python

import torch as th
from torch.autograd import Variable as V

from .base import BaseAgent


class A3C(BaseAgent):

    def __init__(self, policy=None, baseline=None, gamma=0.99, update_frequency=5):
        self.policy = policy
        self.gamma = gamma
        self.update_frequency = update_frequency
        self._reset()
        self.update_ready = False

    def _reset(self):
        self.states = []
        self.rewards = []

    def parameters(self):
        return self.policy.parameters()

    def act(self, state):
        state = V(th.from_numpy(state).float().unsqueeze(0))
        action = self.policy.forward(state)
        return action.data.numpy()
