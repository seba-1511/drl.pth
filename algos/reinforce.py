#!/usr/bin/env python

import torch as th
from torch.autograd import Variable as V

from .base import BaseAgent


class Reinforce(BaseAgent):

    def __init__(self, policy=None, baseline=1.0, gamma=0.99, update_frequency=1000):
        self.policy = policy
        self.gamma = gamma
        self.baseline = baseline
        self.update_frequency = update_frequency
        self._reset()
        self.update_ready = False

    def _reset(self):
        self.rewards = []
        self.states = []

    def parameters(self):
        return self.policy.parameters()

    def act(self, state):
        state = V(th.from_numpy(state).float().unsqueeze(0))
        action = self.policy.forward(state)
        return action.data.numpy(), None

    def learn(self, state, action, reward, next_state, done, info=None):
        self.states.append(state)
        self.rewards.append(reward)
        if done:
            self.update_ready = True

    def get_update(self):
        R = 0
        for s, r in reversed(list(zip(self.states, self.rewards))):
            R = r + self.gamma * R
            s = V(th.from_numpy(s).float().unsqueeze(0))
            preds = self.policy.forward(s)
            log_preds = th.log(preds) * (R - self.baseline)
            th.sum(log_preds).backward()
        return [p.grad.clone() for p in self.parameters()]

    def updatable(self):
        if self.update_ready:
            self.update_ready = False
            return True
        return False
