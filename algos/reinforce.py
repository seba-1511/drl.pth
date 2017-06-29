#!/usr/bin/env python

import torch as th
from torch.autograd import Variable as V
from torch.autograd import backward
from math import pi, exp

from .base import BaseAgent
from .algos_utils import discount, normalize


class Reinforce(BaseAgent):

    def __init__(self, policy=None, baseline=0.5, gamma=0.99, update_frequency=1000, entropy_weight=0.0001):
        self.policy = policy
        self.gamma = gamma
        self.baseline = baseline
        self.entropy_weight = entropy_weight
        self.update_frequency = update_frequency
        self._reset()
        self.update_ready = False
        self.log2pie = th.log(th.Tensor([2.0 * pi * exp(1.0)]))

    def _reset(self):
        self.steps = 0
        self.rewards = [[], ]
        self.entropies = [[], ]
        self.actions = [[], ]

    def parameters(self):
        return self.policy.parameters()

    def act(self, state):
        state = V(th.from_numpy(state).float().unsqueeze(0))
        mean, logstd = self.policy.forward(state)
        std = th.exp(logstd)
        entropy = 0.5 * (V(self.log2pie.expand_as(std)) + 2.0 * logstd)
        action = th.normal(mean, std)
        return action.data.numpy(), (action, entropy)

    def learn(self, state, action, reward, next_state, done, info=None):
        action, entropy = info
        self.rewards[-1].append(reward)
        self.entropies[-1].append(entropy)
        self.actions[-1].append(action)
        self.steps += 1

    def new_episode(self, terminated=False):
        self.rewards.append([])
        self.actions.append([])
        self.entropies.append([])
        self.policy.reset()

    def get_update(self):
        loss = 0.0
        for actions, rewards, entropies in zip(self.actions, self.rewards, self.entropies):
            if len(actions) > 0:
                rewards = normalize(discount(rewards, self.gamma))
                entropy_loss = -(self.entropy_weight * sum(entropies)).sum()
                for action, reward in zip(actions, rewards):
                    action.reinforce(reward - self.baseline)
                loss = [entropy_loss, ] + actions
                backward(loss, [th.ones(1)] + [None for _ in actions], retain_variables=True)
        self._reset()
        return [p.grad.clone() for p in self.parameters()]

    def updatable(self):
        if self.update_frequency > 0:
            if self.steps >= self.update_frequency:
                return True
        else:
            if len(self.actions) > 1:
                return True
        return False
