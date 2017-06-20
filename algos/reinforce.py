#!/usr/bin/env python

import torch as th
from torch.autograd import Variable as V
from torch.autograd import backward
from math import pi

from .base import BaseAgent
from .algos_utils import discount, EPSILON


class Reinforce(BaseAgent):

    def __init__(self, policy=None, baseline=1.0, gamma=0.99, update_frequency=1000):
        self.policy = policy
        self.gamma = gamma
        self.baseline = baseline
        self.update_frequency = update_frequency
        self._reset()
        self.update_ready = False
        self.log2pi = th.log(th.Tensor([2.0 * pi]))

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
        entropy = V(0.5 * (1.0 + self.log2pi)) + logstd
        action = th.normal(mean, std)
        return action.data.numpy(), (action, entropy)

    def learn(self, state, action, reward, next_state, done, info=None):
        action, entropy = info
        self.rewards[-1].append(reward)
        self.entropies[-1].append(entropy)
        # self.actions[-1].append(action)
        self.actions[-1].append(info[0])
        self.steps += 1

    def new_episode(self, terminated=False):
        self.rewards.append([])
        self.actions.append([])
        self.entropies.append([])

    def get_update(self):
        loss = 0.0
        for actions, rewards, entropies in zip(self.actions, self.rewards, self.entropies):
            if len(actions) > 0:
                R = th.zeros(1)
                episode_loss = 0.0
                rewards = discount(rewards, self.gamma)
                rewards = (rewards - rewards.mean()) / (rewards.std() + EPSILON)
                # iterator = reversed(list(zip(actions, rewards, entropies)))
                iterator = list(zip(actions, rewards, entropies))
                for action, reward, entropy in iterator:
                    # R = reward + self.gamma * R
                    # R = th.Tensor([reward])
                    action.reinforce(reward + (0.0001 * entropy).data.sum())
                    # episode_loss -= (th.log(action) * V(R)).sum() - (0.0001 * entropy).sum()
                # episode_loss /= len(actions)
                # loss += episode_loss
                backward(actions, [None for _ in actions])
        # loss.backward()
        self._reset()
        return [p.grad.clone() for p in self.parameters()]

    def updatable(self):
        if self.steps >= self.update_frequency:
            return True
        return False
