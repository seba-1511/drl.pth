#!/usr/bin/env python

import torch as th
import torch.nn.functional as F
from torch.autograd import Variable as V
from torch.autograd import backward
from math import pi, exp

from .base import BaseAgent
from .algos_utils import discount, normalize


class Reinforce(BaseAgent):

    def __init__(self, policy=None, critic=None, gamma=0.99, update_frequency=1000, entropy_weight=0.0001):
        self.policy = policy
        self.gamma = gamma
        if critic is None:
            critic = ConstantCritic(0)
        self.critic = critic
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
        action = self.policy(state)
        return action.value[0, 0], action

    def learn(self, state, action, reward, next_state, done, info=None):
        self.rewards[-1].append(reward)
        self.actions[-1].append(F.log_softmax(info.raw)[0, action])
        self.steps += 1

    def new_episode(self, terminated=False):
        self.rewards.append([])
        self.actions.append([])
#        self.policy.reset()

    def get_update(self):
        R = 0
        rewards = []
        rewards = []
        for r in self.rewards[0][::-1]:
            R = r + self.gamma * R
            rewards.insert(0, R)
        rewards = th.Tensor(rewards)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-6)
        loss = 0.0
        for action, r in zip(self.actions[0], rewards):
            loss = loss - action * r
        backward(loss, [None for _ in self.actions])
        self._reset()
        return [p.grad.clone() for p in self.parameters()]
        # loss = 0.0
        # for actions, rewards in zip(self.actions, self.rewards):
        #     if len(actions) > 0:
        #         rewards = normalize(discount(rewards, self.gamma))
        #         entropy_loss = -(self.entropy_weight * sum(entropies)).sum()
        #         for action, reward in zip(actions, rewards):
        #             action.reinforce(reward)
        #         loss = [entropy_loss, ] + actions
        #         backward(loss, [th.ones(1)] + [None for _ in actions])
        # self._reset()
        # return [p.grad.clone() for p in self.parameters()]

    def updatable(self):
        if self.update_frequency > 0:
            if self.steps >= self.update_frequency:
                return True
        else:
            if len(self.actions) > 1:
                return True
        return False
