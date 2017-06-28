#!/usr/bin/env python

import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor as T
from torch.autograd import Variable as V
from torch.autograd import backward
from math import pi, exp

from .base import BaseAgent
from .algos_utils import discount, normalize


class ActorCriticReinforce(BaseAgent):

    def __init__(self, policy=None, critic=None, gamma=0.99, update_frequency=1000, entropy_weight=0.0001):
        self.policy = policy
        self.gamma = gamma
        self.entropy_weight = entropy_weight
        self.update_frequency = update_frequency
        self._reset()
        self.update_ready = False
        self.log2pie = th.log(th.Tensor([2.0 * pi * exp(1.0)]))
        if critic is None:
            critic = nn.Linear(self.policy.model.critic_state_size, 1)
        self.critic = critic

    def _reset(self):
        self.steps = 0
        self.critics = [[], ]
        self.rewards = [[], ]
        self.entropies = [[], ]
        self.actions = [[], ]

    def parameters(self):
        for p in self.critic.parameters():
            yield p
        for p in self.policy.parameters():
            yield p

    def act(self, state):
        state = V(th.from_numpy(state).float().unsqueeze(0))
        mean, logstd = self.policy.forward(state)
        std = th.exp(logstd)
        entropy = 0.5 * (V(self.log2pie.expand_as(std)) + 2.0 * logstd)
        action = th.normal(mean, std)
        critic = self.critic.forward(self.policy.model.critic_state.view(1, -1))
        return action.data.numpy(), (action, entropy, critic)

    def learn(self, state, action, reward, next_state, done, info=None):
        action, entropy, critic = info
        self.rewards[-1].append(reward)
        self.entropies[-1].append(entropy)
        self.actions[-1].append(action)
        self.critics[-1].append(critic)
        self.steps += 1

    def new_episode(self, terminated=False):
        self.rewards.append([])
        self.actions.append([])
        self.entropies.append([])
        self.policy.reset()

    def get_update(self):
        loss = 0.0
        iterator = zip(self.actions, self.rewards, self.entropies, self.critics)
        for actions, rewards, entropies, critics in iterator:
            if len(actions) > 0:
                rewards = normalize(discount(rewards, self.gamma))
                entropy_loss = -(self.entropy_weight * sum(entropies)).sum()
                critic_loss = 0.0
                for action, reward, critic in zip(actions, rewards, critics):
                    action.reinforce(reward - critic.data[0, 0])
                    critic_loss += F.smooth_l1_loss(critic, V(T([reward])))
                loss = [entropy_loss, critic_loss] + actions
                backward(loss, [th.ones(1), th.ones(1)] + [None for _ in actions])
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
