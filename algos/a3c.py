#!/usr/bin/env python

import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor as T
from torch.autograd import Variable as V
from torch.autograd import backward
from math import pi, exp

from .reinforce import Reinforce
from .algos_utils import discount, normalize, generalized_advantage_estimations

"""
TODO:
    * properly ocmpute the entropy
    * check model
    * try with A3C
"""


class A3C(Reinforce):

    def __init__(self, policy=None, critic=None, gamma=0.99, update_frequency=1000, entropy_weight=0.0001, grad_norm=40.0, tau=1.0):
        super(A3C, self).__init__(policy=policy, gamma=gamma, 
                                  update_frequency=update_frequency,
                                  entropy_weight=entropy_weight)
        if critic is None:
            critic = nn.Linear(self.policy.model.critic_state_size, 1)
        self.critic = critic
        self.grad_norm = grad_norm
        self.tau = tau

    def _reset(self):
        super(A3C, self)._reset()
        self.critics = [[], ]

    def parameters(self):
        for p in self.critic.parameters():
            yield p
        for p in self.policy.parameters():
            yield p

    def act(self, state):
        action, info = super(A3C, self).act(state)
        critic = self.critic.forward(self.policy.model.critic_state.view(1, -1))
        info = info + (critic, )
        return action, info

    def learn(self, state, action, reward, next_state, done, info=None):
        action, entropy, critic = info
        super(A3C, self).learn(state, action, reward, next_state, done, 
                                     info=(action, entropy))
        self.critics[-1].append(critic)

    def new_episode(self, terminated=False):
        super(A3C, self).new_episode(terminated=terminated)
        self.critics.append([])

    def get_update(self):
        loss = 0.0
        iterator = zip(self.actions, self.rewards, self.entropies, self.critics)
        for actions, rewards, entropies, critics in iterator:
            if len(actions) > 0:
                rewards = normalize(discount(rewards, self.gamma))
                gaes = generalized_advantage_estimations(rewards, critics,
                                                         self.gamma, self.tau)
                entropy_loss = -(self.entropy_weight * sum(entropies)).sum()
                critic_loss = 0.0
                for action, reward, critic, gae in zip(actions, rewards, critics, gaes):
                    advantage = V(T([reward])) - critic
                    critic_loss += 0.5 * advantage**2
                    action.reinforce(gae.data[0,0])
                loss = [entropy_loss, 0.5 * critic_loss] + actions
                backward(loss, [th.ones(1), th.ones(1)] + [None for _ in actions])
                nn.utils.clip_grad_norm(self.parameters(), self.grad_norm)
        self._reset()
        return [p.grad.clone() for p in self.parameters()]
