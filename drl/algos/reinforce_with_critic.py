#!/usr/bin/env python

import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor as T
from torch.autograd import Variable as V
from torch.autograd import backward
from math import pi, exp

from .reinforce import Reinforce
from .algos_utils import discount, normalize


class ActorCriticReinforce(Reinforce):

    def __init__(self, policy=None, critic=None, gamma=0.99, update_frequency=1000, entropy_weight=0.0001, critic_weight=0.5):
        super(ActorCriticReinforce, self).__init__(policy=policy, gamma=gamma, update_frequency=update_frequency, entropy_weight=entropy_weight)
        if critic is None:
            critic = nn.Linear(self.policy.model.critic_state_size, 1, bias=False)
            out = th.randn(critic.weight.size())
            # PPO Init
            out *= (1.0/(out**2).sum(1)**0.5).expand_as(critic.weight)
            critic.weight.data = out
        self.critic = critic
        self.critic_weight = critic_weight

    def _reset(self):
        super(ActorCriticReinforce, self)._reset()
        self.critics = [[], ]

    def parameters(self):
        for p in self.critic.parameters():
            yield p
        for p in self.policy.parameters():
            yield p

    def act(self, state):
        action, info = super(ActorCriticReinforce, self).act(state)
        critic = self.critic.forward(self.policy.model.critic_state.view(1, -1))
        info = info + (critic, )
        return action, info

    def learn(self, state, action, reward, next_state, done, info=None):
        action, entropy, critic = info
        super(ActorCriticReinforce, self).learn(state, action, reward, next_state, done, 
                                     info=(action, entropy))
        self.critics[-1].append(critic)

    def new_episode(self, terminated=False):
        super(ActorCriticReinforce, self).new_episode(terminated=terminated)
        self.critics.append([])

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
                    critic_loss -= self.critic_weight * F.smooth_l1_loss(critic, V(T([reward])))
                loss = [entropy_loss, critic_loss] + actions
                backward(loss, [th.ones(1), th.ones(1)] + [None for _ in actions])
        self._reset()
        return [p.grad.clone() for p in self.parameters()]
