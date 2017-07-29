#!/usr/bin/env python

import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor as T
from torch.autograd import Variable as V
from torch.autograd import backward
from math import pi, exp

from .reinforce_with_critic import ActorCriticReinforce
from .algos_utils import discount, normalize, generalized_advantage_estimations


class A3C(ActorCriticReinforce):

    def __init__(self, policy=None, critic=None, gamma=0.99, update_frequency=1000, entropy_weight=0.0001, critic_weight=0.5, grad_norm=40.0, tau=1.0):
        super(A3C, self).__init__(policy=policy, gamma=gamma, 
                                  update_frequency=update_frequency,
                                  entropy_weight=entropy_weight,
                                  critic_weight=critic_weight)
        if critic is None:
            critic = nn.Linear(self.policy.model.critic_state_size, 1)
            critic.bias.data.fill_(0)
            out = th.rand(critic.weight.data.size())
            critic.weight.data = out / th.sqrt(out.pow(2).sum(1).expand_as(out))
        self.critic = critic
        self.grad_norm = grad_norm
        self.tau = tau

    def new_episode(self, terminated=False):
        super(A3C, self).new_episode(terminated=terminated)
        if not terminated:
            self.rewards[-2].append(self.critics[-2][-1].data[0, 0])

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
                    critic_loss += self.critic_weight * advantage**2
                    action.reinforce(gae.data[0,0])
                loss = [entropy_loss, 0.5 * critic_loss] + actions
                backward(loss, [th.ones(1), th.ones(1)] + [None for _ in actions])
        self._reset()
        return [p.grad.clone() for p in self.parameters()]
