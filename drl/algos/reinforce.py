#!/usr/bin/env python

import torch as th
import torch.nn.functional as F


from torch import Tensor as T
from torch.autograd import Variable as V
from torch.autograd import backward
from math import pi, exp
from itertools import chain

from .base import BaseAgent
from .algos_utils import DiscountedAdvantage, normalize, EPSILON

from ..models import ConstantCritic


class Reinforce(BaseAgent):

    def __init__(self, policy=None, critic=None, advantage=None, update_frequency=1000, entropy_weight=0.01, critic_weight=0.5,
                 grad_clip=50.0):
        super(Reinforce, self).__init__()
        self.policy = policy
        if critic is None:
            critic = ConstantCritic(0)
        self.critic = critic
        if advantage is None:
            advantage = DiscountedAdvantage()
        self.advantage = advantage
        self.entropy_weight = entropy_weight
        self.critic_weight = critic_weight
        self.update_frequency = update_frequency
        self.grad_clip = grad_clip
        self._reset()

    def _reset(self):
        self.steps = 0
        self.rewards = [[], ]
        self.entropies = [[], ]
        self.actions = [[], ]
        self.critics = [[], ]

    def parameters(self):
        parameters = chain(self.policy.parameters(),
                           self.critic.parameters())
        return parameters

    def _variable(self, state):
        state = th.from_numpy(state).float()
        if len(state.size()) < 2:
            state = state.unsqueeze(0)
        return V(state)

    def act(self, state, *args, **kwargs):
        state = self._variable(state)
        action = self.policy(state, *args, **kwargs)
        return action.value[0], action

    def learn(self, state, action, reward, next_state, done, info=None):
        self.rewards[-1].append(reward)
        self.actions[-1].append(info.log_prob)
        self.critics[-1].append(self.critic(self._variable(state), *info.args, **info.kwargs))
        self.entropies[-1].append(info.entropy)
        self.steps += 1

    def new_episode(self, terminated=False):
        self.rewards.append([])
        self.actions.append([])
        self.critics.append([])
        self.entropies.append([])

    def get_update(self):
        for actions_ep, rewards_ep, critics_ep, entropy_ep in zip(self.actions, self.rewards, self.critics, self.entropies):
            if len(actions_ep) > 0:
                advantage_ep = self.advantage(rewards_ep, critics_ep)
                critic_loss = advantage_ep.pow(2).sum()
                entropy_loss = th.cat(entropy_ep).mean()
                policy_loss = 0.0
                for action_log, advantage in zip(actions_ep, advantage_ep):
                    policy_loss = policy_loss + action_log.sum() * advantage.data[0]
                critic_loss = self.critic_weight * critic_loss
                entropy_loss = self.entropy_weight * entropy_loss
                loss = - policy_loss + critic_loss - entropy_loss
                loss.backward()
                th.nn.utils.clip_grad_norm(self.parameters(), self.grad_clip)
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
