#!/usr/bin/env python

import torch as th
import torch.nn.functional as F


from torch import Tensor as T
from torch.autograd import Variable as V
from torch.autograd import backward
from math import pi, exp
from itertools import chain

from .base import BaseAgent
from .algos_utils import DiscountedAdvantage

from ..models import ConstantCritic


class Reinforce(BaseAgent):

    def __init__(self, policy=None, critic=None, advantage=None, update_frequency=1000, entropy_weight=0.001, critic_weight=0.5,
                 grad_clip=0.5):
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
        self.reset_stats()
        self._reset()

    def reset_stats(self, stats=None):
        if stats is None:
            stats = {
                    'Num. Updates': 1e-7,
                    'Num. Trajectories': 1e-7,
                    'Num. Steps': 1e-7,
                    'Total Loss': 0.0,
                    'Policy Loss': 0.0,
                    'Entropy Loss': 0.0,
                    'Critic Loss': 0.0,
            }
        self.stats = stats

    def get_stats(self):
        stats = {
                'Num. Updates': int(self.stats['Num. Updates']),
                'Num. Trajectories': int(self.stats['Num. Trajectories']),
                'Num. Steps': int(self.stats['Num. Steps']),
                'Total Loss': self.stats['Total Loss'] / self.stats['Num. Updates'],
                'Policy Loss': self.stats['Policy Loss'] / self.stats['Num. Updates'],
                'Entropy Loss': self.stats['Entropy Loss'] / self.stats['Num. Updates'],
                'Critic Loss': self.stats['Critic Loss'] / self.stats['Num. Updates'],
        }
        return stats


    def _reset(self):
        self.steps = 0
        self.rewards = [[], ]
        self.entropies = [[], ]
        self.actions = [[], ]
        self.critics = [[], ]
        self.terminals = [[], ]

    def parameters(self):
        parameters = chain(self.policy.parameters(),
                           self.critic.parameters())
        return parameters

    def _variable(self, state):
        state = th.from_numpy(state).float()
        if len(state.size()) < 2:
            state = state.unsqueeze(0)
        return V(state)

    def forward(self, state, *args, **kwargs):
        state = self._variable(state)
        action = self.policy(state, *args, **kwargs)
        return action.value.data.tolist()[0], action

    def learn(self, state, action, reward, next_state, done, info=None):
        self.rewards[-1].append(reward)
        self.actions[-1].append(info)
        self.critics[-1].append(self.critic(self._variable(state),
                                            *info.args, **info.kwargs))
        self.entropies[-1].append(info.entropy)
        self.terminals[-1].append(0)
        self.steps += 1

    def new_episode(self, terminated=False):
        self.rewards.append([])
        self.actions.append([])
        self.critics.append([])
        self.entropies.append([])
        self.terminals[-1][-1] = int(terminated)
        self.terminals.append([])

    def get_update(self):
        num_traj = loss_stats = critics_stats = entropy_stats = policy_stats = 0.0
        all_rewards, all_advantages = self.advantage(self.rewards, self.critics, self.terminals)
#        for actions_ep, rewards_ep, critics_ep, entropy_ep, terminals_ep in zip(self.actions, self.rewards, self.critics, self.entropies, self.terminals):
        for actions_ep, rewards_ep, advantage_ep, critics_ep, entropy_ep, terminals_ep in zip(self.actions, all_rewards, all_advantages, self.critics, self.entropies, self.terminals):
            if len(actions_ep) > 0:
                # Compute advantages
                #rewards_ep = V(T(rewards_ep))
                critics_ep = th.cat(critics_ep, 0).view(-1)
                #rewards_ep, advantage_ep = self.advantage(rewards_ep, critics_ep, terminals_ep)
                # Compute losses
                critic_loss = (rewards_ep - critics_ep).pow(2).mean()
                entropy_loss = th.cat(entropy_ep).mean()
                critic_loss = self.critic_weight * critic_loss
                entropy_loss = - self.entropy_weight * entropy_loss
                # Compute policy gradients
                policy_loss = 0.0
                for action, advantage in zip(actions_ep, advantage_ep):
                    policy_loss = policy_loss - action.log_prob.mean() * advantage.data[0]
                loss = policy_loss + critic_loss + entropy_loss
                loss.backward(retain_graph=True)
                if self.grad_clip > 0.0:
                    th.nn.utils.clip_grad_norm(self.parameters(), self.grad_clip)
                # Update running statistics
                loss_stats += loss.data[0]
                critics_stats += critic_loss.data[0]
                entropy_stats += entropy_loss.data[0]
                policy_stats += policy_loss.data[0]
                num_traj += 1.0

        # Store statistics
        self.stats['Num. Updates'] += 1.0
        self.stats['Num. Trajectories'] += num_traj
        self.stats['Critic Loss'] += critics_stats / num_traj
        self.stats['Entropy Loss'] += entropy_stats / num_traj
        self.stats['Policy Loss'] += policy_stats / num_traj
        self.stats['Total Loss'] += loss_stats / num_traj
        self.stats['Num. Steps'] += self.steps
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
