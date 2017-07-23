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


class PPO(ActorCriticReinforce):

    """
    TODO:
        * Implement KL penalty with coefficient
    """

    def __init__(self, policy=None, critic=None, num_epochs=10, batch_size=64,
                 loss_clip=0.2, gamma=0.99, update_frequency=2048, entropy_weight=0.0001, critic_weight=0.5):
        super(PPO, self).__init__(policy=policy, gamma=gamma,
                                  update_frequency=update_frequency,
                                  entropy_weight=entropy_weight, 
                                  critic_weight=critic_weight)
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.loss_clip = loss_clip
        self.epoch_optimized = 0

    def _reset(self):
        super(PPO, self)._reset()
        self.states = [[], ]
        self.epoch_optimized = 0

    def act(self, state):
        state = V(th.from_numpy(state).float().unsqueeze(0))
        mean, logstd = self.policy.forward(state)
        std = th.exp(logstd + V(th.zeros(logstd.size()).uniform_(-0.5, 0.5)))
        entropy = 0.5 * (V(self.log2pie.expand_as(std)) + 2.0 * logstd)
        action = mean + std
        critic = self.critic.forward(self.policy.model.critic_state.view(1, -1))
        return action.data.numpy(), (action, entropy, critic)

    def learn(self, state, action, reward, next_state, done, info=None):
        action, entropy, critic = info
        super(PPO, self).learn(state, action, reward, next_state, done, info)
        self.critics[-1].append(critic)
        self.states[-1].append(state)

    def new_episode(self, terminated=False):
        super(PPO, self).new_episode(terminated=terminated)
        self.states.append([])
        if not terminated:
            self.rewards[-2].append(self.critics[-2][-1].data[0, 0])

    def get_update(self):
        batch_start = (self.epoch_optimized - 1) * self.batch_size
        batch_end = batch_start + self.batch_size
        iterator = zip(self.states[batch_start:batch_end],
                       self.actions[batch_start:batch_end],
                       self.rewards[batch_start:batch_end],
                       self.critics[batch_start:batch_end])
        tot_s = 0.0
        tot_e = 0.0
        tot_c = 0.0
        num = 0
        for states, actions, rewards, critics in iterator:
            if len(actions) > 0:
                # rewards = normalize(discount(rewards, self.gamma))
                rewards = generalized_advantage_estimations(rewards, critics, self.gamma, 0.95)
                rewards = normalize(rewards)
                loss = 0.0
                for state, action, reward, critic in zip(states, actions, rewards, critics):
                    _, (action_new, entropy, _) = self.act(state)
                    advantage = (reward - critic).expand_as(action)
                    surrogate = th.exp(action_new.log() - action.log().detach())
                    clamped = th.clamp(surrogate, 1 - self.loss_clip, 1 + self.loss_clip)
                    clip_loss = th.min(surrogate * advantage, clamped * advantage).sum()
                    critic_loss = self.critic_weight * ((critic - reward)**2).sum()
                    entropy_loss = self.entropy_weight * entropy.sum()
                    tot_s += clip_loss
                    tot_e += entropy_loss
                    tot_c += critic_loss
                    num += 1
                    loss += clip_loss - entropy_loss + critic_loss
                loss /= len(actions)
                loss.backward()
        # NOTE: Don't reset here, since we need to compute several updates
        print('surrogate loss: ', tot_s.data[0] / num)
        print('entropy loss: ', tot_e.data[0] / num)
        print('critic_loss:', tot_c.data[0] / num)
        print(' ')
        return [p.grad.clone() for p in self.parameters()]

    def updatable(self):
        # self.update_frequency = 0 -> optimize after each trajectory
        if self.update_frequency > 0:
            if self.steps >= self.update_frequency:
                if self.epoch_optimized >= self.num_epochs:
                    self._reset()
                    return False
                self.epoch_optimized += 1
                return True
        else:
            if len(self.actions) > 1:
                if self.epoch_optimized >= self.num_epochs:
                    self._reset()
                    return False
                self.epoch_optimized += 1
                return True
        return False
