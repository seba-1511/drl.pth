#!/usr/bin/env python

import numpy as np
import torch as th
from torch import Tensor as T
from torch.autograd import Variable as V
from math import pi, exp
from random import random

from .reinforce_with_critic import ActorCriticReinforce
from .algos_utils import discount, normalize, generalized_advantage_estimations, EPSILON, logp


class PPO(ActorCriticReinforce):

    """
    TODO:
        * Implement KL penalty with coefficient
    """

    def __init__(self, policy=None, critic=None, num_epochs=10, loss_clip=0.2,
                 gamma=0.99, update_frequency=2048, entropy_weight=0.0001, critic_weight=0.5):
        super(PPO, self).__init__(policy=policy, gamma=gamma,
                                  update_frequency=update_frequency,
                                  entropy_weight=entropy_weight,
                                  critic_weight=critic_weight)
        self.batch_size = update_frequency // num_epochs
        self.num_epochs = num_epochs
        self.loss_clip = loss_clip
        self.epoch_optimized = 0

    def _reset(self):
        super(PPO, self)._reset()
        self.states = [[], ]
        self.means = [[], ]
        self.logstds = [[], ]
        self.advantages = [[], ]
        self.epoch_optimized = 0

    def act(self, state):
        state = V(th.from_numpy(state).float().unsqueeze(0))
        mean, logstd = self.policy.forward(state)
        std = th.exp(logstd)
        entropy = 0.5 * V(self.log2pie.expand_as(std)) + logstd
        action = mean + std * V(th.randn(mean.size()))
        critic = self.critic.forward(self.policy.model.critic_state.view(1, -1))
        return action.data.numpy()[0], (action, entropy, critic, mean, logstd)

    def learn(self, state, action, reward, next_state, done, info=None):
        action, entropy, critic, mean, logstd = info
        super(PPO, self).learn(state, action, reward, next_state, done,
                               (action, entropy, critic))
        self.states[-1].append(state)
        self.means[-1].append(mean)
        self.logstds[-1].append(logstd)

    def new_episode(self, terminated=False):
        super(PPO, self).new_episode(terminated=terminated)
        self.states.append([])
        self.means.append([])
        self.logstds.append([])
        if not terminated:
            self.rewards[-2].append(self.critics[-2][-1].data[0, 0])

    def _prepare_for_update(self):
        # Compute everything that is fixed (gae, rewards)
        self.advantages = [generalized_advantage_estimations(ep_rewards, ep_critics, self.gamma, 0.95) for ep_rewards, ep_critics in zip(self.rewards, self.critics)]
        # self.rewards = [normalize(discount(ep_rewards, self.gamma)) for ep_rewards in self.rewards if len(ep_rewards) != 0]
        self.rewards = [discount(ep_rewards, self.gamma) for ep_rewards in self.rewards if len(ep_rewards) != 0]
        self.rewards = [a + r for ep_rewards, ep_advantages in zip(self.critics, self.advantages) for a, r in zip(ep_rewards, ep_advantages)]
        # Flatten all episodes to make them easy to sample
        self.states = [s for ep_states in self.states for s in ep_states]
        self.actions = [V(a.data) for ep_actions in self.actions for a in ep_actions]
        self.rewards = [r for ep_rewards in self.rewards for r in ep_rewards]
        # We use .data since we don't want variables. 
        self.advantages = [a.data for ep_advantages in self.advantages for a in ep_advantages]
        self.advantages = normalize(th.cat(self.advantages))
        self.means = [m.detach() for ep_means in self.means for m in ep_means]
        # Need .clone() otherwise the data from policy.logstd is used.
        self.logstds = [V(l.data.clone()) for ep_logstd in self.logstds for l in ep_logstd]
        self.critics = [c.detach() for ep_critics in self.critics for c in ep_critics]

    def get_update(self):
        tot_s = 0.0
        tot_e = 0.0
        tot_c = 0.0
        tot_kl = 0.0
        ratios = 0.0
        tot_cl = 0.0

        if self.epoch_optimized == 1:
            self._prepare_for_update()

        idx = int(random() * (self.update_frequency // self.batch_size))
        batch_start = idx * self.batch_size
        batch_end = batch_start + self.batch_size

        iterator = zip(self.states[batch_start:batch_end],
                       self.actions[batch_start:batch_end],
                       self.rewards[batch_start:batch_end],
                       self.advantages[batch_start:batch_end],
                       self.means[batch_start:batch_end],
                       self.logstds[batch_start:batch_end],
                       self.critics[batch_start:batch_end],
        )

        # states = self.states[batch_start:batch_end]
        # old_actions = self.actions[batch_start:batch_end]
        # rewards = self.rewards[batch_start:batch_end]
        # advantages = self.advantages[batch_start:batch_end]
        # old_means = self.means[batch_start:batch_end]
        # old_logstds = self.logstds[batch_start:batch_end]
        # old_critics = self.critics[batch_start:batch_end]

        loss = 0.0
        for state, old_action, reward, advantage, old_mean, old_logstd, old_critic in iterator:
            _, (new_action, entropy, new_critic, new_mean, new_logstd) = self.act(state)
            old_std, new_std = th.exp(old_logstd), th.exp(new_logstd)
            # kl = new_logstd - old_logstd + (old_std**2 + (old_mean - new_mean)**2) / (2.0 * new_std**2) - 0.5
            log_new_action = logp(new_action, new_mean, new_std)
            log_old_action = logp(old_action, old_mean, old_std)
            surrogate = th.exp(log_new_action - log_old_action).mean()
            # ratios += surrogate
            clamped = th.clamp(surrogate, 1 - self.loss_clip, 1 + self.loss_clip)
            clip_loss = (surrogate * advantage).mean()
            # clip_loss = th.min(surrogate * advantage, clamped * advantage).mean()
            # Apply same trust-region on critic loss
            critic_loss = ((reward - new_critic)**2).mean()
            critic_clamped = V(T([0.0]))
            # critic_clamped += th.clamp(old_critic - new_critic, -self.loss_clip, self.loss_clip) + old_critic 
            critic_clamped = critic_clamped**2
            critic_loss = th.max(critic_clamped, critic_loss).mean()
            critic_loss *= self.critic_weight

            entropy_loss = self.entropy_weight * entropy.mean()
            # tot_s += clip_loss
            # tot_e += entropy_loss
            # tot_c += critic_loss
            # tot_kl += kl
            # tot_cl += clamped
            loss += clip_loss + critic_loss - entropy_loss
        loss /= self.batch_size
        # Retain graph until batches can be dissmissed
        if self.epoch_optimized >= self.num_epochs:
            loss.backward()
        else:
            loss.backward(retain_variables=True)

        # NOTE: Don't reset here, since we need to compute several updates
        # if self.epoch_optimized == 1:
            # print('ratios: ', ratios.data[0] / self.batch_size)
            # print('clamped loss: ', tot_cl.data[0] / self.batch_size)
            # print('surrogate loss: ', tot_s.data[0] / self.batch_size)
            # print('entropy loss: ', tot_e.data[0] / self.batch_size)
            # print('critic_loss:', tot_c.data[0] / self.batch_size)
            # print('kl:', tot_kl.data[0] / self.batch_size)
            # print(' ')
        # return None
        return [p.grad.clone() for p in self.parameters()]

    def updatable(self):
        # self.update_frequency = 0 -> optimize after each full trajectory
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
