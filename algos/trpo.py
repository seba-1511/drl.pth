#!/usr/bin/env python

import torch as th
from torch.autograd import Variable as V
from base import BaseAgent
from algos_utils import discount, LinearVF


EPSILON = 1e-8


class TRPO(BaseAgent):

    """
    Implementation of TRPO.
    """

    def __init__(self, policy=None, optimizer=None, baseline=None, delta=0.01, gamma=0.99,
                 update_frequency=15000, gae=True, gae_lam=0.97, cg_damping=0.1, 
                 momentum=0.9, *args, **kwargs):
        # Hyper params
        self.policy = policy
        self.optimizer = optimizer
        self.delta = delta
        self.gamma = gamma
        self.update_frequency = update_frequency
        self.gae = gae
        self.gae_lam = gae_lam
        self.cg_damping = cg_damping
        self.momentum = momentum

        if baseline is None:
            baseline = LinearVF(self.policy.num_in, self.policy.num_out)
        self.baseline = baseline

        # Params
        self.action_logstd_param = V(th.rand(1, self.policy.num_out))

        # Book keeping
        self._reset()

    def _reset(self):
        self.iter_actions = [[], ]
        self.iter_actions_mean = [[], ]
        self.iter_actions_logstd = [[], ]
        self.iter_states = [[], ]
        self.iter_rewards = [[], ]
        self.iter_done = []
        self.iter_reward = 0.0
        self.step = 0
        self.num_ep = 0

    def parameters(self):
        # TODO: Include action_logstd_param
        return self.policy.parameters()

    def act(self, state):
        state = V(th.from_numpy(state).float().unsqueeze(0))
        action_mean = self.policy.forward(state)
        action_logstd = self.action_logstd_param
        action = action_mean.data + th.rand(action_mean.size()) * th.exp(self.action_logstd_param).data
        return action.numpy(), {'action_mean': action_mean.data,
                                'action_logstd': action_logstd.data}

    def learn(self, state, action, reward, next_state, done, info=None):
        self.iter_actions[-1].append(action)
        self.iter_states[-1].append(state)
        self.iter_rewards[-1].append(reward)
        self.iter_actions_mean[-1].append(info['action_mean'])
        self.iter_actions_logstd[-1].append(info['action_logstd'])
        self.iter_reward += reward
        self.step += 1

    def new_episode(self, terminated=False):
        self.iter_done.append(terminated)
        if not terminated:
            self.iter_actions.append([])
            self.iter_actions_mean.append([])
            self.iter_actions_logstd.append([])
            self.iter_states.append([])
            self.iter_rewards.append([])
            self.num_ep += 1

    def updatable(self):
        return self.step >= self.update_frequency

    def get_update(self):
        returns = []
        advantages = []

        # Compute Advantages
        for ep in range(self.num_ep + 1):
            r = discount(self.iter_rewards[ep], self.gamma)
            b = self.baseline(self.iter_states[ep])
            if self.gae and len(b) > 0:
                terminated = len(self.iter_done) != ep and self.iter_done[ep]
                b1 = th.cat([b, th.Tensor([0.0 if terminated else b[-1]])])
                deltas = th.Tensor(self.iter_rewards[ep]) + self.gamma * b1[1:] - b1[:-1]
                adv = discount(list(deltas), self.gamma * self.gae_lam)
            else:
                adv = r - b
            returns.append(r)
            advantages.append(adv)
