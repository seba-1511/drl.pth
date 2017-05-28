#!/usr/bin/env python

import torch as th
from torch.autograd import Variable as V
from base import BaseAgent

EPSILON = 1e-8


class TRPO(BaseAgent):

    """
    Implementation of TRPO.
    """

    def __init__(self, policy=None, optimizer=None, delta=0.01, gamma=0.99, update_frequency=15000, gae_lam=0.97, cg_damping=0.1, momentum=0.9, *args, **kwargs):
        # Hyper params
        self.policy = policy
        self.optimizer = optimizer
        self.delta = delta
        self.gamma = gamma
        self.update_frequency = update_frequency
        self.gae_lam = gae_lam
        self.cg_damping = cg_damping
        self.momentum = momentum

        # Params
        self.action_logstd_param = V(th.rand(1, self.policy.num_out))

    def parameters(self):
        return [self.action_logstd_param, ]

    def act(self, state):
        state = V(th.from_numpy(state).float().unsqueeze(0))
        action_mean = self.policy.forward(state)
        action = action_mean + th.exp(self.action_logstd_param) * th.rand(action_mean.size())
        return action.data.numpy()

    def learn(self, state, action, reward, next_state, done):
        pass

    def updatable(self):
        return False

    def get_update(self):
        pass
