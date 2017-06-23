#!/usr/bin/env python

import numpy as np
from gym.spaces import Discrete
from numpy.random import choice

from collections import Iterable
from functools import reduce


def numel(x):
    if hasattr(x, 'shape'):
        return reduce(lambda x, y: x * y, x.shape)
    if hasattr(x, 'size'):
        return reduce(lambda x, y: x * y, x.size)
    if isinstance(x, Iterable):
        return reduce(lambda x, y: x * y, x)
    return x.n


def clip(val, minval, maxval):
    if val < minval:
        return minval
    if val > maxval:
        return maxval
    return val


class EnvConverter(object):

    """
    A `gym.Env` wrapper to enable continuous agents to work in discrete
    worlds.

    There is a single dimensional input, which is normalized and used to 
    determine which action to choose.

    Roughly:
    action = normalized(input) * num_actions
    """

    def __init__(self, env):
        self.env = env

        if isinstance(env.observation_space, Discrete):
            self.state_size = 1
        else:
            self.state_size = numel(env.observation_space.shape)

        if isinstance(self.env.action_space, Discrete):
            self.is_discrete = True
            self.action_size = 1
            self.max_action = env.action_space.n - 1
        else:
            self.is_discrete = False
            self.action_size = numel(env.action_space.sample())

    def __getattr__(self, name):
        if name == 'step':
            return self.step
        else:
            return self.env.__getattribute__(name)

    def step(self, action):
        if self.is_discrete:
            action = self._convert(action)
        else:
            action = self._clip(action)
        return self.env.step(action)

    def _clip(self, action):
        maxs = self.env.action_space.high
        mins = self.env.action_space.low
        if isinstance(action, np.ndarray):
            np.clip(action, mins, maxs, out=action)
        elif isinstance(action, list):
            for i in range(len(action)):
                action[i] = clip(action[i], mins[i], maxs[i])
        else:
            action = clip(action, mins[0], maxs[0])
        return action

    def _convert(self, action):
        if isinstance(action, np.ndarray):
            action *= self.max_action
            action = np.abs(action.astype(int))
            action = np.clip(action, 0, self.max_action)
            action = action[0, 0]
        elif isinstance(action, list):
            action = action[0] * self.max_action
            action = max(min(abs(int(action)), self.max_action), 0)
        else:
            action = max(min(int(action), self.max_action), 0)
        return int(action)


class FullEnvConverter(object):

    """
    A `gym.Env` wrapper to enable continuous agents to work in discrete
    worlds.

    Each discrete action is assigned a dimension, and the execute action 
    is sampled uniformly according to the weights given by the policy.

    Roughly: 
    action = choice(num_actions, p=normalize(input))
    """

    def __init__(self, env, min_prob=0.01, max_prob=0.01):
        self.env = env
        self.min_prob = min_prob
        self.max_prob = max_prob

        if isinstance(env.observation_space, Discrete):
            self.state_size = 1
        else:
            self.state_size = numel(env.observation_space.shape)

        if isinstance(self.env.action_space, Discrete):
            self.is_discrete = True
            self.action_size = env.action_space.n
            self.actions = np.arange(self.action_size)
        else:
            self.is_discrete = False
            self.action_size = numel(env.action_space.sample())

    def __getattr__(self, name):
        if name == 'step':
            return self.step
        else:
            return self.env.__getattribute__(name)

    def step(self, action):
        if self.is_discrete:
            action = self._convert(action)
        else:
            action = self._clip(action)
        return self.env.step(action)

    def _clip(self, action):
        maxs = self.env.action_space.high
        mins = self.env.action_space.low
        if isinstance(action, np.ndarray):
            np.clip(action, mins, maxs, out=action)
        elif isinstance(action, list):
            for i in range(len(action)):
                action[i] = clip(action[i], mins[i], maxs[i])
        else:
            action = clip(action, mins[0], maxs[0])
        return action

    def _convert(self, action):
        if isinstance(action, int):
            return action
        action = np.array(action).reshape(-1)
        action += np.max(np.abs(action))
        action /= sum(action)
        return choice(self.actions, p=action)
