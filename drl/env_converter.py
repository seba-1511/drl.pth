#!/usr/bin/env python

import numpy as np
from gym.spaces import Discrete
from numpy.random import choice

from collections import Iterable
from functools import reduce


EPSILON = 1e-6
HUGE_VALUE = 1e7


def numel(x):
    if hasattr(x, 'shape'):
        return reduce(lambda x, y: x * y, x.shape)
    if hasattr(x, 'size'):
        return reduce(lambda x, y: x * y, x.size)
    if isinstance(x, Iterable):
        return reduce(lambda x, y: x * y, x)
    return x.n


def clip(val, minval, maxval):
    if val > HUGE_VALUE:
        val = HUGE_VALUE
    if val < EPSILON:
        val = EPSILON
    if val < minval:
        return minval
    if val > maxval:
        return maxval
    return val


def softmax(x):
    act = np.exp(x - np.max(x))
    return act / act.sum()


class EnvConverter(object):

    """
    Common methods for the EnvConverter sub-classes.
    """

    def __init__(self, env):
        self.env = env

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


        self.actions = []

    def __getattr__(self, name):
        if name == 'step':
            return self.step
        else:
            try:
                return self.env.__getattribute__(name)
            except:
                return self.env.__getattr__(name)

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
        return action


class StateNormalizer(EnvConverter):

    """
    Normalizes the state using a simple moving average.
    """

    def __init__(self, env, memory_size=100, per_dimension=False):
        self.env = env
        self.per_dimension = per_dimension
        self.per_dimension = True
        self.ratio = 1.0 / float(memory_size)
        self.neg_ratio = 1.0 - self.ratio
        self.mean = 0.0
        self.var = 0.0

    def normalize(self, new_state):
        x = new_state
        if not self.per_dimension:
            x = np.mean(x)
        self.mean = self.neg_ratio * self.mean + self.ratio * x
        self.var = self.neg_ratio * self.var + self.ratio * (x - self.mean)**2
        state = (new_state - self.mean) / (np.sqrt(self.var) + EPSILON)
        return state

    def step(self, action):
        next_state, reward, done, info = self.env.step(action)
        state = self.normalize(next_state)
        return state, reward, done, info

    def reset(self):
        next_state = self.env.reset()
        return self.normalize(next_state)


class SingleActionEnvConverter(EnvConverter):

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


class MultiActionEnvConverter(EnvConverter):

    """
    A `gym.Env` wrapper to enable continuous agents to work in discrete
    worlds.

    Each discrete action is assigned a dimension, and the executed action 
    is given by the max value of the input.

    Roughly: 
    action = argmax(input)
    """

    def _convert(self, action):
        if isinstance(action, int):
            return action
        return np.argmax(action)


class SoftmaxEnvConverter(EnvConverter):

    """
    A `gym.Env` wrapper to enable continuous agents to work in discrete
    worlds.

    Each discrete action is assigned a dimension, and the executed action 
    is sampled according to a softmax on the weights given by the policy.

    Roughly: 
    action = choice(num_actions, p=softmax(input))
    """

    def _convert(self, action):
        if isinstance(action, int):
            return action
        action = np.array(action).reshape(-1)
        return int(choice(self.actions, p=softmax(action)))