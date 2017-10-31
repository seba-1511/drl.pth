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


class EnvWrapper(object):

    """
    Wrapper around Gym environments that provides useful additional
    information.
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

    def step(self, action):
        if self.is_discrete and isinstance(action, list):
            action = action[0]
        return self.env.step(action)
    
    def _update(self):
        pass

    def __getattr__(self, name):
        if name == 'step':
            return self.step
        else:
            try:
                return self.env.__getattribute__(name)
            except:
                return self.env.__getattr__(name)



class EnvConverter(EnvWrapper):

    """
    Common methods for the EnvConverter sub-classes.
    """

    def __init__(self, env):
        super(EnvConverter, self).__init__(env)

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

    def __init__(self, env, shape, clip=10.0, update_freq=100):
        self.env = env
        self.clip = clip
        self.update_freq = update_freq
        self.count = 0
        self.sum = 0.0
        self.sum_sqr = 0.0
        self.mean = np.zeros(shape, dtype=np.double)
        self.std = np.ones(shape, dtype=np.double)

    def _update(self):
        self.mean = self.sum / self.count
        self.std = self.sum_sqr / self.count - self.mean**2
        self.std = np.clip(self.std, 1e-2, 1e9)**0.5

    def normalize(self, new_state):
        # Update
        self.count += 1
        self.sum += new_state
        self.sum_sqr += new_state**2
        if self.count % self.update_freq == 0 and False:
            self._update()
        # Normalize
        new_state = new_state - self.mean
        new_state = new_state / self.std
        new_state = np.clip(new_state, -self.clip, self.clip)
        return new_state

    def step(self, action):
        next_state, reward, done, info = self.env.step(action)
        state = self.normalize(next_state)
        return state, reward, done, info

    def reset(self):
        next_state = self.env.reset()
        return self.normalize(next_state)


class ActionNormalizer(EnvConverter):

    """
    Given an action in the range [0, 1], scales it to the range of a Gym
    environment.

    Only for continuous environments !
    """

    def __init__(self, env):
        self.env = env
        self.high = env.action_space.high
        self.low = env.action_space.low

    def step(self, action):
        action *= (self.high - self.low)
        action += self.low
        return self.env.step(action)
