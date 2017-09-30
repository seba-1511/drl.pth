#!/usr/bin/env python
from __future__ import print_function

"""
    Some utility functions.
"""

import numpy as np
import gym
import pybullet_envs
import torch as th

from collections import Iterable
from functools import reduce
from argparse import ArgumentParser
from torch import optim
from gym.spaces import Discrete

from .algos import Reinforce, Random, PPO
from .models import FC2, LSTM2, Baseline
from .policies import ContinuousPolicy, DiscretePolicy, DiagonalGaussianPolicy, Policy
from .env_converter import EnvWrapper, StateNormalizer, ActionNormalizer, numel
from .algos.algos_utils import DiscountedAdvantage, GeneralizedAdvantageEstimation


def parse_args():
    # Parse arguments
    parser = ArgumentParser()
    parser.add_argument('--expname', '-e', dest='exp', metavar='e', type=str,
                        default='dev', help='Name of the experiment to be run.')
    parser.add_argument('--algo', dest='algo', type=str,
                        default='reinforce', help='Name of the learning algorithm.')
    parser.add_argument('--env', dest='env', type=str,
                        default='InvertedPendulum-v1', help='Name of the environment to learn.')
    parser.add_argument('--n_proc', dest='n_proc', type=int,
                        default=8, help='Number of processes (for async only)')
    parser.add_argument('--model', dest='model', type=str,
                        default='fc', help='What kind of model to use')
    parser.add_argument('--opt', dest='opt', type=str,
                        default='SGD', help='What kind of optimizer to use')
    parser.add_argument('--layer_sizes', dest='layer_sizes', type=int,
                        default=128, help='Size of intermediary layers.')
    parser.add_argument('--dropout', dest='dropout', type=float,
                        default=0.0, help='Dropout rate between layers')
    parser.add_argument('--lr', dest='lr', type=float,
                        default=0.01, help='The learning rate')
    parser.add_argument('--solved', dest='solved', type=float,
                        default=1000.0, help='Threshold at which the environment is considered solved.')
    parser.add_argument('--n_steps', dest='n_steps', type=int,
                        default=300, help='Number of updates to be performed.')
    parser.add_argument('--n_test_iter', dest='n_test_iter', type=int,
                        default=100, help='Number of episodes to test on.')
    parser.add_argument('--seed', dest='seed', type=int,
                        default=1234, help='Random generator seed')
    parser.add_argument('--update_frequency', dest='update_frequency', type=int,
                        default=1500, help='Number of steps before updating parameters.')
    parser.add_argument('--max_path_length', dest='max_path_length', type=int,
                        default=15000, help='Max length for a trajectory/episode.')
    parser.add_argument('--print_interval', dest='print_interval', type=int,
                        default=1000, help='Number of steps between each print summary.')
    parser.add_argument('--momentum', dest='momentum', type=float,
                        default=0.8, help='Default momentum value.')
    parser.add_argument('--gae', dest='gae', type=bool,
                        default=True, help='Whether to use GAE.')
    parser.add_argument('--gae_lam', dest='gae_lam', type=float,
                        default=0.97, help='Lambda value for GAE.')
    parser.add_argument('--delta', dest='delta', type=float,
                        default=0.01, help='Max KL divergence for TRPO')
    parser.add_argument('--cg_damping', dest='cg_damping', type=float,
                        default=0.1, help='Damping used to make CG positive def.')
    parser.add_argument('--gamma', dest='gamma', type=float,
                        default=0.99, help='Discount factor.')
    parser.add_argument('--record', dest='record', type=bool,
                        default=False, help='Whether to record videos at test time.')
    parser.add_argument('--render', dest='render', type=bool,
                        default=False, help='Whether to render display at train time.')
    parser.add_argument('--upload', dest='upload', type=bool,
                        default=False, help='Whether to upload results to the OpenAI servers.')
    return parser.parse_args()


def get_algo(name):
    algos = {
        'reinforce': Reinforce,
        'random': Random,
        'ppo': PPO,
    }
    return algos[name]


def get_model(name):
    model = {
        'fc': FC2,
        'lstm': LSTM2,
        'baseline': Baseline
    }
    return model[name]


def get_opt(name):
    opts = {
        'SGD': optim.SGD,
        'Adam': optim.Adam,
        'Adagrad': optim.Adagrad,
        'RMSprop': optim.RMSprop,
    }
    return opts[name]


def is_discrete(env):
    return isinstance(env.action_space, Discrete)


def get_setup(seed_offset=0):
    args = parse_args()
    args.print_interval = max(args.print_interval, args.update_frequency)
    args.seed += seed_offset
    env = gym.make(args.env)
    if args.render:
        env.render(mode='human')
    env = EnvWrapper(env)
    env.seed(args.seed)
    np.random.seed(args.seed)
    # Don't use the following line in async and version 0.2.0
    th.manual_seed(args.seed)
    discrete = is_discrete(env)
    model, critic = get_model(args.model)(env.state_size,
                                          env.action_size, layer_sizes=(args.layer_sizes, args.layer_sizes),
                                          dropout=args.dropout, discrete=discrete)
    recurrent = True if args.model == 'lstm' else 0
    if discrete:
        policy = DiscretePolicy(model, returns_args=recurrent)
    else:
        env = StateNormalizer(env, env.state_size, clip=5.0)
        policy = ContinuousPolicy(model, action_size=env.action_size,
                                  returns_args=recurrent)
    policy.train()
    agent = get_algo(args.algo)(policy=policy,
                                critic=critic,
                                update_frequency=args.update_frequency,
                                advantage=DiscountedAdvantage())
#                                advantage=GeneralizedAdvantageEstimation())
    opt = None
    if agent.parameters() is not None:
        if args.opt == 'SGD':
            opt = optim.SGD(agent.parameters(), lr=args.lr, momentum=args.momentum)
        elif args.opt == 'Adam':
            opt = optim.Adam(agent.parameters(), lr=args.lr, eps=1e-5)
        else:
            opt = get_opt(args.opt)(agent.parameters(), lr=args.lr)
    return args, env, agent, opt
