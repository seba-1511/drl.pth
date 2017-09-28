#!/usr/bin/env python
from __future__ import print_function

"""
    Some utility functions.
"""

import numpy as np
import gym
import torch as th

from collections import Iterable
from functools import reduce
from argparse import ArgumentParser
from torch import optim
from gym.spaces import Discrete

from .algos import Reinforce, Random
from .models import FC2, LSTM2
from .policies import ContinuousPolicy, DiscretePolicy, DiagonalGaussianPolicy, Policy
from .env_converter import EnvWrapper, StateNormalizer,  numel


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
                        default=543, help='Random generator seed')
    parser.add_argument('--update_frequency', dest='update_frequency', type=int,
                        default=1500, help='Number of steps before updating parameters.')
    parser.add_argument('--max_path_length', dest='max_path_length', type=int,
                        default=15000, help='Max length for a trajectory/episode.')
    parser.add_argument('--print_interval', dest='print_interval', type=int,
                        default=1000, help='Number of steps between each print summary.')
    parser.add_argument('--momentum', dest='momentum', type=float,
                        default=0.0, help='Default momentum value.')
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
    }
    return algos[name]


def get_model(name):
    model = {
        'fc': FC2,
        'lstm': LSTM2,
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
    args.seed += seed_offset
    env = gym.make(args.env)
    env = EnvWrapper(env)
#    env = StateNormalizer(env)
    env.seed(args.seed)
    np.random.seed(args.seed)
    th.manual_seed(args.seed)
    discrete = is_discrete(env)
    model, critic = get_model(args.model)(env.state_size,
                                          # env.action_size, layer_sizes=(8, 8),
                                          #env.action_size, layer_sizes=(64, 64),
                                          env.action_size, layer_sizes=(128, 128),
                                          dropout=args.dropout, discrete=discrete)
    recurrent = True if args.model == 'lstm' else 0
    if discrete:
        policy = DiscretePolicy(model, returns_args=recurrent)
    else:
        policy = ContinuousPolicy(model, action_size=env.action_size,
                                  returns_args=recurrent)
    policy.train()
    agent = get_algo(args.algo)(policy=policy, gamma=args.gamma,
                                critic=critic,
                                update_frequency=args.update_frequency)
    opt = None
    if agent.parameters() is not None:
        opt = get_opt(args.opt)(agent.parameters(), lr=args.lr)
    return args, env, agent, opt
