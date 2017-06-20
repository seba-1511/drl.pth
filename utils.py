#!/usr/bin/env python
from __future__ import print_function

"""
    Some utility functions.
"""

import gym
import mj_transfer
import torch as th

from collections import Iterable
from functools import reduce
from argparse import ArgumentParser
from torch import optim

from algos import A3C, Reinforce, TRPO, Random
from models import FC, Atari, LSTM, StochasticContinuousPolicy
from env_converter import EnvConverter, numel


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
    parser.add_argument('--policy', dest='policy', type=str,
                        default='fc', help='What kind of policy to use')
    parser.add_argument('--opt', dest='opt', type=str,
                        default='SGD', help='What kind of optimizer to use')
    parser.add_argument('--lr', dest='lr', type=float,
                        default=0.01, help='The learning rate')
    parser.add_argument('--filter', dest='filter', type=bool,
                        default=True, help='Whether to filter the environment\'s states.')
    parser.add_argument('--solved', dest='solved', type=float,
                        default=1000.0, help='Threshold at which the environment is considered solved.')
    parser.add_argument('--n_iter', dest='n_iter', type=int,
                        default=300, help='Number of updates to be performed.')
    parser.add_argument('--n_test_iter', dest='n_test_iter', type=int,
                        default=100, help='Number of episodes to test on.')
    parser.add_argument('--seed', dest='seed', type=int,
                        default=1234, help='Random generator seed')
    parser.add_argument('--update_frequency', dest='update_frequency', type=int,
                        default=1500, help='Number of steps before updating parameters.')
    parser.add_argument('--max_path_length', dest='max_path_length', type=int,
                        default=15000, help='Max length for a trajectory/episode.')
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
    parser.add_argument('--upload', dest='upload', type=bool,
                        default=False, help='Whether to upload results to the OpenAI servers.')
    return parser.parse_args()


def get_algo(name):
    algos = {
        'reinforce': Reinforce,
        'trpo': TRPO,
        'a3c': A3C,
        'random': Random,
    }
    return algos[name]


def get_policy(name):
    policies = {
        'fc': FC,
        'lstm': LSTM,
        'atari': Atari,
    }
    return policies[name]


def get_opt(name):
    policies = {
        'SGD': optim.SGD,
        'Adam': optim.Adam,
        'Adagrad': optim.Adagrad,
        'RMSprop': optim.RMSprop,
    }
    return policies[name]


def get_setup(seed_offset=0):
    args = parse_args()
    args.seed += seed_offset
    env = gym.make(args.env)
    env = EnvConverter(env)
    env.seed(args.seed)
    th.manual_seed(args.seed)
    model = get_policy(args.policy)(env.state_size,
                                    env.action_size, layers=(64, 64))
    policy = StochasticContinuousPolicy(model)
    agent = get_algo(args.algo)(policy=policy, gamma=args.gamma,
                                update_frequency=args.update_frequency)
    opt = None
    if agent.parameters() is not None:
        opt = get_opt(args.opt)(agent.parameters(), lr=args.lr, momentum=-0.3)
    return args, env, agent, opt
