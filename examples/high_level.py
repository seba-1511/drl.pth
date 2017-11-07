#!/usr/bin/env python3

import gym
import hierarchical_envs
from hierarchical_envs.orientation_env import StochasticDiscreteOrientation
import randopt as ro
import torch as th

import torch.optim as optim

from time import time

from drl.training import train, test
from drl.utils import parse_min_args
from drl.models import Baseline
from drl.env_converter import EnvWrapper
from drl.policies import DiscretePolicy
from drl.algos import Reinforce
from drl.algos.algos_utils import DiscountedAdvantage


if __name__ == '__main__':
#    args, env, agent, opt = get_setup()
    args = parse_min_args()

#    env = gym.make('DiscreteOrientation-v0', size_noise=0.0)
    env = StochasticDiscreteOrientation(size_noise=0.0)
    env = EnvWrapper(env)
    env.seed(1234)
    model, critic =  Baseline(env.state_size, env.action_size, layer_sizes=(2, 2), discrete=True)
    policy = DiscretePolicy(model)
    agent = Reinforce(policy=policy, critic=critic, update_frequency=args.update_frequency,
                      critic_weight=1.0,
                      entropy_weight=0.0001,
#                      grad_clip=0.5,
                      advantage=DiscountedAdvantage())
#                      advantage=GeneralizedAdvantageEstimation(tau=0.95, gamma=0.99))
    opt = optim.Adam(agent.parameters(), lr=7e-4, eps=1e-5)


    exp = ro.Experiment('DiscreteOrientation-dev-seq', params={})
    train_rewards = train(args, env, agent, opt)
    test_rewards = test(args, env, agent)
    data = {p: getattr(args, p) for p in vars(args)}
    data['train_rewards'] = train_rewards
    data['test_rewards'] = test_rewards
    data['timestamp'] = time()
    exp.add_result(result=sum(test_rewards) / len(test_rewards),
                   data=data)
    th.save(agent.state_dict(), './high_level.pth')
