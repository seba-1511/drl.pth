#!/usr/bin/env python
from __future__ import print_function

import randopt as ro
import torch as th

from time import time

from drl.utils import get_setup
from drl.training import test, train


def seq_update(args, env, agent, opt):
    opt.zero_grad()
    update = agent.get_update()
    opt.step()


def train_update(args, env, agent, opt):
    opt.zero_grad()
    update = agent.get_update()
    opt.step()


if __name__ == '__main__':
    args, env, agent, opt = get_setup()
    exp = ro.Experiment(args.env + '-dev-seq', params={})
    train_rewards = train(args, env, agent, opt, train_update)
    test_rewards = test(args, env, agent)
    data = {p: getattr(args, p) for p in vars(args)}
    data['train_rewards'] = train_rewards
    data['test_rewards'] = test_rewards
    data['timestamp'] = time()
    exp.add_result(result=sum(test_rewards) / len(test_rewards),
                   data=data)
