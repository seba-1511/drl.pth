#!/usr/bin/env python
from __future__ import print_function

import randopt as ro
import torch as th
import torch.multiprocessing as mp
from torch.multiprocessing import Queue

from time import sleep, time

from benchmark import train, train_update
from drl.utils import get_setup
from drl.training import train, test


def async_update(agent, opt, rank, outputs):
    th.set_num_threads(1)
    # Proceed with training but keeping the current agent
    args, env, _, _ = get_setup(seed_offset=rank)
    is_root = (rank == 0)
    train_rewards = train(args, env, agent, opt, train_update, verbose=is_root)
    if is_root:
        for r in train_rewards:
            outputs.put(r)


if __name__ == '__main__':
    args, env, agent, opt = get_setup()
    num_processes = args.n_proc
    processes = []

    # Share parameters of the policy (and opt)
    agent.share_memory()

    exp = ro.Experiment(args.env + '-dev-async', params={})
    train_rewards = Queue()
    for rank in range(num_processes):
        sleep(1.0)
        p = mp.Process(target=async_update, args=(agent, opt, rank, train_rewards))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    test_rewards = test(args, env, agent)
    data = {p: getattr(args, p) for p in vars(args)}
    data['train_rewards'] = [train_rewards.get() for _ in range(train_rewards.qsize())]
    data['test_rewards'] = test_rewards
    data['timestamp'] = time()
    exp.add_result(result=sum(test_rewards) / len(test_rewards),
                   data=data)
