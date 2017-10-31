#!/usr/bin/env python
from __future__ import print_function

import os
import randopt as ro
import torch as th
import torch.distributed as dist
from torch.multiprocessing import Process
from time import time

from drl.utils import get_setup, parse_args
from drl.training import train, test

def sync(tensors):
    size = float(dist.get_world_size())
    for t in tensors:
        dist.all_reduce(t.data)
        t.data /= size


def sync_update(args, env, agent, opt):
    opt.zero_grad()
    update = agent.get_update()
    sync(update)
    agent.set_gradients(update)
    opt.step()


def run(rank, size):
    is_root = (rank == 0)
    args, env, agent, opt = get_setup(seed_offset=rank)
    exp = ro.Experiment(args.env + '-dev-sync', params={})
    sync(list(agent.parameters()))
    train_rewards = train(args, env, agent, opt, sync_update, verbose=is_root)
    if is_root:
        test_rewards = test(args, env, agent)
        data = {p: getattr(args, p) for p in vars(args)}
        data['train_rewards'] = train_rewards
        data['test_rewards'] = test_rewards
        data['timestamp'] = time()
        exp.add_result(result=sum(test_rewards) / len(test_rewards),
                       data=data)


def init_processes(rank, size, fn, backend='tcp'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    th.set_num_threads(1)
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)


if __name__ == "__main__":
    args = parse_args()
    size = args.n_proc
    processes = []
    for rank in range(size):
        p = Process(target=init_processes, args=(rank, size, run))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
