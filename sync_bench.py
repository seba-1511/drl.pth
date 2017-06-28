#!/usr/bin/env python
from __future__ import print_function

import numpy as np
import torch as th
import torch.distributed as dist
import torch.distributed as dist
from torch.multiprocessing import Process
from torch._utils import _flatten_tensors, _unflatten_tensors

from benchmark import train, test
from utils import get_setup, parse_args


def sync(tensors):
    size = float(dist.get_world_size())
    flat_tensors = _flatten_tensors(tensors)
    dist.all_reduce(flat_tensors, dist.reduce_op.SUM)
    for p, u in zip(tensors, _unflatten_tensors(flat_tensors, tensors)):
        p.set_(u / size)


def sync_update(args, env, agent, opt):
    opt.zero_grad()
    update = agent.get_update()
    sync(update)
    agent.set_gradients(update)
    opt.step()


def run(rank, size):
    is_root = (rank == 0)
    args, env, agent, opt = get_setup(seed_offset=rank)
    sync([p for p in agent.parameters()])
    train(args, env, agent, opt, sync_update, verbose=is_root)
    if is_root:
        test(args, env, agent)


def init_processes(rank, size, fn, backend='tcp'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
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
