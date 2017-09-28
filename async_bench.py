#!/usr/bin/env python
from __future__ import print_function

import numpy as np
import torch as th
import torch.multiprocessing as mp

from time import sleep

from benchmark import train, test, train_update
from drl.utils import get_setup


def async_train(agent, opt, rank):
    # Proceed with training but keeping the current agent
    args, env, _, _ = get_setup(seed_offset=rank)
    verbose = (rank == 0) 
    train(args, env, agent, opt, train_update, verbose)


if __name__ == '__main__':
    args, env, agent, opt = get_setup()
    num_processes = args.n_proc
    processes = []

    # Share parameters of the policy (and opt)
    agent.share_memory()

    for rank in range(num_processes):
        sleep(1.0)
        p = mp.Process(target=async_train, args=(agent, opt, rank))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    test(args, env, agent)
