#!/usr/bin/env python
from __future__ import print_function

import numpy as np
import torch as th

from mpi4py import MPI

from benchmark import train, test
from utils import get_setup

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
ROOT = (rank == 0)

def mpi_sync(params):
    for p in params:
        p_np = p.data.cpu().numpy()
        buff = np.zeros_like(p_np)
        comm.Allreduce(p_np, buff, MPI.SUM)
        buff /= float(size)
        buff = th.from_numpy(buff)
        p.data.set_(buff)

def mpi_update(args, env, agent, opt):
    opt.zero_grad()
    update = agent.get_update()
    mpi_sync(update)
    agent.set_gradients(update)
    opt.step()

if __name__ == '__main__':
    args, env, agent, opt = get_setup(seed_offset=rank)
    mpi_sync([p for p in agent.parameters()])
    train(args, env, agent, opt, mpi_update, verbose=ROOT)
    if rank == 0:
        test(args, env, agent)
