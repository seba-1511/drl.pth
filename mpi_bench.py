#!/usr/bin/env python

import gym
import randopt as ro
import mj_envs

from benchmark import test
from utils import parse_args, get_algo

rank = 0

def mpi_average(params):
    pass


def mpi_train(args, env, agent):
    train_iter = 0
    mpi_average(agent.parameters)
    while train_iter < args.n_iter and not agent.done():
        state = env.reset()
        for path in range(args.max_path_length):
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            agent.learn(state, action, reward, next_state, done)
            if agent.updatable():
                update = agent.get_update()
                mpi_average(update)
                agent.update(update)

            if done or agent.done():
                break
            state = next_state
        agent.new_episode(done)
        train_iter += 1


if __name__ == '__main__':
    args = parse_args()

    env = gym.make(args.env)
    agent = get_algo(args.algo)()

    mpi_train(args, env, agent)

    if rank == 0:
        test(args, env, agent)
