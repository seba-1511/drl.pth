#!/usr/bin/env python

import gym
import randopt as ro
import mj_envs

from utils import parse_args, get_algo

def mpi_average(params):
    pass


def main(args):
    env = gym.make(args.env)
    
    agent = get_algo(args.algo)()

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
    main(args)
