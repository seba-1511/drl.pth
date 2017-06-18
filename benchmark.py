#!/usr/bin/env python
from __future__ import print_function

import gym
import mj_transfer
import randopt as ro
import torch as th

from seb.plot import Plot
from gym import wrappers
from time import time

from utils import get_setup

def print_stats(name, ep_rewards, n_iters, timing, steps):
    print('*'*20, name + ' Statistics Iteration ', n_iters, '*'*20)
    print('Total Reward: ', ep_rewards)
    print('Average Reward: ', ep_rewards/float(steps))
    print('Total Timing: ', timing)
    print('Total Steps: ', steps)
    print('\n')

def train_update(args, env, agent, opt):
    opt.zero_grad()
    update = agent.get_update()
    opt.step()

def train(args, env, agent, opt, update, verbose=True):
    train_iter = 1
    train_rewards = []
    episode_reward = 0.0
    train_start = time()
    train_steps = 0
    num_udpates = 0
    while train_iter < args.n_iter and not agent.done():
        state = env.reset()
        for path in range(args.max_path_length):
            train_steps += 1
            action, action_info = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            agent.learn(state, action, reward, next_state, done, info=action_info)
            episode_reward += reward
            if agent.updatable():
                update(args, env, agent, opt)
                num_udpates += 1
                if verbose:
                    print_stats('Train', episode_reward, train_iter, time() - train_start, train_steps)
                    print_stats('Overall', sum(train_rewards), train_iter, time() - train_start, train_steps)

            if done or agent.done():
                break
            state = next_state
        agent.new_episode(done)
        train_rewards.append(episode_reward)
        episode_reward = 0.0
        train_iter += 1
    if verbose:
        reward_plot = Plot('Train Reward')
        reward_plot.plot(range(len(train_rewards)), train_rewards)
        reward_plot.save('./plots/train.pdf')
        print ('num_udpates: ', num_udpates)


def test(args, env, agent):
    if args.record:
        env = wrappers.Monitor(env, './videos/' + args.env + str(time()) + '/')
    test_rewards = 0.0
    test_start = time()
    test_steps = 0
    for iteration in range(1, 1 + args.n_test_iter):
        state = env.reset()
        done = False
        while not done:
            test_steps += 1
            action, _ = agent.act(state)
            state, reward, done, _ = env.step(action)
            test_rewards += reward
    test_end = time()
    if args.record:
        pass
    print_stats('Test', test_rewards / args.n_test_iter, args.n_test_iter, time() - test_start, test_steps)



if __name__ == '__main__':
    args, env, agent, opt = get_setup()
    train(args, env, agent, opt, train_update)
    test(args, env, agent)
