#!/usr/bin/env python
from __future__ import print_function

import randopt as ro
import torch as th

from gym import wrappers
from time import time

from torch.autograd import Variable as V

from drl.utils import get_setup


def print_stats(name, rewards, n_iters, timing, steps, updates):
    denom = max(len(rewards), 1)
    print('*' * 20, name + ' Statistics Iteration ', n_iters, '*' * 20)
    print('Total Reward: ', sum(rewards))
    print('Average Reward: ', sum(rewards) / denom)
    print('Total Timing: ', timing)
    print('Total Steps: ', steps)
    print('Total Updates: ', updates)
    print('\n')


def train_update(args, env, agent, opt):
    opt.zero_grad()
    update = agent.get_update()
    opt.step()


def sample_lstm_state(args):
    hx = V(th.zeros(1, args.layer_sizes))
    cx = V(th.zeros(1, args.layer_sizes))
    return hx, cx


def train(args, env, agent, opt, update, verbose=True):
    train_rewards = []
    iter_reward = []
    train_start = time()
    train_steps = 0
    num_updates = 0
    while train_steps < args.n_steps and not agent.done():
        state = env.reset()
        episode_reward = 0.0
        hidden_state = sample_lstm_state(args)
        for path in range(args.max_path_length):
            while agent.updatable():
                update(args, env, agent, opt)
                num_updates += 1

            train_steps += 1
            if train_steps % args.print_interval == 0:
                denom = max(1, len(iter_reward))
                train_rewards.append(sum(iter_reward) / denom)
                if verbose:
                    n_iter = train_steps // args.print_interval
                    timing = time() - train_start
                    print_stats('Train', iter_reward, n_iter, timing, train_steps, num_updates)
                iter_reward = []

            action, action_info = agent.act(state, hidden_state)
            hidden_state = action_info.returns[0]
            if args.render:
                env.render()
            next_state, reward, done, _ = env.step(action)
            agent.learn(state, action, reward, next_state, done, info=action_info)
            episode_reward += reward

            if done or agent.done():
                break
            state = next_state
        agent.new_episode(done)
        iter_reward.append(episode_reward)
    return train_rewards


def test(args, env, agent):
    if args.record:
        env = wrappers.Monitor(env, './videos/' + args.env + str(time()) + '/')
    test_rewards = []
    test_start = time()
    test_steps = 0
    for iteration in range(1, 1 + args.n_test_iter):
        state = env.reset()
        iter_rewards = 0.0
        done = False
        while not done:
            test_steps += 1
            action, _ = agent.act(state)
            state, reward, done, _ = env.step(action)
            iter_rewards += reward
        test_rewards.append(iter_rewards)
    print_stats('Test', test_rewards, args.n_test_iter, time() - test_start, test_steps, 0)
    return test_rewards


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
