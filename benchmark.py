#!/usr/bin/env python

from time import time

def train(args, env, agent):
    pass

def test(args, env, agent):
    if args.record:
        env.monitor.start('./videos/' + args.env + str(time()) + '/')
    test_rewards = 0
    test_start = time()
    for iteration in range(args.n_test_iter):
        state = env.reset()
        for _ in range(2 * args.max_path_length):
            action = agent.act(state)
            state, reward, done, _ = env.step(action)
            test_rewards += reward
            if done:
                break
    test_end = time()
    if args.record:
        env.monitor.close()
