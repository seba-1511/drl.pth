#!/usr/bin/env python

import torch as th
from torch.autograd import Variable as V
from base import BaseAgent
from algos_utils import discount, normalize, LinearVF, EPSILON, PI, gauss_log_prob


class TRPO(BaseAgent):

    """
    Implementation of TRPO.
    """

    def __init__(self, policy=None, optimizer=None, baseline=None, delta=0.01, gamma=0.99,
                 update_frequency=15000, gae=True, gae_lam=0.97, cg_damping=0.1, 
                 momentum=0.9, *args, **kwargs):
        # Hyper params
        self.policy = policy
        self.optimizer = optimizer
        self.delta = delta
        self.gamma = gamma
        self.update_frequency = update_frequency
        self.gae = gae
        self.gae_lam = gae_lam
        self.cg_damping = cg_damping
        self.momentum = momentum

        if baseline is None:
            baseline = LinearVF(self.policy.num_in, self.policy.num_out)
        self.baseline = baseline

        # Params
        self.action_logstd_param = V(th.rand(1, self.policy.num_out), requires_grad=True)

        # Book keeping
        self._reset()

    def _reset(self):
        self.iter_actions = [[], ]
        self.iter_actions_mean = [[], ]
        self.iter_actions_logstd = [[], ]
        self.iter_states = [[], ]
        self.iter_rewards = [[], ]
        self.iter_done = []
        self.iter_reward = 0.0
        self.step = 0
        self.num_ep = 0

    def parameters(self):
        # TODO: Include action_logstd_param
        for p in self.policy.parameters():
            yield p
        yield self.action_logstd_param
        # return self.policy.parameters()

    def act(self, state):
        state = V(th.from_numpy(state).float().unsqueeze(0))
        action_mean = self.policy.forward(state)
        action_logstd = self.action_logstd_param
        action = action_mean.data + th.rand(action_mean.size()) * th.exp(self.action_logstd_param).data
        return action.tolist(), {'action_mean': action_mean.data.tolist(),
                                 'action_logstd': action_logstd.data.tolist()}

    def learn(self, state, action, reward, next_state, done, info=None):
        self.iter_actions[-1].append(action)
        self.iter_states[-1].append(state.tolist())
        self.iter_rewards[-1].append(reward)
        self.iter_actions_mean[-1].append(info['action_mean'])
        self.iter_actions_logstd[-1].append(info['action_logstd'])
        self.iter_reward += reward
        self.step += 1

    def new_episode(self, terminated=False):
        self.iter_done.append(terminated)
        if not terminated:
            self.iter_actions.append([])
            self.iter_actions_mean.append([])
            self.iter_actions_logstd.append([])
            self.iter_states.append([])
            self.iter_rewards.append([])
            self.num_ep += 1

    def updatable(self):
        return self.step >= self.update_frequency

    def get_update(self):
        returns = []
        advantages = []

        # Compute Advantages
        for ep in range(self.num_ep + 1):
            r = discount(self.iter_rewards[ep], self.gamma)
            b = self.baseline(self.iter_states[ep])
            if self.gae and len(b) > 0:
                terminated = len(self.iter_done) != ep and self.iter_done[ep]
                b1 = th.cat([b, th.Tensor([0.0 if terminated else b[-1]])])
                deltas = th.Tensor(self.iter_rewards[ep]) + self.gamma * b1[1:] - b1[:-1]
                adv = discount(list(deltas), self.gamma * self.gae_lam)
            else:
                adv = r - b
            returns.append(r)
            advantages.append(adv)

        # Fit baseline for next iter
        self.baseline.learn(self.iter_states, returns)

        # Create variables
        states = V(th.Tensor(self.iter_states)[0])
        actions = V(th.Tensor(self.iter_actions)[0])
        means = V(th.Tensor(self.iter_actions_mean))
        means = means.view(states.size(0), -1)
        logstds = V(th.Tensor(self.iter_actions_logstd).view(states.size(0), -1))
        advantages = V(normalize(th.cat(advantages)))

        # Start Computing the actual update
        inputs = [actions, states, means, logstds, advantages]
        surr_loss, surr_gradients = self._surrogate(*inputs) 

        # CG, to be taken out
        def fisher_vec_prod(vectors):
            a_logstds = logstds.repeat(means.size(0), 1)
            args = [actions, states, means, a_logstds]
            res = self._grads_gvp(args + vectors)
            return [r + (p * self.cg_damping) for r, p in zip(res, vectors)]

        # END CG

        # At last, reset iteration statistics
        self._reset()
        for p, g in zip(self.parameters(), surr_gradients):
            p.grad.data[:] = -g
        return [g for g in surr_gradients]

    def _grads_gvp(self, actions, states, means, logstds, vectors):
        pass

    def _surrogate(self, actions, states, means, logstds, advantages):
        # Computes the gauss_log_prob on sampled data
        old_log_p_n = gauss_log_prob(means, logstds, actions)

        # Computes the gauss_log_prob wrt current params
        new_a_means = self.policy.forward(states)
        new_a_logstds = self.action_logstd_param.repeat(new_a_means.size(0), 1)
        new_log_p_n = gauss_log_prob(new_a_means, new_a_logstds, actions)

        # Compute the actual surrogate
        ratio = th.exp(new_log_p_n - old_log_p_n)
        advantages = advantages.view(-1, 1)

        surr = th.mean(ratio * advantages)
        surr.backward()
        gradients = [p.grad.data.clone() for p in self.parameters()]
        return surr, gradients
