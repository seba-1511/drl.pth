#!/usr/bin/env python3

import torch as th
from torch import Tensor as T
from torch.autograd import Variable as V

from .reinforce import Reinforce


class PPO(Reinforce):

    def __init__(self, num_epochs=10, batch_size=64, clip=0.2, *args, **kwargs):
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.clip = clip
        self.steps = 0
        super(PPO, self).__init__(*args, **kwargs)

    def _reset(self):
        self.stats['Num. Steps'] += self.steps
        super(PPO, self)._reset()
        self.epoch_optimized = 0
        self.states = [[], ]
        self.terminal = [[], ]
        self.processed = False

    def learn(self, state, action, reward, next_state, done, info=None):
        assert not self.processed, 'Can\'t add more experience while optimizing !'
        super(PPO, self).learn(state, action, reward, next_state, done, info)
        self.states[-1].append(state)
        self.terminal[-1].append(0)

    def new_episode(self, terminated=False):
        super(PPO, self).new_episode(terminated)
        self.states.append([])
        self.terminal[-1][-1] = 1
        self.terminal.append([])

    def _process(self):
        self.stats['Num. Trajectories'] += len(self.rewards)
        # flatten everything and properly compute the advantages
        actions = []
        states = []
        entropies = []
        rewards = []
        critics = []
        advantages = []
        terminal = []
        # TODO: Following can be cleaned up with a function `flatten_list(list)`
        for actions_ep, states_ep, entropies_ep, rewards_ep, critics_ep, terminal_ep in zip(self.actions, self.states, self.entropies, self.rewards, self.critics, self.terminal):
            if len(rewards_ep) > 0:
                actions += actions_ep
                states += [self._variable(s) for s in states_ep] 
                entropies += entropies_ep
                critics += critics_ep
                rewards += rewards_ep
                terminal += terminal_ep

        # Compute advantages
        rewards = V(T(rewards))
        critics = th.cat(critics, 0).view(-1)
        rewards, advantages = self.advantage(rewards, critics, terminal)
#        rewards = rewards.split(1)
#        advantages = advantages.split(1)
        # Assign processed values
        self.actions = actions
        self.states = states
        self.entropies = entropies
#        self.rewards = rewards
        self.rewards = advantages.clone()
        self.critics = critics
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)
        self.advantages = advantages.split(1)

    def _sample(self):
        if not self.processed:
            self._process()
            self.processed = True
        indices = (th.rand(self.batch_size) * len(self.rewards)).int()
        # TODO: Cleanup
        log_actions = []
        rewards = []
        critics = []
        entropies = []
        states = []
        advantages = []
        actions = []
        for i in indices:
            actions.append(self.actions[i].value)
            log_actions.append(self.actions[i].log_prob)
            rewards.append(self.rewards[i])
            critics.append(self.critics[i])
            entropies.append(self.entropies[i])
            states.append(self.states[i])
            advantages.append(self.advantages[i])
        actions = th.cat(actions, 0)
        log_actions = th.cat(log_actions, 0)
        rewards = th.cat(rewards, 0).view(-1)
        critics = th.cat(critics, 0).view(-1)
        entropies = th.cat(entropies, 0).view(-1)
        states = th.cat(states, 0)
        advantages = th.cat(advantages, 0).view(-1)
        return actions, log_actions, rewards, critics, entropies, states, advantages

    def get_update(self):
        actions, log_actions, rewards, critics, entropies, states, advantages = self._sample()
        # Compute auxiliary losses
        critics = self.critic(states)
        critic_loss = (rewards - critics).pow(2).mean()
        critic_loss = self.critic_weight * critic_loss
        entropy_loss = entropies.mean()
        entropy_loss = - self.entropy_weight * entropy_loss
        # Compute policy loss
        advantages = advantages.detach().view(-1, 1)
        new_actions = self.policy(states)
        log_prob = -0.5 * ((actions - new_actions.raw) / self.policy.logstd.exp()).pow(2) - self.policy.halflog2pi - self.policy.logstd
        ratios = (log_prob.sum(1) - log_actions.detach().sum(1)).exp()
#        ratios = (new_actions.compute_log_prob(actions).sum(1) - log_actions.detach().sum(1)).exp()
        surr = ratios.view(-1, 1) * advantages
        clipped = th.clamp(ratios, 1.0 - self.clip, 1.0 + self.clip).view(-1, 1) * advantages
        policy_loss = - th.min(surr, clipped).mean()
        # Proceed to optimization
        loss = policy_loss + critic_loss + entropy_loss
        if self.epoch_optimized == self.num_epochs:
            loss.backward(retain_graph=False)
        else:
            loss.backward(retain_graph=True)
        if self.grad_clip > 0.0:
            th.nn.utils.clip_grad_norm(self.parameters(), self.grad_clip)

        # Store statistics
        self.stats['Num. Updates'] += 1.0
        self.stats['Critic Loss'] += critic_loss.data[0]
        self.stats['Entropy Loss'] += entropy_loss.data[0]
        self.stats['Policy Loss'] += policy_loss.data[0]
        self.stats['Total Loss'] += loss.data[0]
        return [p.grad.clone() for p in self.parameters()]

    def updatable(self):
        # self.update_frequency = 0 -> optimize after each full trajectory
        if self.update_frequency > 0:
            if self.steps >= self.update_frequency:
                if self.epoch_optimized >= self.num_epochs:
                    self._reset()
                    return False
                self.epoch_optimized += 1
                return True
        else:
            if len(self.actions) > 1:
                if self.epoch_optimized >= self.num_epochs:
                    self._reset()
                    return False
                self.epoch_optimized += 1
                return True
        return False
