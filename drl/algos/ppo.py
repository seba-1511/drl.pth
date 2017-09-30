#!/usr/bin/env python3

import torch as th
from torch import Tensor as T
from torch.autograd import Variable as V

from .reinforce import Reinforce


class PPO(Reinforce):

    def __init__(self, num_epochs=4, batch_size=64, clip=0.2, *args, **kwargs):
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.clip = clip
        super(PPO, self).__init__(*args, **kwargs)

    def _reset(self):
        super(PPO, self)._reset()
        self.epoch_optimized = 0
        self.states = [[], ]
        self.processed = False

    def learn(self, state, action, reward, next_state, done, info=None):
        assert(not self.processed, 'Can\'t add more experience while optimizing !')
        self.rewards[-1].append(reward)
        self.actions[-1].append(info)
        self.critics[-1].append(self.critic(self._variable(state), *info.args, **info.kwargs))
        self.entropies[-1].append(info.entropy)
        self.states[-1].append(state)
        self.steps += 1

    def new_episode(self, terminated=False):
        super(PPO, self).new_episode(terminated)
        self.states.append([])

    def _process(self):
        # flatten everything and properly compute the advantages
        actions = []
        states = []
        entropies = []
        rewards = []
        critics = []
        advantages = []
        # TODO: Following can be cleaned up with a function `flatten_list(list)`
        for actions_ep, states_ep, entropies_ep, rewards_ep, critics_ep in zip(self.actions, self.states, self.entropies, self.rewards, self.critics):
            if len(rewards_ep) > 0:
                actions += actions_ep
                states += [self._variable(s) for s in states_ep] 
                entropies += entropies_ep
                critics += critics_ep
                # Compute advantages
                rewards_ep = V(T(rewards_ep))
                critics_ep = th.cat(critics_ep, 0).view(-1)
                rewards_ep, advantages_ep = self.advantage(rewards_ep, critics_ep)
                rewards_ep = rewards_ep.split(1)
                rewards += rewards_ep
                advantages_ep = advantages_ep.split(1)
                advantages += advantages_ep
        self.actions = actions
        self.states = states
        self.entropies = entropies
        self.rewards = rewards
        self.critics = critics
        self.advantages = advantages

    def _sample(self):
        if not self.processed:
            self._process()
            self.processed = True
        indices = (th.rand(self.batch_size) * len(self.rewards)).int()
        log_actions = []
        rewards = []
        critics = []
        entropies = []
        states = []
        advantages = []
        for i in indices:
            log_actions.append(self.actions[i].log_prob)
            rewards.append(self.rewards[i])
            critics.append(self.critics[i])
            entropies.append(self.entropies[i])
            states.append(self.states[i])
            advantages.append(self.advantages[i])
        log_actions = th.cat(log_actions, 0)
        rewards = th.cat(rewards, 0).view(-1)
        critics = th.cat(critics, 0).view(-1)
        entropies = th.cat(entropies, 0).view(-1)
        states = th.cat(states, 0)
        advantages = th.cat(advantages, 0).view(-1)
        return log_actions, rewards, critics, entropies, states, advantages

    def get_update(self):
        num_traj = loss_stats = critics_stats = entropy_stats = policy_stats = 0.0
        for epoch in range(self.num_epochs):
            log_actions, rewards, critics, entropies, states, advantages = self._sample()
            # Compute auxiliary losses
            critic_loss = (rewards - critics).pow(2).mean()
            critic_loss = self.critic_weight * critic_loss
            entropy_loss = entropies.mean()
            entropy_loss = - self.entropy_weight * entropy_loss
            # Compute policy loss
            advantages = advantages.detach()
            new_actions = self.policy(states)
            ratios = (new_actions.log_prob - log_actions.detach()).exp()
            surr1 = ratios * advantages
            surr2 = th.clamp(ratios, 1.0 - self.clip, 1.0 + self.clip) * advantages
            clipped_loss = th.min(surr1.mean(), surr2.mean())
            policy_loss = - clipped_loss
            # Proceed to optimization
            loss = policy_loss + critic_loss + entropy_loss
            if epoch == self.num_epochs -1:
                loss.backward(retain_graph=False)
            else:
                loss.backward(retain_graph=True)
            th.nn.utils.clip_grad_norm(self.parameters(), self.grad_clip)
            # Update running statistics
            loss_stats += loss.data[0]
            critics_stats += critic_loss.data[0]
            entropy_stats += entropy_loss.data[0]
            policy_stats += policy_loss.data[0]
            num_traj += 1.0

        # Store statistics
        self.stats['Num. Updates'] += 1.0
        self.stats['Num. Trajectories'] += num_traj
        self.stats['Critic Loss'] += critics_stats / num_traj
        self.stats['Entropy Loss'] += entropy_stats / num_traj
        self.stats['Policy Loss'] += policy_stats / num_traj
        self.stats['Total Loss'] += loss_stats / num_traj
        self.stats['Num. Steps'] += self.steps
        self.epoch_optimized += 1
        self._reset()
        return [p.grad.clone() for p in self.parameters()]


    def updatable(self):
        if self.update_frequency > 0:
            if self.steps >= self.update_frequency:
                return True
        else:
            if len(self.actions) > 1:
                return True
        return False
