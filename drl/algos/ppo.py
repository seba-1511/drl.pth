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
        self.stats['Num. Trajectories'] += len(self.rewards)
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
#            log_actions.append(self.actions[i].compute_log_prob(self.actions[i].value).sum(1, keepdim=True))
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
        critic_loss = (rewards - critics).pow(2).mean()
        critic_loss = self.critic_weight * critic_loss
        entropy_loss = entropies.mean()
        entropy_loss = - self.entropy_weight * entropy_loss
        # Compute policy loss
        advantages = advantages.detach().view(-1, 1)
        new_actions = self.policy(states)
        ratios = (new_actions.compute_log_prob(actions) - log_actions.detach()).exp()
        surr1 = advantages.mul(ratios)
        surr2 = advantages.mul(th.clamp(ratios, 1.0 - self.clip, 1.0 + self.clip))
        clipped_loss = th.min(surr1, surr2).mean()
        policy_loss = - clipped_loss
#        import pdb; pdb.set_trace()
        # Proceed to optimization
        loss = policy_loss + critic_loss + entropy_loss
        if self.epoch_optimized == self.num_epochs:
            loss.backward(retain_graph=False)
        else:
            loss.backward(retain_graph=True)
        th.nn.utils.clip_grad_norm(self.parameters(), self.grad_clip)

        # TODO: Fix the stats !
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

#    def updatable(self):
#        if self.update_frequency > 0:
#            if self.steps >= self.update_frequency:
#                return True
#        else:
#            if len(self.actions) > 1:
#                return True
#        return False
