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
        self.epoch_optimized = 0

        self.steps = 0
        self.rewards = [[], ]
        self.entropies = [[], ]
        self.actions = [[], ]
        self.critics = [[], ]
        self.states = [[], ]

    def learn(self, state, action, reward, next_state, done, info=None):
        self.rewards[-1].append(reward)
        self.actions[-1].append(info)
        self.states[-1].append(state)
        self.critics[-1].append(self.critic(self._variable(state), *info.args, **info.kwargs))
        self.entropies[-1].append(info.entropy)
        self.steps += 1

    def new_episode(self, terminated=False):
        self.rewards.append([])
        self.actions.append([])
        self.critics.append([])
        self.entropies.append([])
        self.states.append([])

    def get_update(self):
        num_traj = loss_stats = critics_stats = entropy_stats = policy_stats = 0.0
        for actions_ep, rewards_ep, critics_ep, entropies_ep, states_ep in zip(self.actions, self.rewards, self.critics, self.entropies, self.states):
            if len(actions_ep) > 0:
                # Compute advantages
                rewards_ep = V(T(rewards_ep))
                critics_ep = th.cat(critics_ep, 0).view(-1)
                rewards_ep, advantage_ep = self.advantage(rewards_ep, critics_ep)
                # Compute losses
                critic_loss = (rewards_ep - critics_ep).pow(2).mean()
                entropy_loss = th.cat(entropies_ep).mean()
                critic_loss = self.critic_weight * critic_loss
                entropy_loss = - self.entropy_weight * entropy_loss
                # Compute policy gradients
                policy_loss = 0.0
                for action, advantage, state in zip(actions_ep, advantage_ep, states_ep):
                    _, new_action = self.forward(state, *action.args, **action.kwargs)
                    ratios = (new_action.log_prob - action.log_prob.detach()).exp().mean()
                    surr1 = ratios * advantage.data[0]
                    surr2 = th.clamp(ratios, 1.0 - self.clip, 1.0 + self.clip) * advantage.data[0]
                    clipped_loss = th.min(surr1, surr2)
                    policy_loss = policy_loss - clipped_loss
                loss = policy_loss + critic_loss + entropy_loss
                loss.backward()
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

#    def updatable(self):
#        if self.update_frequency > 0:
#            if self.steps >= self.update_frequency:
#                if self.epoch_optimized >= self.num_epochs:
#                    self._reset()
#                    return False
#                self.epoch_optimized += 1
#                return True
#        else:
#            if len(self.actions) > 1:
#                if self.epoch_optimized >= self.num_epochs:
#                    self._reset()
#                    return False
#                self.epoch_optimized += 1
#                return True
#        return False
