#!/usr/bin/env python

import torch as th
from torch.autograd import Variable as V

from .base import BaseAgent


class Random(BaseAgent):

    def __init__(self, policy, *args, **kwargs):
        self.output_size = policy.model.num_out

    def act(self, state):
        return th.rand(self.output_size).numpy(), None
