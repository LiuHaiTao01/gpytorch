from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import random
import torch
from .random_variable import RandomVariable


class SamplesRandomVariable(RandomVariable):
    def __init__(self, samples):
        """
        Constructs a random variable from samples

        samples should be a Tensor, with the first dimension representing the samples

        Params:
        - samples (Tensor: b x ...) samples
        """
        if not torch.is_tensor(samples):
            raise RuntimeError("samples should be a Tensor")
        super(SamplesRandomVariable, self).__init__(samples)
        self._samples = samples

    def sample(self):
        ix = random.randrange(len(self._sample_list))
        return self._samples.data[ix]

    def representation(self):
        return self._samples

    def mean(self):
        return self._samples.mean(-1)

    def var(self):
        if self._samples.size(-1) == 1:
            return self._samples.data.new(self._samples.squeeze(-1).size()).zero_()
        return self._samples.var(-1)
