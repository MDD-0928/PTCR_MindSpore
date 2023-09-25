# encoding: utf-8

import mindspore as ms
import mindspore.nn as nn
from cmath import inf
from mindspore import Tensor
import mindspore.ops as P

class GeneralizedMeanPoolingP(nn.Cell):
    r"""Applies a 2D power-average adaptive pooling over an input signal composed of several input planes.
    The function computed is: :math:`f(X) = pow(sum(pow(X, p)), 1/p)`
        - At p = infinity, one gets Max Pooling
        - At p = 1, one gets Average Pooling
    The output is of size H x W, for any input size.
    The number of output features is equal to the number of input planes.
    Args:
        output_size: the target output size of the image of the form H x W.
                     Can be a tuple (H, W) or a single H for a square image H x H
                     H and W can be either a ``int``, or ``None`` which means the size will
                     be the same as that of the input.
    """

    def __init__(self, norm=3, output_size=1, eps=1e-6):
        super(GeneralizedMeanPoolingP, self).__init__()
        assert norm > 0
        self.p = ms.Parameter(default_input=ms.ops.ones(1) * norm, name='p')
        self.output_size = output_size
        self.eps = eps

    def construct(self, x):
        x = x.clamp(min=self.eps).pow(self.p)
        return ms.ops.adaptive_avg_pool2d(x, self.output_size).pow(1. / self.p)

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + str(self.p) + ', ' \
               + 'output_size=' + str(self.output_size) + ')'
