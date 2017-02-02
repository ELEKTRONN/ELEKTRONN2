# -*- coding: utf-8 -*-
# ELEKTRONN2 Toolkit
# Copyright (c) 2015 Marius F. Killinger
# All rights reserved

from __future__ import absolute_import, division, print_function
from builtins import filter, hex, input, int, map, next, oct, pow, range, \
    super, zip

import logging

import os
import numpy as np
import theano
from theano import tensor as T

from .. import neuromancer
from ..neuromancer.computations import pooling, fragmentpool, fragments2dense, \
    upsampling, unpooling


logger = logging.getLogger('elektronn2log')

import theano.sandbox.cuda
import matplotlib.pyplot as plt


theano.sandbox.cuda.use("gpu0")

from ..neuromancer import graphutils as utils


def test_pooling_3d():
    sig_shape = (1, 4, 30, 200, 200)
    spatial_axes = (2, 3, 4)
    x_val = np.random.rand(*sig_shape).astype(np.float32)
    x = T.TensorType('float32', (False,) * len(sig_shape),
                     name='x_cnn_input')()
    y1 = pooling(x, (3, 2, 2), spatial_axes)
    g1 = theano.grad(T.log(y1).sum(), [x])

    f1 = utils.make_func([x], y1, profile_execution=20, name='poolnew')
    f3 = utils.make_func([x], g1, profile_execution=20, name='poolnewg')

    r1 = f1(x_val)
    r3 = f3(x_val)[0]

    y3 = upsampling(x, (1, 2, 2), spatial_axes)
    gy = theano.grad(T.log(y3).sum(), [x])
    f5 = utils.make_func([x], y3, profile_execution=20, name='unpool')
    f6 = utils.make_func([x], gy, profile_execution=20, name='unpoolg')

    r5 = f5(x_val)
    r6 = f6(x_val)

    y4 = unpooling(x, (1, 2, 2), spatial_axes)
    gy = theano.grad(T.log(y4).sum(), [x])
    f7 = utils.make_func([x], y4, profile_execution=20, name='unpool')
    f8 = utils.make_func([x], gy, profile_execution=20, name='unpoolg')

    r7 = f7(x_val)
    r8 = f8(x_val)


def test_pooling_2d():
    img = plt.imread(os.path.expanduser('~/devel/Lichtenstein.png')).transpose(
        (2, 0, 1))[None,]
    sig_shape = img.shape
    # x_val = np.random.rand(*sig_shape).astype(np.float32)
    inp = neuromancer.Input(sig_shape, 'b,f,x,y')
    conv = neuromancer.Conv(inp, 3, (3, 3), (3, 3), activation_func='tanh')
    w = np.zeros((3, 3, 3, 3), dtype=np.float32)
    w[[0, 1, 2], [0, 1, 2]] = 1
    upconv = neuromancer.UpConv(conv, 3, (3, 3), activation_func='tanh')
    upconv.w.set_value(upconv.w.get_value() * 0.07 + w)

    y0 = conv(img)
    y2 = upconv(img)

    plt.imshow(y0[0].transpose((1, 2, 0)), interpolation='none')
    plt.show()
    plt.figure()
    plt.imshow(y2[0].transpose((1, 2, 0)), interpolation='none')
    plt.show()


#    x = T.TensorType('float32', (False,)*len(sig_shape), name='x_cnn_input')()
#    pool = (1,2,2)
#    offsets = [[0,0,0]]
#    strides = (1,1,1)
#    pool = (2,2)
#    offsets = [[0,0]]
#    strides = (1,1)
#    spatial_axes = [2,3]
#    mfp, offsets_new, strides_new = fragmentpool(x, pool, offsets, strides, spatial_axes)
#    dense = fragments2dense(mfp, offsets_new, strides_new, spatial_axes)
#    f = utils.make_func([x], dense, profile_execution=20, name='dense')
#    rx = f(img)



test_pooling_2d()
