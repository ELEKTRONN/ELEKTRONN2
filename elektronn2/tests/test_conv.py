from __future__ import absolute_import, division, print_function
from builtins import filter, hex, input, int, map, next, oct, pow, range, \
    super, zip

import numpy as np
import theano
import theano.tensor as T
from theano.tensor.nnet.conv import conv2d
from theano.tensor.nnet.conv3d2d import conv3d
from theano.sandbox.cuda import dnn
from theano.tensor.signal.downsample import max_pool_2d

import theano.sandbox.cuda


theano.sandbox.cuda.use("gpu0")
from ..neuromancer import graphutils as utils
import elektronn2.neuromancer.variables

from ..neuromancer.computations import *


def test_conv():
    if False:
        sig_shape = (100000, 20)
        fil_shape = (20, 30)
        x_val = np.random.rand(*sig_shape).astype(np.float32)
        W_val = np.random.rand(*fil_shape).astype(
            np.float32)  # (nof, z, ch, xf, yf)

        x = T.TensorType('float32', (False,) * 2, name='x_cnn_input')()
        W = elektronn2.neuromancer.variables.VariableParam(W_val)
        y1 = T.dot(x, W)
        y2 = dot(x, W, 1)

        g1 = theano.grad(T.log(y1).sum(), [x])
        g2 = theano.grad(T.log(y2).sum(), [x])

        f1 = utils.make_func([x], y1, profile_execution=10)
        f2 = utils.make_func([x], y2, profile_execution=10)

        r1 = f1(x_val)
        r2 = f2(x_val)
        assert np.allclose(r1, r2)

        sig_shape = (1, 5, 300, 200)
        fil_shape = (7, 5, 1, 1)
        x_val = np.random.rand(*sig_shape).astype(np.float32)
        W_val = np.random.rand(*fil_shape).astype(
            np.float32)  # (nof, z, ch, xf, yf)

        x = T.TensorType('float32', (False,) * 4, name='x_cnn_input')()
        W = elektronn2.neuromancer.variables.VariableParam(W_val)
        y1 = conv(x, W, w_shape=fil_shape)
        y2 = conv2d(x, W)

        g1 = theano.grad(T.log(y1).sum(), [x])
        g2 = theano.grad(T.log(y2).sum(), [x])

        f1 = utils.make_func([x], y1, profile_execution=5)
        f2 = utils.make_func([x], y2, profile_execution=5)

        r1 = f1(x_val)
        r2 = f2(x_val)
        assert np.allclose(r1, r2)

        sig_shape = (1, 100, 5, 300, 200)
        # x_shape = (None, 100, 5, 300, 200)
        fil_shape = (7, 3, 5, 3, 3)
        x_val = np.random.rand(*sig_shape).astype(np.float32)
        W_val = np.random.rand(*fil_shape).astype(
            np.float32)  # (nof, z, ch, xf, yf)

        x = T.TensorType('float32', (False,) * 5, name='x_cnn_input')()
        W = elektronn2.neuromancer.variables.VariableParam(W_val)
        y1 = conv(x, W, x_shape=sig_shape, w_shape=fil_shape)
        y2 = conv3d(x, W)

        g1 = theano.grad(T.log(y1).sum(), [x])
        g2 = theano.grad(T.log(y2).sum(), [x])

        f1 = utils.make_func([x], y1, profile_execution=5)
        f2 = utils.make_func([x], y2, profile_execution=5)

        r1 = f1(x_val)
        r2 = f2(x_val)
        assert np.allclose(r1, r2)

        sig_shape = (1, 1, 100)
        fil_shape = (1, 1, 20)
        x_val = np.random.rand(*sig_shape).astype(np.float32)
        W_val = np.random.rand(*fil_shape).astype(
            np.float32)  # (nof, z, ch, xf, yf)

        x = T.TensorType('float32', (False,) * 3, name='x_cnn_input')()
        W = elektronn2.neuromancer.variables.VariableParam(W_val)
        y1 = conv(x, W, w_shape=fil_shape)

        f1 = elektronn2.neuromancer.graphutils.make_func([x], y1,
                                                         profile_execution=10)

        r1 = f1(x_val)
        r2 = np.convolve(x_val[0, 0], W_val[0, 0], mode='valid')[None, None]
        assert np.allclose(r1, r2)

    if True:
        sig_shape = (1, 5, 100, 300, 200)
        fil_shape = (7, 5, 3, 3, 3)
        x_val = np.random.rand(*sig_shape).astype(np.float32)
        W_val = np.random.rand(*fil_shape).astype(
            np.float32)  # (nof, ch, zf xf, yf)

        x = T.TensorType('float32', (False,) * 5, name='x_cnn_input')()
        W = elektronn2.neuromancer.variables.VariableParam(W_val)

        # test conv
        y1 = dnn.dnn_conv3d(x, W, border_mode='valid')
        y2 = conv3d(x.dimshuffle(0, 2, 1, 3, 4),
                    W.dimshuffle(0, 2, 1, 3, 4)).dimshuffle(0, 2, 1, 3, 4)

        f1 = elektronn2.neuromancer.graphutils.make_func([x], y1,
                                                         profile_execution=5)
        f2 = elektronn2.neuromancer.graphutils.make_func([x], y2,
                                                         profile_execution=5)

        r3 = np.array(f1(x_val))
        r4 = f2(x_val)
        assert np.allclose(r3,
                           r4)  # cudnn and reshaped conv2d3d give same result, but cudnn ist faster!

        y1 = dnn.dnn_conv3d(x, W, border_mode='valid')
        y1 = dnn.dnn_pool(y1, (2, 2, 2), stride=(2, 2, 2), pad=(0, 0, 0),
                          mode='max')
        f1 = elektronn2.neuromancer.graphutils.make_func([x], y1,
                                                         profile_execution=5)
        r3 = np.array(f1(x_val))

        y2 = conv3d(x.dimshuffle(0, 2, 1, 3, 4), W.dimshuffle(0, 2, 1, 3, 4))
        y2 = pooling(y2, (2, 2, 2))
        f2 = elektronn2.neuromancer.graphutils.make_func([x], y2,
                                                         profile_execution=5)
        r4 = f2(x_val)
        assert np.allclose(r3, r4.transpose(0, 2, 1, 3,
                                            4))  # pooling als works, not it is not so much faster anymore....

        y1 = dnn.dnn_conv3d(x, W, border_mode='valid')
        y1 = dnn.dnn_pool(y1, (2, 2, 2), stride=(2, 2, 2), pad=(0, 0, 0),
                          mode='max')
        sm = dnn.GpuDnnSoftmax('bc01', 'fast', 'channel')
        sh = y1.shape
        y1 = sm(y1.flatten(4)).reshape(sh)
        f1 = elektronn2.neuromancer.graphutils.make_func([x], y1,
                                                         profile_execution=5)
        r3 = np.array(f1(x_val))

        y2 = conv3d(x.dimshuffle(0, 2, 1, 3, 4), W.dimshuffle(0, 2, 1, 3, 4))
        y2 = pooling(y2, (2, 2, 2))
        y2 = softmax(y2, axis=2)
        f2 = elektronn2.neuromancer.graphutils.make_func([x], y2,
                                                         profile_execution=5)
        r4 = f2(x_val)
        assert np.allclose(r3, r4.transpose(0, 2, 1, 3, 4),
                           atol=1e-5)  # sm also works but diff is ~1e-5
