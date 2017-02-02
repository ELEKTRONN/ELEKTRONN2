# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 16:57:36 2016

@author: Marius Felix Killinger
"""
from __future__ import absolute_import, division, print_function
from builtins import filter, hex, input, int, map, next, oct, pow, range, \
    super, zip

import numpy as np
import logging

import theano

from ..neuromancer import Perceptron, Conv, FragmentsToDense
from .. import neuromancer


logger = logging.getLogger('elektronn2log')


def test_neural():
    from elektronn2.neuromancer.graphutils import make_func, as_floatX
    from elektronn2.neuromancer.node_basic import Input
    from elektronn2.neuromancer.loss import AggregateLoss
    import matplotlib.pyplot as plt

    dnn = True
    if dnn:
        x_sh = (1, 4, 100, 100, 30)
        tags = 'b,f,x,y,z'
    else:
        x_sh = (1, 30, 4, 100, 100)
        tags = 'b,z,f,x,y'

    x_val = np.random.rand(*x_sh).astype(np.float32)

    if False:  # Perce[tron]
        # x = Input((1,3,20,20), 'b,f,x,y')
        x = Input(x_sh, 'b,s,f,x,y')
        W = [np.zeros((3, 5), dtype=np.float32), 'const']
        b = np.arange(2 * 5, dtype=np.float32).reshape(5, 2)
        y = Perceptron(x, 5, activation_func='prelu', W=W, b=b,
                       batch_normalisation='train')
        print(y.all_params_count)
        ret = y(x_val)
        assert np.allclose(ret[0, :, 0, 0], np.arange(0, 9, 2))

        W = [np.zeros((1200, 5), dtype=np.float32), 'trainable']
        b = np.arange(2 * 5, dtype=np.float32).reshape(5, 2)
        y = Perceptron(x, 5, activation_func='prelu', W=W, b=b,
                       batch_normalisation='train', flatten=True)
        print(y.all_params_count)
        ret = y(x_val)
        assert np.allclose(ret, np.arange(0, 9, 2))

        y = Perceptron(x, 5)
        z = y.make_dual(y, share_w=True)
        ret = z(x_val)
        logger.debug(ret.shape)

    if False:  # 3d Conv + MFP
        x = Input(x_sh, tags)
        y = Conv(x, 16, (3, 3, 3), (2, 2, 2), mfp=True)
        ret = y(x_val)
        y2 = FragmentsToDense(y)
        ret2 = y2(x_val)
        logger.debug(y2.last_exec_time)

    if True:  # visual 2d conv + MFP
        img = plt.imread('/docs/devel/svn/Marius/Lichtenstein.png').transpose(
            (2, 0, 1))[None,]
        x = Input(img.shape, 'b,f,x,y')
        y = Conv(x, 3, (3, 3), (3, 3), activation_func='tanh', mfp=True)
        y1 = FragmentsToDense(y)

        out1 = y1(img)[0]
        logger.debug(y1.last_exec_time)
        out1 = out1.transpose((1, 2, 0))

        plt.figure()
        plt.imshow(out1, interpolation='none')
        plt.show()

    if True:  # visual 2d conv + upconv
        img = plt.imread('/docs/devel/svn/Marius/Lichtenstein.png').transpose(
            (2, 0, 1))[None,]
        x = Input(img.shape, 'b,f,x,y')
        y = Conv(x, 3, (3, 3), (3, 3), activation_func='tanh')
        z = y.make_dual(y, share_w=True)

        out = z(img)[0]
        logger.debug(z.last_exec_time)
        out = out.transpose((1, 2, 0))

        plt.figure()
        plt.imshow(out, interpolation='none')
        plt.show()

    if False:  # Benchmark 3d Conv
        #        in_sh = (1, 1, 20, 174, 174)
        #        x = Input(in_sh, 'b,f,z,x,y')
        #        in_val = np.random.rand(*in_sh).astype(np.float32)
        #
        #        y1 = Conv(x, 12, (1,6,6), (1,2,2), mfp=True)
        #        y2 = Conv(y1, 24, (4,4,4), (2,2,2), mfp=True)
        #        y3 = Conv(y2, 64, (4,4,4), (1,2,2), mfp=True)
        #        y4 = Conv(y3, 64, (4,4,4), (1,1,1), mfp=False, name='Fragments')

        in_sh = (1, 1, 174, 174, 20)
        x = Input(in_sh, 'b,f,z,x,y')
        in_val = np.random.rand(*in_sh).astype(np.float32)

        y1 = Conv(x, 12, (1, 6, 6)[::-1], (1, 2, 2)[::-1], mfp=True)
        y2 = Conv(y1, 24, (4, 4, 4)[::-1], (2, 2, 2)[::-1], mfp=True)
        y3 = Conv(y2, 64, (4, 4, 4)[::-1], (1, 2, 2)[::-1], mfp=True)
        y4 = Conv(y3, 64, (4, 4, 4)[::-1], (1, 1, 1)[::-1], mfp=False,
                  name='Fragments')

        y5 = FragmentsToDense(y4, name='MFP-reshape')

        out4 = y4(in_val)
        logger.debug(y4.last_exec_time)  # 0.208 dnn # 0.325
        out5 = y5(in_val)
        logger.debug(y5.last_exec_time)  # 0.205 dnn # 0.2969

        y6 = AggregateLoss([y5, ], name='loss')
        g = theano.grad(y6.output, y6.all_trainable_params.values())
        g_func = make_func([x.output, ], g, name='grad')
        outg = g_func(in_val)
        logger.debug(g_func.last_exec_time)  # 1.11, 3.5GB dnn # 1.27

    data = neuromancer.Input((3, 4, 7, 10, 10), 'r, b, f, x,y')
    data0, _ = neuromancer.split(data, axis='r', index=1,
                                 strip_singleton_dims=True)
    mem_init0 = neuromancer.InitialState_like(data0, override_f=13,
                                              init_kwargs=dict(mode='fix-uni',
                                                               scale=0.1))
    mem_init1 = neuromancer.InitialState_like(data0, override_f=26,
                                              init_kwargs=dict(mode='fix-uni',
                                                               scale=0.1))
    gru_state = neuromancer.GRU(data0, mem_init0, 13)
    lstm_state = neuromancer.LSTM(data0, mem_init1, 13)

    y0 = gru_state.test_run()
    y1 = lstm_state.test_run()

    gru_s = neuromancer.Scan(gru_state, mem_init0, gru_state, in_iterate=data,
                             in_iterate_0=data0)
    lstm_s = neuromancer.Scan(lstm_state, mem_init1, lstm_state,
                              in_iterate=data, in_iterate_0=data0)

    z0 = gru_s.test_run()
    z1 = lstm_s.test_run()
