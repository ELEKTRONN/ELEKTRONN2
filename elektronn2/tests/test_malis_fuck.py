# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 16:57:36 2016

@author: Marius Felix Killinger
"""
from __future__ import absolute_import, division, print_function
from builtins import filter, hex, input, int, map, next, oct, pow, range, \
    super, zip


def test_malis_fuck():
    import numpy as np
    # import theano.sandbox.cuda
    # theano.sandbox.cuda.use("gpu0")
    import theano
    import theano.tensor as T
    from elektronn2.malis import malis_utils
    from elektronn2.malis.malisop import malis_weights

    print("TESTING/DEMO:")

    aff_pred_t = T.TensorType('float32', [False, ] * 4, name='aff_pred')()
    aff_gt_t = T.TensorType('int16', [False, ] * 4, name='aff_gt')()
    seg_gt_t = T.TensorType('int16', [False, ] * 3, name='seg_gt')()
    neigh_t = T.TensorType('int32', [False, ] * 2, name='neighb')()

    pos_t, neg_t = malis_weights(aff_pred_t, aff_gt_t, seg_gt_t, neigh_t)
    loss_t = T.sum(pos_t * aff_pred_t)

    f = theano.function([aff_pred_t, aff_gt_t, seg_gt_t, neigh_t],
                        [pos_t, neg_t, loss_t])
    grad_t = theano.grad(loss_t, aff_pred_t)
    f2 = theano.function([aff_pred_t, aff_gt_t, seg_gt_t, neigh_t], [grad_t, ])

    nhood = np.array([[0., 1., 0.], [0., 0., 1.]], dtype=np.int32)

    test_id2 = np.array([[[1, 1, 2, 2, 0, 0, 3, 3], [1, 1, 2, 2, 0, 0, 3, 3],
                          [1, 1, 2, 2, 0, 0, 3, 3], [1, 1, 2, 2, 0, 0, 3, 3]]],
                        dtype=np.int32)

    aff_gt = malis_utils.seg_to_affgraph(test_id2, nhood)
    seg_gt = malis_utils.affgraph_to_seg(aff_gt, nhood)[0].astype(np.int16)
    aff_pred = np.array([[[[1., 1., 1., 1., 0., 0., 1., 1.],
                           [1., 1., 1., 1., 0., 0., 1., 1.],
                           [0.9, 0.8, 1., 1., 0., 0., 1., 1.],
                           [0., 0., 0., 0., 0., 0., 1., 1.]]],

                         [[[1., 0., 1., 0.3, 0.2, 0.3, 1., 0.],
                           [0.7, 0., 1., 0., 0., 0., 1., 0.],
                           [1., 0.2, 1., 0., 0., 0., 1., 0.],
                           [1., 0., 1., 0., 0., 0., 1., 0.]]]]).astype(
        np.float32)

    pos, neg, loss = f(aff_pred, aff_gt, seg_gt, nhood)
    print("loss", loss)
    print("pos counts\n", pos)
    print('-' * 40)
    print("neg counts\n", neg)
    print('-' * 40)

    g = f2(aff_pred, aff_gt, seg_gt, nhood)
    print(g[0])

    g_true = np.array([[[[3., 2., 4., 0., 0., 3.], [8., 0., 16., 0., 0., 2.],
                         [12., 0., 1., 0., 0., 1.], [0., 0., 0., 0., 0., 0.]]],

                       [[[1., 0., 1., 0., 0., 0.], [0., 0., 1., 0., 0., 0.],
                         [1., 0., 2., 0., 0., 0.], [1., 0., 3., 0., 0., 0.]]]])

    pos_true = np.array([[[[3, 2, 4, 0, 0, 3], [8, 0, 16, 0, 0, 2],
                           [12, 0, 1, 0, 0, 1], [0, 0, 0, 0, 0, 0]]],

                         [[[1, 0, 1, 0, 0, 0], [0, 0, 1, 0, 0, 0],
                           [1, 0, 2, 0, 0, 0], [1, 0, 3, 0, 0, 0]]]],
                        dtype=np.int32)

    assert np.allclose(g[0], g_true)
    assert np.allclose(pos, pos_true)

    nhood = np.array([[0., 1., 0.], [0., 0., 1.]])

    test_id2 = np.array([[[1, 1, 2, 2, 0, 3], [1, 1, 2, 2, 0, 3],
                          [1, 1, 2, 2, 0, 3], [1, 1, 2, 2, 0, 3]]],
                        dtype=np.int32)

    aff_gt = malis_utils.seg_to_affgraph(test_id2, nhood)
    aff_pred = np.array([[[[1., 1., 1., 1., 0., 1.], [1., 1., 1., 1., 0., 1.],
                           [0.9, 0.8, 1., 1., 0., 1.],
                           [0., 0., 0., 0., 0., 1.]]],

                         [[[1., 0., 1., 0.3, 0.4, 0.],
                           [0.7, 0., 1., 0., 0., 0.],
                           [1., 0.2, 1., 0., 0., 0.],
                           [1., 0., 1., 0., 0., 0.]]]]).astype(np.float32)
