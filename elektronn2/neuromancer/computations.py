# -*- coding: utf-8 -*-
# ELEKTRONN2 Toolkit
# Copyright (c) 2015 Marius F. Killinger
# All rights reserved

from __future__ import absolute_import, division, print_function
from builtins import filter, hex, input, int, map, next, oct, pow, range, super, zip

from itertools import product
import logging
import re

import numpy as np
import theano
import theano.tensor as T
from theano.tensor.nnet import conv2d
from theano.tensor.nnet.abstract_conv import conv2d_grad_wrt_inputs
from theano.tensor.nnet.conv3d2d import conv3d
from theano.sandbox.cuda import dnn
from theano.tensor.signal.pool import pool_2d

from ..config import config

logger = logging.getLogger('elektronn2log')

dnn_avail = dnn.dnn_available()
logger.warning("Manual dnn calls for conv, are much faster for prediction "
               "and much slower for training.")

dnn_algo = 'small' # Only 'none' is implemented for the conv3d

def apply_except_axis(x, axis, func):
    """
    Apply a contraction function on all but one axis.

    Parameters
    ----------
    x: T.Tensor
        Input tensor.
    axis: int
        Axis to exclude on application.
    func: function
        A function with signature ``func(x, axis=)`` eg T.mean, T.std ...

    Returns
    -------
    T.Tensor
        Contraction of ``x``, but of the same dimensionality.
    """
    x = T.swapaxes(x, 0, axis) # put axis on front
    x = T.flatten(x, 2) # flatten remainder
    y = func(x, axis=1)
    return y


def apply_activation(x, activation_func, b1=None):
    """
    Return an activation function callable matching the name
    Allowed names: 'relu', 'tanh','prelu', 'sigmoid', 'maxout <i>',
    'lin','abs','soft+'.

    Parameters
    ----------
    x: T.Tensor
        Input tensor.
    activation_func: str
        Name of the activation function.
    b1
        Optional b1 parameter for the activation function.
        If this is None, no parameter is passed.

    Returns
    -------
    T.Tensor
        Activation function applied to ``x``.
    """
    func = None
    if activation_func=='tanh': # range = [-1,1]
        func = T.tanh
    elif activation_func=='relu': # rectified linear unit ,range = [0,inf]
        def relu(y):
            return (0.5 * (y + abs(y)))
        func = relu

    elif activation_func=='prelu': # parameterised relu
        def prelu(y, alpha):
            pos = ((y + abs(y)) / 2.0)
            neg = alpha * ((y - abs(y)) / 2.0)
            return pos + neg
        func = prelu

    elif activation_func=='abs': # abs unit ,range = [0,inf]
        func = T.abs_
    elif activation_func in ['sig', 'logistic', 'sigmoid']: # range = [0,1]
        func = T.nnet.sigmoid
    elif activation_func=='soft+':
        func = T.nnet.softplus
    elif activation_func in ['lin', 'linear']:
        def lin(y):
            return y
        func = lin
    elif activation_func=='concentration':
        def func(y):
            #return T.square(T.nnet.softplus(y)) + 2.0
            return T.nnet.softplus(y)*7 + 2.0
    elif activation_func=='radius':
        def func(y):
            return 3+15*T.exp(0.15*y) #T.nnet.softplus(y+5)**1.5
    elif activation_func.startswith("maxout"):
        r = int(re.findall('\d+', activation_func)[0])
        def _maxout(y):
            return maxout(y, factor=r)

        func = _maxout

    else:
        funcs = ['relu', 'tanh','prelu', 'sigmoid', 'maxout <i>',
                 'lin','abs','soft+', 'concentration', 'radius']
        raise NotImplementedError("%s. Permitted activation_funcs :%s" \
                                  %( activation_func, funcs,))

    if b1 is None:
        y = func(x)
    else:
        y = func(x, b1)
    return y


def softmax(x, axis=1, force_builtin=False):
    """
    Calculate softmax (pseudo probabilities).

    Parameters
    ----------
    x: T.Tensor
        Input tensor.
    axis: int
        Axis on which to apply softmax.
    force_builtin: bool
        force usage of ``theano.tensor.nnet.softmax`` (more stable).

    Returns
    -------
    T.Tensor
        ``x`` with softmax applied, same shape.
    """
    if dnn_avail and config.use_manual_cudnn_conv: # oder must always be bc01, and x must always be 4d
        if axis==1:
            dnn_sm = dnn.GpuDnnSoftmax('bc01', 'accurate',  'channel')
            if x.ndim==4:
                logger.debug("Using cuDNN softmax")
                y = dnn_sm(x)
                return y
            elif x.ndim==5: # remap to 4d, this is memory friendly
                logger.debug("Using cuDNN softmax")
                sh = x.shape
                y  = dnn_sm(x.flatten(4)).reshape(sh)
                return y
        # if axis!=1 use own softmax (I don't want to do dimshuffles just to use
        # dnn, it is not that much faster anyway)

    if x.ndim==2 and axis==1:
        return T.nnet.softmax(x)
    elif force_builtin:
        raise NotImplementedError()
    else:
        e_x = T.exp(x - x.max(axis, keepdims=True))
        y   = e_x / e_x.sum(axis, keepdims=True)
        return y

def dot(x, W, axis=1):
    """
    Calculate a tensordot between 1 axis of ``x`` and the first axis of ``W``.

    Requires ``x.shape[axis]==W.shape[0]``.
    Identical to dot if ``x``, ``W`` 2d and ``axis==1``.

    Parameters
    ----------
    x: T.Tensor
        Input tensor.
    W: T.Tensor
        Weight tensor, (f_in, f_out).
    axis: int
        Axis on ``x`` to apply dot.

    Returns
    -------
    T.Tensor
        ``x`` with dot applied. The shape of ``x`` changes on ``axis``
        to ``n_out``.
    """
    if axis<1:
        raise ValueError("Dot on first axis is not supported")

    if x.ndim==2 and axis==1:
        return T.dot(x, W)

    y = T.tensordot(x, W, axes=[axis,0]) # (1, 100, 5, 300, 200) (5, 7) --> (1, 100, 300, 200, 7)
    k = x.ndim - axis
    end = np.roll(np.arange(k)+axis, 1)
    begin = np.arange(axis)
    pattern = np.concatenate((begin,end))
    y = y.transpose(*pattern)
    return y


def upconv(x, w, stride, x_shape=None, w_shape=None, axis_order='dnn'):
    assert stride is not None
    stride = tuple(stride)
    conv_dim = len(stride)
    border_mode = 'valid'

    # if (x_shape is None) or (None in x_shape):  # variable batch size or so
    #     x_shape = None

    if conv_dim==1:
        x = x.dimshuffle(0, 1, 2, 'x')
        w = w.dimshuffle(0, 1, 2, 'x')
        if w_shape is not None:
            w_shape = list(w_shape) + [1, ]
        if x_shape is not None:
            x_shape = list(x_shape) + [1, ]

        stride  = list(stride) + [1, ]
        y = conv2d_grad_wrt_inputs(x, w, x_shape, w_shape, border_mode,
                                   subsample=stride, filter_flip=False)
        y = y[:, :, :, 0]

    elif conv_dim==2:
        y = conv2d_grad_wrt_inputs(x, w, x_shape, w_shape, border_mode,
                                       subsample=stride, filter_flip=False)

    elif conv_dim==3:
        if not dnn_avail or axis_order!='dnn':
            raise ValueError("Need dnn and dnn axis order")
        kerns = dnn.gpu_contiguous(w)
        image = dnn.gpu_contiguous(x)
        k = kerns.shape[1]
        img_sh = list(image.shape)
        out_sh = img_sh[:1] + [k,] + [st*sh for st, sh in zip(stride, img_sh[2:])]
        out = dnn.gpu_alloc_empty(*out_sh)
        desc = dnn.GpuDnnConvDesc(border_mode='valid', subsample=stride,
                                  conv_mode='cross')(out.shape, kerns.shape)
        y = dnn.GpuDnnConv3dGradI()(kerns, image, out, desc)

    return y

### TODO clearer error messages about which arguments can be used
# for dnn and theano respectively. Transform asserts into exceptions
def conv(x, w, axis_order=None, conv_dim=None, x_shape=None, w_shape=None,
         border_mode='valid', stride=None):
    """
    Apply appropriate convolution depending on input and filter dimensionality.
    If input ``w_shape`` is known, conv might be replaced by tensordot

    There are static assumptions which axes are spatial.

    Parameters
    ----------
    x: T.Tensor
        | Input data (mini-batch).
        | Tensor of shape ``(b, f, x)``, ``(b, f, x, y)``, ``(b, z, f, x, y)``
          or ``(b,f,x,y,z)``.
    w: T.Tensor
        | Set of convolution filter weights.
        | Tensor of shape ``(f_out, f_in, x)``, ``(f_out, f_in, x, y)``,
          ``(f_out, z, f_in, x, y)`` or ``(f_out, f_in, x, y, z)``.
    axis_order: str
        | (only relevant for 3d)
        | ``'dnn'`` ``(b,f,x,y(,z))`` or ``'theano'`` ``(b, z, f, x, y)``.
    conv_dim: int
        Dimensionality of the applied convolution (not the absolute dim of
        the inputs).
    x_shape: tuple
        shape tuple (``TaggedShape`` supported).
    w_shape: tuple
        shape tuple, see ``w``.
    border_mode: str
        * ``'valid'``: only apply filter to complete patches of the image.
          Generates output of shape: image_shape -filter_shape + 1.
        * ``'full'`` zero-pads image to multiple of filter shape to generate
          output of shape: image_shape + filter_shape - 1.
    stride: tuple
        | (tuple of len 2)
        | Factor by which to subsample the output.

    Returns
    -------
    T.Tensor
        Set of feature maps generated by convolution.
    """
    if (x_shape is None) or (None in x_shape): # variable batch size or so
        x_shape = None

    assert axis_order in ['dnn', 'theano', None]

    if conv_dim is not None:
        if x.ndim!=conv_dim+2 or w.ndim!=conv_dim+2:
            raise ValueError("Cannot perform %id conv on input and filter of"
                             "dim %i, %i" % (conv_dim, x.ndim, w.ndim))

    else: # infer conv_dim
        conv_dim = x.ndim-2
        if w.ndim!=conv_dim+2:
            raise ValueError("Dimension mismatch for conv: tried to do %id conv"
                             "on %id input x. This requires %id filter, but got"
                             "%id" % (conv_dim, x.ndim, x.ndim, w.ndim))
        if conv_dim>3:
            raise ValueError("Input tensor dim to big. No conv for dim>5.")

    if border_mode=='same':
        assert w_shape is not None
        assert np.all(np.remainder(w_shape[-conv_dim:], 2)==1), "For conv same kernels must be uneven"
        border_mode='full'
        crop_full = True
    else:
        crop_full = False

    use_tensordot = False
    if (w_shape is not None) and (stride is None): # cannot use tensordot with strides
        if conv_dim<3 or axis_order=='dnn':
            use_tensordot = np.all(np.equal(w_shape[2:], 1))
        else: # theano order for 3d conv
            use_tensordot = w_shape[1] == 1 and np.all(np.equal(w_shape[3:], 1))

    y = None
    if conv_dim==1:
        x = x.dimshuffle(0, 1, 2, 'x')
        w = w.dimshuffle(0, 1, 2, 'x')
        if w_shape is not None:
            w_shape = list(w_shape) + [1, ]
        if x_shape is not None:
            x_shape = list(x_shape) + [1,]
        if stride is None:
            stride = (1, 1)
        y = conv2d(x, w, x_shape, w_shape, border_mode, subsample=stride)
        y = y[:, :, :, 0]

    elif conv_dim==2:
        if stride is None:
            stride = (1, 1)
        if use_tensordot:
            logger.debug("Using dot for 2d conv")
            w = w[:, :, 0, 0].T # (f_in, f_out) (5, 7)
            y = dot(x, w, axis=1)

        elif dnn_avail and config.use_manual_cudnn_conv:
            logger.debug("Using cuDNN 2dconv")
            y = dnn.dnn_conv(x, w, border_mode, subsample=stride, algo=dnn_algo)
        else: # fallback to theano
            y = conv2d(x, w, x_shape, w_shape, border_mode, subsample=stride)

    elif conv_dim==3:
        assert axis_order in ['dnn', 'theano']
        use_dnn = dnn_avail
        if not config.use_manual_cudnn_conv:
            use_dnn = False
        if w_shape[2]==1 and config.use_manual_cudnn_conv_not_w1:
            use_dnn = False
            logger.debug("Ignoring manual 3d cuDNN conv because kernel is "
            "1 for first axis") # then theano automatically uses dnn 2d conv which
            # has faster gradient than dnn 3d conv

        if stride is not None:
            raise NotImplementedError("Cannot use strided conv with 3d conv")
        if use_tensordot:
            logger.debug("Using dot for 3d conv")
            if axis_order=='theano':
                w = w[:, 0, :, 0, 0].T # (f_in, f_out)
                y = dot(x, w, axis=2)
            elif axis_order=='dnn':
                w = w[:, :, 0, 0, 0].T # (f_in, f_out)
                y = dot(x, w, axis=1)

        elif use_dnn:
            if stride is None:
                stride = (1, 1, 1)
            if axis_order=='dnn':
                logger.debug("Using cuDNN 3dconv")
                y = dnn.dnn_conv3d(x, w, border_mode, subsample=stride, algo=dnn_algo) # (b, f, x, y, z)
            else:
                if config.show_axis_order_warning:
                    logger.warning("cuDNN available but axis order is "
                    "for theano (z before f). This leads to possibly "
                    "inefficient dimshuffles. use cuDNN axis order.\n"
                    "Using dnn 3dconv")
                x = x.dimshuffle(0,2,1,3,4)
                w = w.dimshuffle(0, 2, 1, 3, 4)
                y = dnn.dnn_conv3d(x, w, border_mode, subsample=stride, algo=dnn_algo) # (b, f, x, y, z)
                y = y.dimshuffle(0,2,1,3,4)

        else: # fallback to theano
            if axis_order=='theano':
                logger.debug("Using theano 3dconv")
                y = conv3d(x, w, x_shape, w_shape, border_mode) # (b, z, f, x, y)
            else:
                if config.use_manual_cudnn_conv and not dnn_avail:
                    if config.show_axis_order_warning:
                        logger.warning("cuDNN not available but axis order is"
                         "for cuDNN (z after features). This leads to possibly "
                         "inefficient dimshuffles Use theano axis order or "
                         "install cuDNN.\nUsing theano 3dconv")
                x = x.dimshuffle(0,2,1,3,4)
                w = w.dimshuffle(0, 2, 1, 3, 4)
                # Also swap shapes!
                w_shape = list(w_shape)
                z,f = w_shape[1], w_shape[2]
                w_shape[2] = z
                w_shape[1] = f
                if x_shape is not None:
                    x_shape = list(x_shape)
                    z,f = x_shape[1], x_shape[2]
                    x_shape[2] = z
                    x_shape[1] = f

                y = conv3d(x, w, x_shape, w_shape, border_mode) # (b, z, f, x, y)
                y = y.dimshuffle(0,2,1,3,4)

    if crop_full:
        cropper = []
        off = np.divide(w_shape[-conv_dim:], 2).astype(np.int)
        k = 0
        if axis_order=='theano' and conv_dim==3:
            for i in range(y.ndim):
                if i in [1,3,4]:
                    cropper.append(slice(off[k], -off[k]))
                    k += 1
                else:
                    cropper.append(slice(None))
        else:
            for i in range(y.ndim):
                if i >= y.ndim - conv_dim:
                    cropper.append(slice(off[k], -off[k]))
                    k += 1
                else:
                    cropper.append(slice(None))

        cropper = tuple(cropper)
        y = y[cropper]

    return y


def maxout(x, factor=2, axis=None):
    """
    Maxpooling along the feature axis.

    The feature count is reduces by ``factor``.

    Parameters
    ----------
    x: T.Tensor
        Input tensor (b, f, x, y), (b, z, f, x, y).
    factor: int
        Pooling factor.
    axis: int or None
        Feature axis of ``x`` (1 or 2).
        If None, 5d tensors get axis 2 and all others axis 1.

    Returns
    -------
    T.Tensor
        ``x`` with pooling applied.
    """
    if axis is None:
        axis = 2 if x.ndim==5 else 2

    if axis not in [1,2]:
        raise ValueError("Maxout only permitted on axis 1 or 2")

    y = None
    if axis==1:
        y =  x[:,0::factor]
        for i in range(1, factor):
            t = x[:,i::factor]
            y = T.maximum(y, t)

    elif axis==2:
        y =  x[:,:,0::factor]
        for i in range(1, factor):
            t = x[:,:,i::factor]
            y = T.maximum(y, t)

    return y


# def pooling_3d2d(x, pool, spatial_axes, func=T.max,):
#     """
#     Pooling along spatial axes of 3d and 2d tensors.
#     There are static assumptions which axes are spatial.
#     The axes of ``x`` must be divisible by the corresponding pooling factor,
#     otherwise the computation might crash later.
#     :param x: tensor (b, f, x, y), (b, z, f, x, y)
#     :param pool:2/3-tuple of pooling factors. They refer to the spatial
#     axes of ``x`` (x,y)/(z,x,y)
#     :param func: function with signature ``f(x, axis=)`` e.g. ``T.mean``, ``T.max``
#     :return: ``x`` with pooling applied. The spatial axes are decrease by the
#     corresponding pooling factors
#     """
#     spatial_axes = list(spatial_axes)
#     _pool = np.ones(x.ndim, np.int)
#     _pool[spatial_axes] = pool
#     return pooling_nd(x, _pool, func)
#
#
# def pooling_nd(x, pool, func=T.max):
#     """
#     Pooling along spatial axes of nd tensors.
#     The axes of ``x`` must be divisible by the corresponding pooling factor,
#     otherwise the computation might crash later.
#     :param x: nd tensor
#     :param pool: tuple of length n
#     :param func: function with signature ``f(x, axis=)`` e.g. ``T.mean``, ``T.max``
#     :return: ``x`` with pooling applied. The axis lengths are decrease by the
#     corresponding pooling factors
#     """
#     accum =[]
#     for i,ix in enumerate(product(*map(lambda x: list(np.arange(x)), pool))):
#         sl = tuple([slice(x[0], x[1], x[2]) for x in zip(ix, x.shape, pool)])
#         accum.append(x[sl])
#
#     accum = T.stacklists(accum)
#     y = func(accum, axis=0)
#     return y


def pooling(x, pool, spatial_axes, mode='max', stride=None):
    """
    Pooling along spatial axes of 3d and 2d tensors.

    There are static assumptions which axes are spatial.
    The spatial axes must be divisible by the corresponding pooling factor,
    otherwise the computation might crash later.

    Parameters
    ----------
    x: T.Tensor
        Input tensor (b, f, x, y), (b, z, f, x, y).
    pool: tuple
        2/3-tuple of pooling factors.
        They refer to the spatial axes of ``x`` (x,y)/(z,x,y).
    spatial_axes: tuple
    mode: str
        Can be any of the modes supported by Theano's dnn_pool():
        ('max', 'average_inc_pad', 'average_exc_pad', 'sum').

        'max' (default): max-pooling
        'average' or 'average_inc_pad': average-pooling
        'sum': sum-pooling
    stride: tuple

    Returns
    -------
    T.Tensor
        ``x`` with maxpooling applied. The spatial axes are decreased by the
        corresponding pooling factors
    """  # TODO: Some params are undocumented.
    if np.all(np.equal(pool, 1)): # Short circuit no pool
        return x

    pool = tuple(pool)
    if stride is None:
        stride = pool
    else:
        stride = tuple(stride)

    spatial_axes = list(spatial_axes)
    if not dnn_avail and config.use_manual_cudnn_pool and mode!='max':
        logger.warning("Pooling modes can only be selected if cuDNN is available, mode is ignored")

    if spatial_axes==[2,3,4]:
        axis_order = 'dnn'
    elif spatial_axes==[1,3,4]:
        axis_order = 'theano'
    else:
        axis_order = None

    if mode == 'average':
        mode = 'average_inc_pad'  # Theano's internal name. 'average' is deprecated.

    ndim = len(pool)
    if ndim==3:
        if not axis_order:
            raise ValueError("Axis order not recognised, must be [2,3,4] (dnn) or [1,3,4] (theano).")
        if dnn_avail and config.use_manual_cudnn_pool:
            pad = (0,0,0)
            if axis_order=='dnn':
                logger.debug("Using dnn 3dpool")
                y = dnn.dnn_pool(x, pool, stride=stride, pad=pad, mode=mode) # (b, f, x, y, z)
            else:
                logger.warning("cuDNN is available but the used axis order is "
                "for theano (z before features). This requires possibly "
                "inefficient dimshuffles, consider using cuDNN axis order. "
                "Using dnn 3pool")

                x = x.dimshuffle(0,2,1,3,4)
                y = dnn.dnn_pool(x, pool, stride=stride, pad=pad, mode=mode) # (b, f, x, y, z)
                y = y.dimshuffle(0,2,1,3,4)

        else: # fallback to theano
            if pool != stride:
                raise NotImplementedError("Stride!=Pool using theano 3d pooling (dnn pooling will work)")

            if axis_order=='theano':
                logger.debug("Using theano 3dpool")
                y = pool_2d(x, pool[1:], st=stride[1:], ignore_border=True) # (b, z, f, x, y)
                m = y[:,0::pool[0]]
                for z in range(1,pool[0]): ### TODO obey stride
                    t = y[:, z::pool[0]]
                    m = T.maximum(t, m) # (b, z, f, x, y)
                y = m
            else:
                logger.debug("Using theano 3dpool")
                y = pool_2d(x, pool[1:], st=stride[1:], ignore_border=True) # (b, z, f, x, y)
                m = y[:, :, 0::pool[0]]
                for z in range(1,pool[0]): ### TODO obey stride
                    t = y[:, :, z::pool[0]]
                    m = T.maximum(t, m)

                y = m

    elif ndim==2:
        if spatial_axes!=[2,3]:
            raise NotImplemented("Can only pool on last axes [2,3], this input hat spatial axes %s" %(spatial_axes,))
        if dnn_avail and config.use_manual_cudnn_pool:
            y = dnn.dnn_pool(x, pool, stride=stride, pad=(0,0), mode=mode)
        else:
            y = pool_2d(x, pool, st=stride, ignore_border=True)

    elif ndim==1:
        x = x.dimshuffle(0,1,2,'x')
        pool = [pool[0], 1]
        stride = [stride[0], 1]
        y = pool_2d(x, pool, st=stride ,ignore_border=True)[:,:,:,0]
    else:
        raise NotImplementedError("Only 1/2/3-dim maxpooling with this function.")

    return y


def fragmentpool(conv_out, pool, offsets, strides, spatial_axes, mode='max'):
    if np.all(np.equal(pool, 1)): # Short circuit no pool
        return conv_out, offsets, strides

    spatial_axes = list(spatial_axes)
    result = []
    offsets_new = []
    offsets = np.array(offsets, np.int)
    strides = np.array(strides, np.int)
    sh = conv_out.shape
    _pool = np.ones(conv_out.ndim, np.int)
    _pool[spatial_axes] = pool
    ###TODO maybe unify this loop and the pooling loop for speedup... timing!
    for i,ix in enumerate(product(*map(lambda x: list(np.arange(x)), _pool))):
        sl = tuple([slice(x[0], x[0] + x[1] - x[2] + 1) for x in zip(ix, sh, _pool)])
        result.append(pooling(conv_out[sl], pool, spatial_axes, mode=mode))
        for p in offsets:
            new = p.copy()
            ix_spatial = [ix[ax] for ax in spatial_axes]
            new += np.multiply(ix_spatial, strides)
            offsets_new.append(new)

    result = T.concatenate(result, axis=0)
    offsets_new = np.array(offsets_new)
    strides_new = np.multiply(pool, strides)

    return result, offsets_new, strides_new


def fragments2dense(fragments, offsets, strides, spatial_axes):
    spatial_axes = list(spatial_axes)
    example_stride = np.prod(strides) # This strides is conceptually unneeded but theano-grad fails otherwise
    sh             = fragments.shape
    #spatial_axes   = [1,3,4] if len(strides)==3 else [2,3]
    out_sh         = list(sh)
    out_sh[0]      = 1
    for i,ax in enumerate(spatial_axes):
        out_sh[ax] *= strides[i]

    # assert np.prod(out_sh)==np.prod(sh) # cannot work on symbolical....
    zero = np.array((0,), dtype=theano.config.floatX)
    embedding = T.alloc(zero, *out_sh)
    for i,off in enumerate(offsets):
        sl = [slice(None),]*len(out_sh) # defaults to ":" slice
        for k,ax in enumerate(spatial_axes):
            sl[ax] = slice(off[k], None, strides[k])
        embedding = T.set_subtensor(embedding[tuple(sl)],
                                    fragments[i::example_stride])

    return embedding


def upsampling_nd(x, pool):
    new_sh = [xi*pi for xi, pi in zip(x.shape, pool)]
    zero = T.cast(0, x.dtype)
    out = T.alloc(zero, *new_sh) # T.zeros is slower
    for i,ix in enumerate(product(*map(lambda s: list(np.arange(s)), pool))):
        sl = tuple([slice(s[0], s[1], s[2]) for s in zip(ix, out.shape, pool)])
        out = T.set_subtensor(out[sl], x)

    return out


def upsampling(x, pool, spatial_axes):
    """
    Upsamling through repetition: s_new = s*p.

    e.g for pool=3: aaabbbccc...

    Parameters
    ----------
    x: T.Tensor
        Input tensor.
    pool: int
        Upsampling factor.
    spatial_axes: list
        List of axes on which to perform upsampling.

    Returns
    -------
    T.Tensor
        ``x`` with upsampling applied.
    """
    """
    Upsamling through repetition: s_new = s*p
    e.g for p=3: aaabbbccc...
    :param x:
    :param pool:
    :param spatial_axes:
    :return:
    """
    spatial_axes = list(spatial_axes)
    _pool = np.ones(x.ndim, np.int)
    _pool[spatial_axes] = pool
    return upsampling_nd(x, _pool)


def unpooling_nd(x, pool):
    new_sh = [xi*pi + (pi-1) for xi, pi in zip(x.shape, pool)]
    zero = T.cast(0, x.dtype)
    out = T.alloc(zero, *new_sh) # T.zeros is slower
    sl = tuple([slice(pi-1, xi*pi, pi) for xi, pi in zip(x.shape, pool)])
    out = T.set_subtensor(out[sl], x)

    return out


def unpooling(x, pool, spatial_axes):
    """
    Symmetric unpooling with border: s_new = s*pool + pool-1.

    Insert values strided, e.g for pool=3: 00x00x00x...00x00.

    Parameters
    ----------
    x: T.Tensor
        Input tensor.
    pool: int
        Unpooling factor.
    spatial_axes: list
        List of axes on which to perform unpooling.

    Returns
    -------
    T.Tensor
        ``x`` with unpooling applied.
    """
    spatial_axes = list(spatial_axes)
    _pool = np.ones(x.ndim, np.int)
    _pool[spatial_axes] = pool
    return unpooling_nd(x, _pool)
