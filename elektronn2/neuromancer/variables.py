# -*- coding: utf-8 -*-
# ELEKTRONN2 Toolkit
# Copyright (c) 2015 Marius F. Killinger
# All rights reserved

from __future__ import absolute_import, division, print_function
from builtins import filter, hex, input, int, map, next, oct, pow, range, super, zip

import copy
import logging

import numpy as np
import theano.tensor as T
from theano.tensor.sharedvar import TensorSharedVariable
from theano.tensor import TensorConstant, TensorType

from .. import config
from .graphutils import floatX, as_floatX


logger = logging.getLogger('elektronn2log')

__all__ = ['VariableParam', 'VariableWeight', 'ConstantParam', 'initweights']

class VariableParam(TensorSharedVariable):
    """
    Extension of theano ``TensorSharedVariable``. Additional features are
    described by the parameters, otherwise identical

    Parameters
    ----------
    value
    name: str
    apply_reg flag: bool
        whether to apply regularisation (e.g. L2) on this param
    apply_train flag: bool
        whether to train this parameter (as opposed to a meta-parameter or
        a parameter that is kept const. during a training phase)
    dtype:
    strict: bool
    allow_downcast: bool
    borrow: bool
    broadcastable
    """

    def __init__(self, value=None, name=None, apply_train=True, apply_reg=True,
                 dtype=None, strict=False, allow_downcast=None, borrow=False,
                 broadcastable=None):
        self.apply_reg   = apply_reg
        self.apply_train = apply_train
        self._updates     = None
        self.constant    = False
        if not apply_train:
            name += "_noTrain"

        if isinstance(value, (int, float)):
            value = np.array(value, dtype=dtype)

        if dtype is not None:
            value = value.astype(dtype)

        value=np.array(value, copy=(not borrow))

        if broadcastable is None:
            broadcastable = (False,) * len(value.shape)
        t_type = T.TensorType(value.dtype, broadcastable=broadcastable)

        super(VariableParam, self).__init__(type=t_type, value=value, name=name,
                                          strict=strict,
                                          allow_downcast=allow_downcast)

    @property
    def updates(self):
        return self._updates

    @updates.setter
    def updates(self, up):
        if self.apply_train or self.apply_reg:
            raise ValueError("Cannot register extra updates for trainable "
                             "parameter %s" %(repr(self),))
        self._updates = up

    def clone(self):
        cp = TensorSharedVariable(
            name=self.name,
            type=self.type,
            value=None,
            strict=None,
            container=self.container)
        cp.tag = copy.copy(self.tag)
        return cp


class VariableWeight(VariableParam):
    def __init__(self, shape=None, init_kwargs=None, value=None, name=None,
                 apply_train=True, apply_reg=True, dtype=None, strict=False,
                 allow_downcast=None, borrow=False, broadcastable=None):
        """
        Extension of theano ``TensorSharedVariable`` and subclass of ``VariableParam``.
        Additional features are described by the parameters, otherwise identical

        Parameters
        ----------
        shape: list/tuple of int
            Shape of weights (if value=None)
        init_kwargs: dict
            kwargs for the ``initweights``-function (if value=None)
        value: numpy array
            initial value (if shape/init_kwargs=None)
        name: str
        apply_train flag: bool
            Whether to train this parameter (as opposed to a meta-parameter or
            a parameter that is kept const. during a training phase)
        apply_reg flag: bool
            Whether to apply regularisation (e.g. L2) on this param
        dtype
        strict: bool
        allow_downcast: bool
        borrow: bool
        broadcastable
        """
        if value is None: # create new values
            if (shape is None) or (init_kwargs is None):
                raise ValueError("shape and init_kwargs are required if value is None")
            value = initweights(shape, **init_kwargs)

        elif shape is not None:
            if np.array(value).ndim > 1:
                raise ValueError("If value and shape are specified, "
                                 "value must be scalar.")
            value = np.ones(shape) * value

        super(VariableWeight, self).__init__(value, name, apply_train, apply_reg,
                                           dtype, strict, allow_downcast,
                                           borrow, broadcastable)

    def set_value(self, new_value, borrow=False):
        sh = self.get_value().shape
        if isinstance(new_value, np.ndarray):
            if not (sh == new_value.shape):
                raise NotImplementedError("given shape: %s, required shape: %s "
                "Crop value or extend with similar numbers"%(new_value.shape, sh))
        elif isinstance(new_value, (float, int)):
            pass
        else:
            raise ValueError("Value/type not understood")

        try:
            super(VariableWeight, self).set_value(new_value, borrow)
        except TypeError as e:
            if config.allow_floatX_downcast:
                new_value = as_floatX(new_value)
                super(VariableWeight, self).set_value(new_value, borrow)
            else:
                raise

class ConstantParam(TensorConstant):
    """
    Identical to theano ``VariableParam`` except that there are two
    two addition attributes ``apply_train`` and `apply_reg``, which are
    both false.
    This is just to tell ELEKTRONN2 that this parameter is to be
    exempted from training. Obviously the ``set_value`` method raises
    an exception because this is a real constant. Constants are faster
    in the theano graph.
    """

    def __init__(self, value, name=None, dtype=None,
                 make_singletons_broadcastable=True):
        name += "_const"

        if isinstance(value, (int, float)):
            value = np.array(value, dtype=dtype)

        if dtype is not None:
            value = value.astype(dtype)
        if make_singletons_broadcastable:
            broadcastable = [d == 1 for d in value.shape]
        else:
            broadcastable = [False for d in value.shape]

        dtype_t = TensorType(dtype=value.dtype, broadcastable=broadcastable)

        self.apply_train = False
        self.apply_reg   = False
        self.constant    = True

        super(ConstantParam, self).__init__(dtype_t, value, name=name)

    def clone(self):
        return TensorConstant(self.type, self.data, self.name)

    def set_value(self, new_value, borrow=False):
        raise RuntimeError("Cannot set value for ConstantParam")


    def get_value(self, borrow=False):
        return self.value

    @property
    def updates(self):
        return None


def initweights(shape, dtype=floatX, scale='glorot', mode='normal', pool=None, spatial_axes=None):
    if mode=='const':
        W = np.ones(shape) * scale

    elif mode=='prelu':
        # assuming shape is (n_out, 2)
        W = np.ones(shape) * scale
        W[:,1] = 1.0
        # GradNet inspired initial full linearity, conventional relu would be 0

    elif mode=='fix-uni':
        W = np.random.uniform(-scale, scale, shape)

    elif scale=='glorot':
        if len(shape)==2: # (fin, nof)
            n_in, n_out = shape[0], shape[1]
            s = n_in + n_out
        else:
            assert spatial_axes is not None
            other, kernel = [], []
            for i,s in enumerate(shape):
                if i in spatial_axes:
                    kernel.append(s)
                else:
                    other.append(s)

            assert len(other)==2
            n_out = other[0]
            n_in  = other[1]
            fov = np.prod(kernel)
            ps  = np.prod(pool)
            s = (n_in + float(n_out)/ps) * fov

        W_scale = np.sqrt(2.0 / s)

        if mode=='normal':
            W = np.random.normal(0, W_scale, shape)
        elif mode=='uni':
            W = np.random.uniform(-W_scale, W_scale, shape)
        elif mode=='ortho':
            M = np.random.normal(0, W_scale, size=shape)
            M = M.reshape((n_out, -1))
            # more vectors needed than can be orthogonal in this dimension
            strip_required = False
            n_in = M.shape[1]
            if n_out > n_in:
                M = np.random.normal(0, W_scale, size=(n_out, n_out))
                strip_required = True

            U, S, V = np.linalg.svd(M, full_matrices=False)
            W = V / V.std(1)[:,None] * W_scale
            #W -= W.mean(axis=1)[:,None] # This changes whether they are orthogonal!
            if strip_required:
                W = W[:, :n_in]

            W = W.reshape(shape)
    else:
        raise ValueError("Invalid weigh initialisation parameters")

    logger.debug("Init: shape=%s, mean=%f, std=%f"%(shape, W.mean(), W.std()))

    return np.ascontiguousarray(W, dtype=dtype)
