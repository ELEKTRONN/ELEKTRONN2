# -*- coding: utf-8 -*-
# ELEKTRONN2 Toolkit
# Copyright (c) 2015 Marius Killinger and Philipp J. Schubert
# All rights reserved

from __future__ import absolute_import, division, print_function
from builtins import filter, hex, input, int, map, next, oct, pow, range, super, zip

import logging
import time
from functools import reduce

import numpy as np
import theano
import theano.tensor as T

from ..config import config
from . import computations
from .variables import VariableWeight, ConstantParam, VariableParam
from .graphutils import floatX, TaggedShape, as_floatX
from .node_basic import Node, Concat

logger = logging.getLogger('elektronn2log')

__all__ = ['Perceptron', 'Conv', 'UpConv', 'Crop', 'LSTM',
           'FragmentsToDense', 'Pool', 'Dot', 'FaithlessMerge',
           'GRU', 'LRN', 'ImageAlign', 'UpConvMerge']

################################################################################

### TODO gradnet stuff anpassen von Conv layer in den anderen layers? Vlt nicht jetzt...

class NeuralLayer(Node):
    """
    Dummy class to add parameter initialisation methods for neural layers.
    """
    def _register_param(self, param, shape, name, init_kwargs=None,
                       apply_train=False, apply_reg=False):
        """
        Create parameter, set parameter as attribute and add to self.params if
        not shared from another Layer.

        Parameters
        ----------
        param: None or np.ndarray or T.Variable or list
            Possible forms of ``param``:
            * Passing ``None`` creates new parameter with default
              initialisation.
            * Passing a np.ndarray creates new parameter with the values
              of the array as initialisation.
            * A shared parameter is created by passing a T.Variable as
              ``param``.
            * A constant parameter is created by passing [np.ndarray, 'const']
              as ``param``.
              This parameter cannot be changed (no set_value) but makes the
              compiled function faster.
        shape: tuple
            Shape of the new the parameter (VariableWeight).
        name: str
            Parameter name.
        init_kwargs
            kwargs for utils.initialisation.
        apply_train: bool
            Train flag of the new parameter (VariableWeight).
        apply_reg: bool
            Regularisation flag of the new parameter (VariableWeight).
        """
        add_to_params = True
        if self.name=='':
            p_name = '<%s%s>'%(name, tuple(shape))
        else:
            p_name = '<%s_%s%s>'%(self.name, name, tuple(shape))
        # create new trainable by initialistaion
        if param is None:
            p = VariableWeight(shape=shape,
                               init_kwargs=init_kwargs,
                               name=p_name,
                               apply_train=apply_train,
                               apply_reg=apply_reg,)

        # create new trainable from values
        elif isinstance(param, np.ndarray):
            if param.shape!=tuple(shape):
                if not (param.ndim==0 and shape==(1,)):
                    raise ValueError("Shape mismatch. Required %s, given %s"\
                                 %(shape, param.shape))
            p = VariableWeight(value=param,
                               name=p_name,
                               apply_train=apply_train,
                               apply_reg=apply_reg,
                               dtype=floatX)

        # share a variable from elsewhere, not trainable
        elif isinstance(param, T.Variable): # (elektronn2.tensor.variables are T.Variable)
            try:
                sh = param.get_value().shape
                if sh!=tuple(shape):
                    raise ValueError("Shape mismatch. Required %s, given %s" \
                                     % (shape, param.shape))
            except AttributeError:
                logger.warning("Could not check correct shape of given weight %s, "
                               "make sure it has shape %s" %(param, shape))
            p = param
            add_to_params = False

        # create constant variable (or explicitly trainable
        elif isinstance(param, (list, tuple)):
            fail = False
            if not isinstance(param[0] , np.ndarray):
                fail = True
            if param[0].shape!=tuple(shape):
                raise ValueError("Shape mismatch. Required %s, given %s"\
                                 %(shape, param[0].shape))
            if param[1] == 'const':
                value = as_floatX(param[0])
                p = ConstantParam(value, p_name)
            elif param[1] == 'trainable':
                value = as_floatX(param[0])
                p = VariableWeight(value=value,
                                   name=p_name,
                                   apply_train=True,
                                   apply_reg=apply_reg)
            else:
                fail = True

            if fail:
                raise ValueError("If a parameter is passed as a list, the "
                                 "first entry must contain the parameter "
                                 "value (np.ndarray) and the second entry "
                                 "must be either 'const' or 'trainable' "
                                 "to indicate whether this param is "
                                 "trainable. Got [%s, %s]" \
                                 %(type(param[0]), param[1]))
        else:
            raise ValueError("Parameter %s must be either <np.ndarray>, "
                             "<theano.TensorVariable>, a tuple or None "
                             "(to create new param)" %(name,))

        setattr(self, name, p) #
        if add_to_params:
            self.params[name] = p
        else:
            logger.debug("Sharing theano variable %s. This parameter is not added to self.params" %(p,))


    def _setup_params(self, w_sh, w, b, gamma, mean, std, dropout_rate,
                      pool_shape=None, gradnet_rate=None):
        """
        Register each parameter, choose appropriate initialisation.
        """
        # Dot/Conv/Bias Parameters #############################################
        self.w = None

        # TODO: Pass w_init mode from layer to setup_params
        if config.use_ortho_init or isinstance(self, GRU) or isinstance(self, LSTM):
            w_init = dict(scale='glorot', mode='ortho', pool=pool_shape,
                          spatial_axes=self.spatial_axes)
        else:
            w_init = dict(scale='glorot', mode='normal', pool=pool_shape,
                          spatial_axes=self.spatial_axes)

        self._register_param(w, w_sh, 'w', init_kwargs=w_init,
                             apply_train=True, apply_reg=True)

        activation_func = self.activation_func
        n_f = self.n_f
        self.b = None
        if isinstance(self, GRU):
            b_sh=(3 * n_f, )
        elif isinstance(self, LSTM):
            b_sh = (4 * n_f, )
        else:
            b_sh=(n_f,)
        if activation_func=='relu' or activation_func.startswith("maxout"):
            norm = 1.0
            if len(w_sh) > 2:
                fov = 1
                for i in self.spatial_axes:
                    fov = fov * w_sh[i]
                norm = fov

            b_init=dict(scale=1.0/norm, mode='const')

        elif activation_func=='sigmoid':
            b_init=dict(scale=0.5, mode='const')
        elif activation_func=='prelu':
            norm = 1.0
            if len(w_sh) > 2:
                fov = 1
                for i in self.spatial_axes:
                    fov = fov * w_sh[i]
                norm = fov

            b_init=dict(scale=1.0/norm, mode='prelu')
            if isinstance(self, GRU):
                 b_sh=(3 * n_f, 2)
            elif isinstance(self, LSTM):
                b_sh = (4 * n_f, 2)
            else:
                b_sh=(n_f, 2)
        else: # all other activations
            b_init=dict(scale=1e-6, mode='fix-uni')
        self._register_param(b, b_sh, 'b', init_kwargs=b_init,
                            apply_train=True, apply_reg=False)

        # Batch Normalisation ##################################################
        batch_normalisation = self.batch_normalisation
        if batch_normalisation in ['train', 'fadeout']:
            # mean and std are created as TensorVariables in _calc_output
            self.gamma = None
            sh = (n_f,)
            g_init =dict(scale=1.0, mode='const')
            self._register_param(gamma, sh, 'gamma', init_kwargs=g_init,
                                apply_train=True, apply_reg=3.0) ###TODO maybe even stronger reg for this?
            if mean is not None or std is not None:
                raise ValueError("Cannot pass mean and std for training, they "
                                 "are computed in the theano graph.")

            # create mean and std for training to accumulate running avgs
            self.mean = None
            m_init =dict(scale=0.0, mode='const')
            self._register_param(None, sh, 'mean', init_kwargs=m_init)

            self.std = None
            s_init =dict(scale=1.0, mode='const')
            self._register_param(None, sh, 'std', init_kwargs=s_init)

        elif batch_normalisation=='predict':
            sh = (n_f,)
            self.gamma = None
            g_init =dict(scale=1.0, mode='const')
            self._register_param(gamma, sh, 'gamma', init_kwargs=g_init)

            self.mean = None
            m_init =dict(scale=0.0, mode='const')
            self._register_param(mean, sh, 'mean', init_kwargs=m_init)

            self.std = None
            s_init =dict(scale=1.0, mode='const')
            self._register_param(std, sh, 'std', init_kwargs=s_init)
        else:
            if batch_normalisation is not False:
                raise ValueError("Unknown value %s for batchnormalisation" %batch_normalisation)

        # Dropout ##############################################################
        self.dropout_rate = None
        if dropout_rate:
            value = as_floatX(dropout_rate)
            self._register_param(value, (1,), 'dropout_rate')


        # GradNet ##############################################################
        self.gradnet_rate = None
        if gradnet_rate:
            value = as_floatX(gradnet_rate)
            self._register_param(value, (1,), 'gradnet_rate')

###############################################################################

class Perceptron(NeuralLayer):
    """
    Perceptron Layer.

    Parameters
    ----------
    parent: Node or list of Node
        The input node(s).
    n_f: int
        Number of filters (nodes) in layer.
    activation_func: str
        Activation function name.
    flatten: bool
    batch_normalisation: str or None
        Batch normalisation mode.
        Can be False (inactive), "train" or "fadeout".
    dropout_rate: float
        Dropout rate (probability that a node drops out in a training step).
    name: str
        Perceptron name.
    print_repr: bool
        Whether to print the node representation upon initialisation.
    w: np.ndarray or T.TensorVariable
        Weight matrix.
        If this is a np.ndarray, its values are used to initialise a
        shared variable for this layer.
        If it is a T.TensorVariable, it is directly used (weight sharing
        with the layer which this variable comes from).
    b: np.ndarray or T.TensorVariable
        Bias vector.
        If this is a np.ndarray, its values are used to initialise a
        shared variable for this layer.
        If it is a T.TensorVariable, it is directly used (weight sharing
        with the layer which this variable comes from).
    gamma
        (For batch normalisation) Initializes gamma parameter.
    mean
        (For batch normalisation) Initializes mean parameter.
    std
        (For batch normalisation) Initializes std parameter.
    gradnet_mode
    """  # TODO: Write docs on batch normalisation modes.
    # TODO: gradnet_mode seems to be unused. Should it be removed?

    def __init__(self, parent, n_f, activation_func='relu',
                 flatten=False, batch_normalisation=False, dropout_rate=0,
                 name="dot", print_repr=True, w=None, b=None, gamma=None,
                 mean=None, std=None, gradnet_mode=None):
        super(Perceptron, self).__init__(parent, name, print_repr)

        self.n_f = n_f
        self.activation_func = activation_func
        self.batch_normalisation = batch_normalisation
        self.gradnet_mode = gradnet_mode
        self.axis = parent.shape.tag2index('f') #retrieve feature shape's index
        self.flatten = flatten
        self.spatial_axes = parent.shape.spatial_axes
        if flatten:
            n_in = parent.shape.stripbatch_prod
        else:
            n_in = parent.shape['f'] #retrieve feature shape

        w_sh = (n_in, n_f)
        self._setup_params(w_sh, w, b, gamma, mean, std, dropout_rate)


    def _make_output(self):
        """
        Computation of Theano output.
        """
        if self.flatten:
            if self.axis is not 1:
                raise NotImplementedError("Cannot flatten tensor for "
                                          "Perceptron layer when batchsize is "
                                          "not on first axis")
            input_tensor = self.parent.output.flatten(2)
            pattern  = ['x', 0]
        else:
            input_tensor = self.parent.output
            pattern  = ['x' for i in input_tensor.shape]
            pattern[self.axis] = 0

        activation_func = self.activation_func
        if activation_func.startswith("maxout"):
            r=int(activation_func.split(" ")[1])
            assert r>=2
            self.n_f /= r

        if activation_func=='prelu':
            b  = self.b[:,0].dimshuffle(pattern)
            b1 = self.b[:,1].dimshuffle(pattern)
        else:
            b   = self.b.dimshuffle(pattern)
            b1  = None

        lin_output = computations.dot(input_tensor, self.w, self.axis)

        if self.batch_normalisation in ['train', 'fadeout']:
            mean = computations.apply_except_axis(
                lin_output,self.axis, T.mean).dimshuffle(pattern)
            std = computations.apply_except_axis(
                lin_output,self.axis, T.std).dimshuffle(pattern) + 1e-6
            gamma = self.gamma.dimshuffle(pattern)

            if self.batch_normalisation=='fadeout':
                logger.warning("Batch normalisation mode 'fadeout' does not "
                               "work for less than 50%...")
                mean = self.gradnet_rate * mean
                std  = self.gradnet_rate * std + (1-self.gradnet_rate) * 1.0
                gamma = self.gradnet_rate * gamma

            self.mean.updates = (self.mean,
                                 0.9995 * self.mean + 0.0005 * T.extra_ops.squeeze(mean))
            self.std.updates  = (self.std,
                                 0.9995 * self.std + 0.0005 * T.extra_ops.squeeze(std))

        elif self.batch_normalisation=='predict':
            mean = self.mean.dimshuffle(pattern)
            std = self.std.dimshuffle(pattern)
            gamma = self.gamma.dimshuffle(pattern)
        else:
            mean = 0
            std = 1
            gamma = 1




        lin_output =  (gamma / std) * lin_output + b - (gamma * mean / std)
        lin_output = computations.apply_activation(lin_output, activation_func, b1)

        if self.dropout_rate:
            rng = T.shared_randomstreams.RandomStreams(int(time.time()))
            p   = 1 - self.dropout_rate
            dropout_gate = rng.binomial(size=(self.n_f,), n=1, p=p,
                                        dtype=theano.config.floatX)
            dropout_gate *= 1.0 / p
            lin_output =  lin_output * dropout_gate.dimshuffle(pattern)

        self.output = lin_output


    def _calc_shape(self):
        """
        Calculate shape from parent shape and n_f and set it as self.shape.
        """
        sh = self.parent.shape
        if self.flatten:
            self.shape = TaggedShape((sh['b'], self.n_f), 'b,f')
        else:
            self.shape = sh.updateshape('f', self.n_f)


    def _calc_comp_cost(self):
        """
        Calculate abstract computational cost from parent shape and n_f and
        set it as self.computational_cost.
        """
        n = self.parent.shape.stripnone_prod
        self.computational_cost = n * self.n_f


    def make_dual(self, parent, share_w=False, **kwargs):
        """
        Create the inverse of this ``Perceptron``.

        Most options are the same as for the layer itself.
        If ``kwargs`` are not specified, the values of the primal
        layers are re-used and new parameters are created.

        Parameters
        ----------
        parent: Node
            The input node.
        share_w: bool
            If the weights (``w``) should be shared from the primal layer.
        kwargs: dict
            kwargs that are passed through to the constructor of the inverted
            Perceptron (see signature of ``Perceptron``).
            ``n_f`` is copied from the existing node on
            which ``make_dual`` is called.
            Every other parameter can be changed from the original
            ``Perceptron``'s defaults by specifying it in ``kwargs``.

        Returns
        -------
        Perceptron
            The inverted perceptron layer.
        """
        if self.flatten:
            raise NotImplementedError("Cannot make dual Layer for flattened "
                                      "Perceptron layer.")

        dropout_rate = 0.0 if not self.dropout_rate else self.dropout_rate.get_value()
        defaults = dict(activation_func=self.activation_func,
                        batch_normalisation=self.batch_normalisation,
                        dropout_rate=dropout_rate,
                        name=self.name+'.T',
                        print_repr=self._print_repr,
                        w=None, b=None,gamma=None, mean=None, std=None)

        defaults.update(kwargs)
        kwargs = defaults

        if share_w:
            if kwargs['w'] is not None:
                logger.debug("Ignoring passed w because w is shared from primal Layer.")
            kwargs['w'] = self.w.T

        n_f = self.parent.shape['f'] # This is the output of the dual Layer

        if self.n_f is not parent.shape['f']: # input of dual Layer  #q: Shouldn't this be "!=", instead of "is not"?
            raise ValueError("Cannot make dual layer of:\n"
                             "%s\n"
                             "with input: %s!\n"
                             "The output shape of the input for the dual layer "
                             "must match the the input shape of the primal layer."\
                             %(self, parent))

        return Perceptron(parent, n_f, **kwargs)


    def __repr__(self):
        s = super(NeuralLayer, self).__repr__()
        s += "\n"
        s += "  n_f=%i, " %(self.n_f,)
        s += "act='%s', " %(self.activation_func,)
        if self.flatten:
            s += "input was flattened, "
        if self.dropout_rate:
            s += "dropout rate = %.1f, "%(self.dropout_rate.get_value())
        if self.batch_normalisation:
            s += "BN in '%s' mode "%(self.batch_normalisation,)

        return s

Dot = Perceptron

###############################################################################

class Conv(Perceptron):
    """
    Convolutional layer with subsequent pooling.

    Examples
    --------
    Examples for constructing convolutional neural networks can be found
    in examples/neuro3d.py and examples/mnist.py.

    Parameters
    ----------
    parent: Node
        The input node.
    n_f: int
        Number of features.
    filter_shape: tuple
        Shape of the convolution filter kernels.
    pool_shape: tuple
        Size/shape of pooling after the convolution.
    conv_mode: str
        Possible values:
        * "valid": only apply filter to complete patches of the image.
          Generates output of shape: image_shape -filter_shape + 1.
        * "full" zero-pads image to multiple of filter shape to generate
          output of shape: image_shape + filter_shape - 1.
    activation_func: str
        Activation function name.
    mfp: bool
        Whether to apply Max-Fragment-Pooling in this Layer.
    batch_normalisation: str or None
        Batch normalisation mode.
        Can be False (inactive), "train" or "fadeout".
    dropout_rate: float
        Dropout rate (probability that a node drops out in a training step).
    name: str
        Layer name.
    print_repr: bool
        Whether to print the node representation upon initialisation.
    w: np.ndarray or T.TensorVariable
        Weight matrix.
        If this is a np.ndarray, its values are used to initialise a
        shared variable for this layer.
        If it is a T.TensorVariable, it is directly used (weight sharing
        with the layer which this variable comes from).
    b: np.ndarray or T.TensorVariable
        Bias vector.
        If this is a np.ndarray, its values are used to initialise a
        shared variable for this layer.
        If it is a T.TensorVariable, it is directly used (weight sharing
        with the layer which this variable comes from).
    gamma
        (For batch normalisation) Initializes gamma parameter.
    mean
        (For batch normalisation) Initializes mean parameter.
    std
        (For batch normalisation) Initializes std parameter.
    gradnet_mode
    """

    def __init__(self, parent, n_f, filter_shape, pool_shape,
                 conv_mode='valid', activation_func='relu',
                 mfp=False, batch_normalisation=False, dropout_rate=0,
                 name="conv", print_repr=True, w=None, b=None, gamma=None,
                 mean=None, std=None, gradnet_mode=None):
        super(Perceptron, self).__init__(parent, name, print_repr)

        self.n_f  = n_f
        self.filter_shape = filter_shape
        self.pool_shape = pool_shape
        self.conv_mode = conv_mode
        self.activation_func = activation_func
        self.batch_normalisation = batch_normalisation
        self.gradnet_mode = gradnet_mode
        self.mfp = mfp

        self.strides = parent.shape.strides
        self.mfp_offsets = parent.shape.mfp_offsets
        self.axis = parent.shape.tag2index('f') #retrieve feature shape's index
        self.axis_order = None

        self.spatial_axes = self.parent.shape.spatial_axes
        conv_dim = len(self.spatial_axes)
        x_dim    = len(self.parent.shape)
        if len(self.spatial_axes)!=len(filter_shape) or \
                        len(filter_shape)!=len(pool_shape):
            raise ValueError("The filter_shape dimensionality (%i), the number "
                             "of spatial dimensions in the input (%i) and "
                             "the dimensionality of pool_shape (%i) differ! "
                             "Use filter size 1 on axes which should not be "
                             "convolved."\
                             %(len(filter_shape), conv_dim, len(pool_shape)))

        n_in = parent.shape['f'] #retrieve feature shape
        fail = False
        if conv_dim==1:
            if x_dim!=3 or self.spatial_axes!=[2]:
                fail = True
            w_sh = [n_f, n_in] + list(filter_shape)

        elif conv_dim==2:
            if x_dim!=4 or self.spatial_axes!=[2,3]:
                fail = True
            w_sh = [n_f, n_in] + list(filter_shape)

        elif conv_dim==3:
            if x_dim!=5:
                fail = True
            if self.spatial_axes==[2,3,4]:
                self.axis_order = 'dnn'
                w_sh = [n_f, n_in] + list(filter_shape)
            elif self.spatial_axes==[1,3,4]:
                self.axis_order = 'theano'
                w_sh = [n_f, filter_shape[0], n_in] + list(filter_shape[1:])
            else:
                fail = True

        if fail:
            raise NotImplementedError("Cannot convolve non-standard shapes / axis orders. "
                                      "Implement reshaping before conv "
                                      "and re-reshaping after!")

        self.conv_dim = conv_dim
        self.w_sh = w_sh

        gradnet_rate = 1.0 if gradnet_mode else None

        self._setup_params(w_sh, w, b, gamma, mean, std, dropout_rate,
                           pool_shape, gradnet_rate)


    def _make_output(self):
        """
        Computation of Theano output.
        """
        input_tensor = self.parent.output
        input_shape  = list(self.parent.shape)
        pattern  = ['x' for i in input_tensor.shape]
        pattern[self.axis] = 0
        activation_func = self.activation_func
        if activation_func.startswith("maxout"):
            r=int(activation_func.split(" ")[1])
            assert r>=2
            self.filter_shape /= r

        if activation_func=='prelu':
            b  = self.b[:,0].dimshuffle(pattern)
            b1 = self.b[:,1].dimshuffle(pattern)
        else:
            b   = self.b.dimshuffle(pattern)
            b1  = None

        lin_output = computations.conv(input_tensor, self.w, self.axis_order,
                                       border_mode=self.conv_mode,
                                       x_shape=input_shape, w_shape=self.w_sh)

        if self.mfp:
            if self.input_nodes[0].shape['b']!=1:
                raise ValueError("For MFP the batchsize of the raw image input must be 1.")

            lin_output, offsets_new, strides_new = computations.fragmentpool(lin_output,
                                                                        self.pool_shape,
                                                                        self.mfp_offsets,
                                                                        self.strides,
                                                                        self.spatial_axes)
            self.mfp_offsets = offsets_new
            self.strides = strides_new
        else:
            lin_output = computations.pooling(lin_output, self.pool_shape, self.spatial_axes)
            self.strides = np.multiply(self.pool_shape, self.strides)

        if self.batch_normalisation in ['train', 'fadeout']:
            mean = computations.apply_except_axis(
                lin_output,self.axis, T.mean).dimshuffle(pattern)
            std = computations.apply_except_axis(
                lin_output,self.axis, T.std).dimshuffle(pattern) + 1e-6
            gamma = self.gamma.dimshuffle(pattern)

            if self.batch_normalisation=='fadeout':
                logger.warning("Batch Normalisation mode 'fadeout' does not "
                               "work for less than 50%...")
                mean = self.gradnet_rate * mean
                std  = self.gradnet_rate * std + (1-self.gradnet_rate) * 1.0
                gamma = self.gradnet_rate * gamma

            self.mean.updates = (self.mean,
                                 0.9995 * self.mean + 0.0005 * T.extra_ops.squeeze(mean))
            self.std.updates  = (self.std,
                                 0.9995 * self.std + 0.0005 * T.extra_ops.squeeze(std))

        elif self.batch_normalisation=='predict':
            mean = self.mean.dimshuffle(pattern)
            std = self.std.dimshuffle(pattern)
            gamma = self.gamma.dimshuffle(pattern)
        else:
            mean = 0
            std = 1
            gamma = 1



        lin_output =  (gamma / std) * lin_output + b - (gamma * mean / std)
        lin_output = computations.apply_activation(lin_output, activation_func, b1)

        if self.dropout_rate:
            rng = T.shared_randomstreams.RandomStreams(int(time.time()))
            p   = 1 - self.dropout_rate
            dropout_gate = rng.binomial(size=lin_output.shape, n=1, p=p,
                                        dtype=theano.config.floatX)
            dropout_gate *= 1.0 / p
            lin_output *= dropout_gate  #.dimshuffle(('x', 0))

        self.output = lin_output


    def _calc_shape(self):
        """
        Calculate and set self.shape.
        """
        sh = self.parent.shape
        for j,(i,f,p) in enumerate(zip(self.spatial_axes, self.filter_shape, self.pool_shape)):
            if self.conv_mode=='valid':
                k = 1 - f
            elif self.conv_mode=='full':
                k = f - 1
            elif self.conv_mode=='same':
                k = 0
            s = (sh[i] + k)//p
            if self.mfp:
                if (sh[i] + k - p + 1)%p!=0:
                    raise ValueError("Cannot pool spatial axis '%s' of length %i "
                                     "by factor %i after convolving with "
                                     "kernel of size %i and using MFP."\
                                     %(sh.tags[i], sh[i], p, f))
            else:
                if (sh[i] + k)%p!=0:
                    raise ValueError("Cannot pool spatial axis '%s' of length %i "
                                     "by factor %i after convolving with "
                                     "kernel of size %i."\
                                     %(sh.tags[i], sh[i], p, f))
            sh = sh.updateshape(i, s)
            if sh.fov[j]>0:
                fov = sh.fov[j] + (f+p-2) * sh.strides[j]
            else:
                fov = -1
            sh = sh.updatefov(j, fov)

        if self.mfp:
            sh = sh.updatemfp_offsets(self.mfp_offsets)
            sh = sh.updateshape('b', np.prod(self.pool_shape), mode='mult')

        sh = sh.updatestrides(self.strides)
        sh = sh.updateshape('f', self.n_f)
        self.shape = sh


    def _calc_comp_cost(self):
        """
        Calculate and set self.computational_cost.
        """
        sh = self.parent.shape
        n_position = 1
        for i,f,p in zip(self.spatial_axes, self.filter_shape, self.pool_shape):
            s = 1 - f if self.conv_mode=='valid' else f -1
            n_position *= sh[i] + s

        b = 1 if sh['b'] is None else sh['b']
        self.computational_cost = np.product(self.w_sh) * n_position * b


    def make_dual(self, parent, share_w=False, mfp=False, **kwargs):
        """
        Create the inverse (``UpConv``) of this ``Conv`` node.

        Most options are the same as for the layer itself.
        If ``kwargs`` are not specified, the values of the primal
        layers are re-used and new parameters are created.

        Parameters
        ----------
        parent: Node
            The input node.
        share_w: bool
            If the weights (``w``) should be shared from the primal layer.
        mfp: bool
            If max-fragment-pooling is used.
        kwargs: dict
            kwargs that are passed through to the new ``UpConv`` node (see
            signature of ``UpConv``).
            ``n_f`` and ``pool_shape`` are copied from the existing node on
            which ``make_dual`` is called.
            Every other parameter can be changed from the original
            ``Conv``'s defaults by specifying it in ``kwargs``.

        Returns
        -------
        UpConv
            The inverted conv layer (as an ``UpConv`` node).
        """
        if mfp:
            parent = FragmentsToDense(parent, print_repr=False)

        dropout_rate = 0.0 if not self.dropout_rate else self.dropout_rate.get_value()
        defaults = dict(conv_mode='valid', activation_func=self.activation_func,
                        batch_normalisation=self.batch_normalisation,
                        dropout_rate=dropout_rate,
                        name=self.name+'.T',
                        print_repr=self._print_repr,
                        w=None, b=None,gamma=None, mean=None, std=None)

        defaults.update(kwargs)
        kwargs = defaults

        if share_w:
            if kwargs['w'] is not None:
                logger.debug("Ignoring passed w because w is shared from primal Layer.")
            w = self.w
            # Exchange n_in and n_f
            swap = (0,2) if (self.conv_dim==3 and self.axis_order=='theano') else (0,1)
            w = T.swapaxes(w, *swap)
            kwargs['w'] = w

        n_f = self.parent.shape['f'] # This is the output of the dual Layer

        if self.w_sh[0] is not parent.shape['f']: # input of dual Layer
            raise ValueError("Cannot make dual layer of:\n"
                             "%s\n"
                             "with input: %s!\n"
                             "The output shape of the input for the dual layer "
                             "must match the the input shape of the primal layer."\
                             %(self, parent))

        return UpConv(parent, n_f, self.pool_shape, **kwargs)

    def __repr__(self):
        s = super(NeuralLayer, self).__repr__()
        s += "\n"
        s += "  n_f=%i, " %(self.n_f,)
        s += "%id conv, kernel=%s, pool=%s, "\
             %(self.conv_dim, self.filter_shape, self.pool_shape)
        s += "act='%s', " %(self.activation_func,)
        if self.dropout_rate:
            s += "Dropout rate=%.1f, "%(self.dropout_rate.get_value())
        if self.batch_normalisation:
            s += "BN in '%s' mode "%(self.batch_normalisation,)
        if self.mfp:
            s += "MFP active, "

        return s

###############################################################################

class FragmentsToDense(Node):
    def __init__(self, parent, name="to_dense", print_repr=True):
        super(FragmentsToDense, self).__init__(parent, name, print_repr)

    def _make_output(self):
        """
        Computation of Theano output.
        """
        fragments  = self.parent.output
        sh = self.parent.shape
        if sh['b']!=len(sh.mfp_offsets) or sh['b']!=np.prod(sh.strides):
            raise ValueError("Need %i fragments on the batch axis. "
                             "Is MFP active at all?" %np.prod(sh.strides))

        self.output = computations.fragments2dense(fragments, sh.mfp_offsets,
                                              sh.strides, sh.spatial_axes)

    def _calc_shape(self):
        """
        Calculate and set self.shape.
        """
        sh = self.parent.shape
        for ax, st in zip(sh.spatial_axes, sh.strides):
            sh = sh.updateshape(ax, st, mode='mult')

        sh = sh.updateshape('b', 1)
        new_strides = np.ones(len(sh.spatial_axes), np.int)
        new_offsets = np.zeros((1,len(sh.spatial_axes)), np.int)
        self.shape = TaggedShape(sh.shape, sh.tags, new_strides,
                                 new_offsets, sh.fov)


    def _calc_comp_cost(self):
        """
        Calculate and set self.computational_cost.

        For this Node type this is hard-coded to 0.
        """
        self.computational_cost = 0

###############################################################################

###############################################################################

class UpConv(Conv):
    """
    Upconvolution layer.

    E.g. pooling + upconv with p=3::

          x x x x x x x x x    before pooling (not in this layer)
           \|/   \|/   \|/     pooling (not in this layer)
            x     x     x      input to this layer
        0 0 x 0 0 x 0 0 x 0 0  unpooling + padding (done in this layer)
           /|\   /|\   /|\     conv on unpooled (done in this layer)
          y y y y y y y y y    result of this layer

    Parameters
    ----------
    parent: Node
        The input node.
    n_f: int
        Number of filters (nodes) in layer.
    pool_shape: tuple
        Size/shape of pooling.
    activation_func: str
        Activation function name.
    identity_init: bool
        Initialise weights to result in pixel repetition upsampling
    batch_normalisation: str or None
        Batch normalisation mode.
        Can be False (inactive), "train" or "fadeout".
    dropout_rate: float
        Dropout rate (probability that a node drops out in a training step).
    name: str
        Layer name.
    print_repr: bool
        Whether to print the node representation upon initialisation.
    w: np.ndarray or T.TensorVariable
        Weight matrix.
        If this is a np.ndarray, its values are used to initialise a
        shared variable for this layer.
        If it is a T.TensorVariable, it is directly used (weight sharing
        with the layer which this variable comes from).
    b: np.ndarray or T.TensorVariable
        Bias vector.
        If this is a np.ndarray, its values are used to initialise a
        shared variable for this layer.
        If it is a T.TensorVariable, it is directly used (weight sharing
        with the layer which this variable comes from).
    gamma
        (For batch normalisation) Initializes gamma parameter.
    mean
        (For batch normalisation) Initializes mean parameter.
    std
        (For batch normalisation) Initializes std parameter.
    gradnet_mode
    """

    def __init__(self, parent, n_f, pool_shape, activation_func='relu',
                 identity_init=True, batch_normalisation=False, dropout_rate=0,
                 name="upconv", print_repr=True, w=None, b=None, gamma=None,
                 mean=None, std=None, gradnet_mode=None):
        filter_shape = pool_shape
        super(UpConv, self).__init__(parent, n_f, filter_shape, pool_shape,
                                     'valid', activation_func, mfp=False,
                                     batch_normalisation=batch_normalisation,
                                     dropout_rate=dropout_rate, name=name,
                                     print_repr=print_repr, w=w, b=b,
                                     gamma=gamma, mean=mean, std=std,
                                     gradnet_mode=gradnet_mode)

        if identity_init:
            try:
                w_val = self.w.get_value() * 0.1
                s = np.minimum(w_val.shape[0], w_val.shape[1])
                s = np.arange(s)
                w_val[s,s] = 1.0
                self.w.set_value(w_val)
                self.b.set_value(self.b.get_value()*0.0)
            except:
                logger.warn("identity_init failed")


    def _make_output(self):
        """
        Computation of Theano output.
        """
        input_tensor = self.parent.output
        input_shape  = list(self.parent.shape)
        pattern  = ['x' for i in input_tensor.shape]
        pattern[self.axis] = 0
        activation_func = self.activation_func
        if activation_func.startswith("maxout"):
            r=int(activation_func.split(" ")[1])
            assert r>=2
            self.filter_shape /= r

        if activation_func=='prelu':
            b  = self.b[:,0].dimshuffle(pattern)
            b1 = self.b[:,1].dimshuffle(pattern)
        else:
            b   = self.b.dimshuffle(pattern)
            b1  = None

        spax = self.spatial_axes
        pool = np.array(self.pool_shape)
        input_shape_up = np.array(input_shape)
        if len(spax)==3 and not computations.dnn_avail:
          unpooled = computations.unpooling(input_tensor, self.pool_shape, self.spatial_axes)
          self._debug_outputs.append(unpooled)
          input_shape_up[spax] = input_shape_up[spax] * pool + pool - 1
          input_shape_up = list(input_shape_up)
          lin_output = computations.conv(unpooled, self.w, self.axis_order,
                                           border_mode=self.conv_mode,
                                           x_shape=input_shape_up, w_shape=self.w_sh)
        else:
            input_shape_up[spax] = input_shape_up[spax] * pool
            input_shape_up = list(input_shape_up)
            w = T.swapaxes(self.w, 0, 1)
            w_sh = list(self.w_sh)
            w_sh[0], w_sh[1] = w_sh[1], w_sh[0]
            lin_output = computations.upconv(input_tensor, w, self.pool_shape,
                                             x_shape=input_shape_up,
                                             w_shape=w_sh,
                                             axis_order='dnn')

        if self.batch_normalisation in ['train', 'fadeout']:
            mean = computations.apply_except_axis(
                lin_output,self.axis, T.mean).dimshuffle(pattern)
            std = computations.apply_except_axis(
                lin_output,self.axis, T.std).dimshuffle(pattern) + 1e-6
            gamma = self.gamma.dimshuffle(pattern)

            if self.batch_normalisation=='fadeout':
                logger.warning("Batch normalisation mode 'fadeout' does not "
                               "work for less than 50%...")
                mean = self.gradnet_rate * mean
                std  = self.gradnet_rate * std + (1-self.gradnet_rate) * 1.0
                gamma = self.gradnet_rate * gamma

            self.mean.updates = (self.mean,
                                 0.9995 * self.mean + 0.0005 * T.extra_ops.squeeze(mean))
            self.std.updates  = (self.std,
                                 0.9995 * self.std + 0.0005 * T.extra_ops.squeeze(std))

        elif self.batch_normalisation=='predict':
            mean = self.mean.dimshuffle(pattern)
            std = self.std.dimshuffle(pattern)
            gamma = self.gamma.dimshuffle(pattern)
        else:
            mean = 0
            std = 1
            gamma = 1


        lin_output =  (gamma / std) * lin_output + b - (gamma * mean / std)
        lin_output = computations.apply_activation(lin_output, activation_func, b1)

        if self.dropout_rate:
            rng = T.shared_randomstreams.RandomStreams(int(time.time()))
            p   = 1 - self.dropout_rate
            dropout_gate = rng.binomial(size=lin_output.shape, n=1, p=p,
                                        dtype=theano.config.floatX)
            dropout_gate *= 1.0 / p
            lin_output *= dropout_gate  #.dimshuffle(('x', 0))

        self.output = lin_output

    def _calc_shape(self):
        """
        Calculate and set self.shape.
        """
        self.strides = np.divide(self.strides,self.pool_shape)
        sh = self.parent.shape
        for j,(i,f,p) in enumerate(zip(self.spatial_axes, self.filter_shape, self.pool_shape)):
            s = 1 - f if self.conv_mode=='valid' else f -1
            s = (sh[i] * p) + p - 1 + s # unpool with margin then apply conv
            sh = sh.updateshape(i, s)
            # Unpooling creates asymmetric FOV (left/right is different for
            # some neurons), therefore we flag the FOV as exceptional with '-1'
            sh = sh.updatefov(j, -1)

        sh = sh.updateshape('f', self.n_f)
        sh = sh.updatestrides(self.strides)
        self.shape = sh


    def _calc_comp_cost(self):
        """
        Calculate and set self.computational_cost.
        """
        sh = self.parent.shape
        n_position = 1
        for i,f,p in zip(self.spatial_axes, self.filter_shape, self.pool_shape):
            s = 1 - f if self.conv_mode=='valid' else f -1
            n_position *= (sh[i] * p) + s

        b = 1 if sh['b'] is None else sh['b']
        self.computational_cost = np.product(self.w_sh) * n_position * b



    def __repr__(self):
        s = super(NeuralLayer, self).__repr__()
        s += "\n"
        s += "  n_f=%i, " %(self.n_f,)
        s += "%id upconv, kernel=%s, pool=%s, "\
             %(self.conv_dim, self.filter_shape, self.pool_shape)
        s += "act='%s', " %(self.activation_func,)
        if self.dropout_rate:
            s += "Dropout rate=%.1f, "%(self.dropout_rate.get_value())
        if self.batch_normalisation:
            s += "BN in '%s' mode "%(self.batch_normalisation,)
        return s

    def make_dual(self, *args, **kwargs):
        raise NotImplementedError("Use Conv instead?")


class Crop(Node):
    """
    This node type crops the output of its parent.

    Parameters
    ----------
    parent: Node
        The input node whose output should be cropped.
    crop: tuple or list of ints
        Crop each spatial axis from either side by this number.
    name: str
        Node name.
    print_repr: bool
        Whether to print the node representation upon initialisation.
    """  # TODO: Write an example
    def __init__(self, parent, crop, name="crop", print_repr=False):

        super(Crop, self).__init__(parent, name, print_repr)
        self.crop=crop

    def _make_output(self):
        """
        Computation of Theano output.
        """
        # It is assumed that all other dimensions are matching
        cropper = []
        k = 0
        for i,s in enumerate(self.parent.shape):
            if i in self.parent.shape.spatial_axes:
                off = self.crop[k]
                cropper.append(slice(off, s-off))
                k += 1
            else:
                cropper.append(slice(None))

        cropper = tuple(cropper)
        self.output = self.parent.output[cropper]

    def _calc_shape(self):
        """
        Calculate and set self.shape.
        """
        sh = self.parent.shape.copy()
        k = 0
        for i,s in enumerate(self.parent.shape):
            if i in self.parent.shape.spatial_axes:
                off = self.crop[k]
                sh = sh.updateshape(i,s-2*off)
                k += 1

        self.shape = sh

    def _calc_comp_cost(self):
        """
        Calculate and set self.computational_cost.

        For this Node type this is hard-coded to 0.
        """
        self.computational_cost = 0


def ImageAlign(hi_res, lo_res, hig_res_n_f,
                    activation_func='relu', identity_init=True,
                    batch_normalisation=False, dropout_rate=0,
                    name="upconv", print_repr=True, w=None, b=None, gamma=None,
                    mean=None, std=None, gradnet_mode=None):
    """
    Try to automatically align and concatenate a high-res and a low-res
    convolution output of two branches of a CNN by applying UpConv and Crop to
    make their shapes and strides compatible.
    UpConv is used if the low-res Node's strides are at least twice as large
    as the strides of the high-res Node in any dimension.

    This function can be used to simplify creation of e.g. architectures similar to
    U-Net (see https://arxiv.org/abs/1505.04597).

    If a ValueError that the shapes cannot be aligned is thrown,
    you can try changing the filter shapes and pooling factors of the
    (grand-)parent Nodes or add/remove Convolutions and Crops in the preceding
    branches until the error disappears (of course you should try to keep
    those changes as minimal as possible).

    (This function is an alias for UpConvMerge.)

    Parameters
    ----------
    hi_res: Node
        Parent Node with high resolution output.
    lo_res: Node
        Parent Node with low resolution output.
    hig_res_n_f: int
        Number of filters for the aligning UpConv.
    activation_func: str
        (passed to new UpConv if required).
    identity_init: bool
        (passed to new UpConv if required).
    batch_normalisation: bool
        (passed to new UpConv if required).
    dropout_rate: float
        (passed to new UpConv if required).
    name: str
        Name of the intermediate UpConv node if required.
    print_repr: bool
        Whether to print the node representation upon initialisation.
    w
        (passed to new UpConv if required).
    b
        (passed to new UpConv if required).
    gamma
        (passed to new UpConv if required).
    mean
        (passed to new UpConv if required).
    std
        (passed to new UpConv if required).
    gradnet_mode
        (passed to new UpConv if required).

    Returns
    -------
    Concat
        Concat Node that merges the aligned high-res and low-res outputs.
    """
    ###TODO exchange UpConv and Crop to save computation in some cases

    sh_hi = hi_res.shape
    sh_lo = lo_res.shape
    assert len(sh_hi)==len(sh_lo)
    assert sh_hi.spatial_axes == sh_lo.spatial_axes

    unpool = sh_lo.strides // sh_hi.strides
    if np.any(unpool>1):
        lo_res = UpConv(lo_res, hig_res_n_f, unpool,
               activation_func=activation_func, identity_init=identity_init,
               batch_normalisation=batch_normalisation, dropout_rate=dropout_rate,
               name=name, print_repr=print_repr, w=w, b=b, gamma=gamma,
               mean=mean, std=std, gradnet_mode=gradnet_mode)

    # No both have same stride
    # Shapes may have changed
    sh_hi = hi_res.shape.spatial_shape
    sh_lo = lo_res.shape.spatial_shape

    crop_lo = []
    crop_hi = []
    for i in range(len(sh_hi)):
        diff = sh_hi[i] - sh_lo[i]  # different in orignal space
        if diff % 2!=0:
            raise ValueError("hi_res and lo_res maps cannot "
                             "be aligned with shapes:\n%s\n%s" % (sh_hi,sh_lo))
        if diff > 0:
            crop_hi.append(diff // 2 )
            crop_lo.append(0)
        else:
            crop_lo.append(-diff // 2)
            crop_hi.append(0)

    if np.any(crop_lo):
        lo_res = Crop(lo_res, crop_lo, print_repr=True)
    if np.any(crop_hi):
        hi_res = Crop(hi_res, crop_hi, print_repr=True)

    out = Concat((lo_res, hi_res), axis='f', name='merge', print_repr=True)

    return out

UpConvMerge = ImageAlign

class Pool(Node):
    """
    Pooling layer.

    Reduces the count of training parameters by reducing the spatial size
    of its input by the factors given in ``pool_shape``.

    Pooling modes other than max-pooling can only be selected if cuDNN is
    available.

    Parameters
    ----------
    parent: Node
        The input node.
    pool_shape: tuple
        Tuple of pooling factors (per dimension) by which the input
        is downsampled.
    stride: tuple
        Stride sizes (per dimension).
    mfp: bool
        If max-fragment-pooling should be used.
    mode: str
        (only if cuDNN is available)
        Mode can be any of the modes supported by Theano's dnn_pool():
        ('max', 'average_inc_pad', 'average_exc_pad', 'sum').

        'max' (default): max-pooling
        'average' or 'average_inc_pad': average-pooling
        'sum': sum-pooling
    name: str
        Name of the pooling layer.
    print_repr: bool
        Whether to print the node representation upon initialisation.
    """

    def __init__(self, parent, pool_shape, stride=None, mfp=False, mode='max',
                 name="pool", print_repr=True):
        super(Pool, self).__init__(parent, name, print_repr)

        if mfp and stride is not None:
            raise ValueError("Cannot use custom stride and MFP together")

        if stride is None:
            stride = pool_shape

        if mode == 'average':
            mode = 'average_inc_pad'  # Theano's internal name. 'average' is deprecated.

        self.pool_shape = pool_shape
        self.pool_stride = stride
        self.mfp = mfp
        self.mode = mode

        self.strides = parent.shape.strides
        self.mfp_offsets = parent.shape.mfp_offsets
        self.axis = parent.shape.tag2index('f') #retrieve feature shape's index
        self.axis_order = None

        spatial_axes = self.parent.shape.spatial_axes
        conv_dim = len(pool_shape)
        x_dim    = len(self.parent.shape)
        n_in = parent.shape['f'] #retrieve feature shape
        fail = False
        if conv_dim==1:
            if x_dim!=3 or spatial_axes!=[2]:
                fail = True
        elif conv_dim==2:
            if x_dim!=4 or spatial_axes!=[2,3]:
                fail = True

        elif conv_dim==3:
            if x_dim!=5:
                fail = True
            if spatial_axes==[2,3,4]:
                self.axis_order = 'dnn'
            elif spatial_axes==[1,3,4]:
                self.axis_order = 'theano'

            else:
                fail = True

        if fail:
            raise NotImplementedError("Cannot convolve non-standard shapes / axis orders. "
                                      "Implement reshaping before conv "
                                      "and re-reshaping afer!")

        self.spatial_axes = spatial_axes
        self.conv_dim = conv_dim


    def _make_output(self):
        """
        Computation of Theano output.
        """
        input_tensor = self.parent.output
        pattern  = ['x' for i in input_tensor.shape]
        pattern[self.axis] = 0

        if self.mfp:
            assert self.pool_stride == self.pool_shape
            if self.input_nodes[0].shape['b']!=1:
                raise ValueError("For MFP the batchsize of the raw image input must be 1.")
            lin_output, offsets_new, strides_new = computations.fragmentpool(input_tensor,
                                                                        self.pool_shape,
                                                                        self.mfp_offsets,
                                                                        self.strides,
                                                                        self.spatial_axes,
                                                                        mode=self.mode)
            self.mfp_offsets = offsets_new
            self.strides = strides_new
        else:
            lin_output = computations.pooling(input_tensor,self.pool_shape,
                                              self.spatial_axes, stride=self.pool_stride,
                                              mode=self.mode)
            self.strides = np.multiply(self.pool_stride, self.strides)

        self.output = lin_output


    def _calc_shape(self):
        """
        Calculate and set self.shape.
        """
        sh = self.parent.shape
        for j,(i,p,st) in enumerate(zip(self.spatial_axes , self.pool_shape, self.pool_stride)):
            tmp = sh[i] - p + st - 1
            s = tmp//st + 1
            if self.mfp:
                raise NotImplementedError("Check this first before use")
                if (tmp - p + 1)%st!=0:
                    raise ValueError("Cannot downsample spatial axis '%s' of length %i "
                                     "by factor %i with pool %i, and using MFP."\
                                     %(sh.tags[i], sh[i], st, p))
            else:
                if (tmp+1)%st!=0:
                    raise ValueError("Cannot downsample spatial axis '%s' of length %i "
                                     "by factor %i with pool %i."\
                                     %(sh.tags[i], sh[i], st, p ))
            sh = sh.updateshape(i, s)
            if sh.fov[j]>0:
                fov = sh.fov[j] + (p-1) * sh.strides[j]
            else:
                fov = -1
            sh = sh.updatefov(j, fov)

        if self.mfp:
            sh = sh.updatemfp_offsets(self.mfp_offsets)
            sh = sh.updateshape('b', np.prod(self.pool_shape), mode='mult')

        sh = sh.updatestrides(self.strides)
        self.shape = sh


class FaithlessMerge(Node):
    """
    FaithlessMerge node.

    Parameters
    ----------
    hard_features: Node
    easy_features: Node
    axis
    failing_prob: float
        The higher the more often merge is unreliable
    hardeasy_ratio: float
        The higher the more often the harder features fail instead of the easy ones
    name: str
            Name of the pooling layer.
        print_repr: bool
            Whether to print the node representation upon initialisation.
    """

    def __init__(self, hard_features, easy_features, axis='f', failing_prob=0.5,
                 hardeasy_ratio=0.8, name="faithless_merge", print_repr=True):
        parent_nodes = (hard_features, easy_features)
        super(FaithlessMerge, self).__init__(parent_nodes, name, print_repr)

        if isinstance(axis, str):
            self.axis = parent_nodes[0].shape.tag2index(axis)
        else:
            self.axis = axis

        failing_prob = VariableParam(value=failing_prob,
                                      name="failing_prob",
                                      dtype=floatX,
                                      apply_train=False)

        hardeasy_ratio = VariableParam(value=hardeasy_ratio,
                                     name="hardeasy_ratio",
                                     dtype=floatX,
                                     apply_train=False)


        self.params['failing_prob'] = failing_prob
        self.params['hardeasy_ratio'] = hardeasy_ratio
        self.failing_prob = failing_prob
        self.hardeasy_ratio = hardeasy_ratio


    def _make_output(self):
        """
        Computation of Theano output.
        """
        # It is assumed that all other dimensions are matching
        rng = T.shared_randomstreams.RandomStreams(int(time.time()))
        size = [1,] * self.parent[0].output.ndim
        axes = list(range(self.parent[0].output.ndim))

        not_failing = rng.binomial(size=size, n=1, p=self.failing_prob,
                                    dtype=theano.config.floatX)
        not_failing = T.addbroadcast(not_failing, *axes)
        hard_fails = rng.binomial(size=size, n=1, p=1-self.hardeasy_ratio,
                                    dtype=theano.config.floatX)
        hard_fails = T.addbroadcast(hard_fails, *axes)

        hard = self.parent[0].output * (1 - hard_fails * not_failing)
        easy = self.parent[1].output * (1 - (1 - hard_fails) * not_failing)

        self.output = T.concatenate([hard, easy], axis=self.axis)

    def _calc_shape(self):
        """
        Calculate and set self.shape.
        """
        joint_axis_size = reduce(lambda x, y: x + y.shape[self.axis],
                                 self.parent, 0)
        # assuming all other dimensions are equal
        sh = self.parent[0].shape.updateshape(self.axis, joint_axis_size)
        self.shape = sh

    def _calc_comp_cost(self):
        """
        Calculate and set self.computational_cost.

        For this Node type this is hard-coded to 0.
        """
        self.computational_cost = 0


class GRU(NeuralLayer):
    """
    Gated Recurrent Unit Layer.

    Parameters
    ----------
    parent: Node
        The input node.
    memory_state: Node
        Memory node.
    n_f: int
        Number of features.
    activation_func: str
        Activation function name.
    flatten: bool
        (Unsupported).
    batch_normalisation: str or None
        Batch normalisation mode.
        Can be False (inactive), "train" or "fadeout".
    dropout_rate: float
        Dropout rate (probability that a node drops out in a training step).
    name: str
        Layer name.
    print_repr: bool
        Whether to print the node representation upon initialisation.
    w: np.ndarray or T.TensorVariable
        (Unsupported).
        Weight matrix.
        If this is a np.ndarray, its values are used to initialise a
        shared variable for this layer.
        If it is a T.TensorVariable, it is directly used (weight sharing
        with the layer which this variable comes from).
    b: np.ndarray or T.TensorVariable
        (Unsupported).
        Bias vector.
        If this is a np.ndarray, its values are used to initialise a
        shared variable for this layer.
        If it is a T.TensorVariable, it is directly used (weight sharing
        with the layer which this variable comes from).
    gamma
        (For batch normalisation) Initializes gamma parameter.
    mean
        (For batch normalisation) Initializes mean parameter.
    std
        (For batch normalisation) Initializes std parameter.
    gradnet_mode
    """

    def __init__(self, parent, memory_state, n_f, activation_func='tanh',
                 flatten=False, batch_normalisation=False, dropout_rate=0,
                 name="gru", print_repr=True, w=None, b=None,
                 gamma=None, mean=None, std=None, gradnet_mode=None):
        parent_nodes = (parent, memory_state)
        super(GRU, self).__init__(parent_nodes, name, print_repr)

        self.n_f = n_f
        self.n_f_memory = memory_state.shape['f']
        self.activation_func = activation_func
        self.batch_normalisation = batch_normalisation
        self.gradnet_mode = gradnet_mode
        self.axis = parent.shape.tag2index('f') #retrieve feature shape's index
        self.spatial_axes = parent.shape.spatial_axes
        self.flatten = flatten

        if flatten:
            raise NotImplementedError("Flatten is not yet supported for GRU.")
            n_in = parent.shape.stripbatch_prod
        else:
            n_in = parent.shape['f']


        if self.n_f_memory != n_f:
            raise ValueError("n_f_memory != n_f not possible.")
        if parent.shape.hastag('r'):
            raise ValueError("Input must not have 'r' axis.")

        n_comb = self.n_f_memory + n_in
        if w != None or b != None:
             raise NotImplementedError("Initial weights are not yet supported for GRU.")

        w_sh = (n_comb, 3*n_f) #  [h_t-1, x] x [W_z/x, W_r/x, W_h/x]
        self._setup_params(w_sh, w, b, gamma, mean, std, dropout_rate)


    def _make_output(self):
        """
        Computation of Theano output.
        """
        parent = self.parent[0].output
        memory = self.parent[1].output
        pattern  = ['x' for i in parent.shape]
        pattern[self.axis] = 0
        broad_caster_shape = list(parent.shape)
        broad_caster_shape[self.axis] = self.n_f_memory
        broad_caster = T.ones(broad_caster_shape, dtype=memory.dtype)
        memory = memory * broad_caster
        input_tensor = T.concatenate([memory, parent] , axis=self.axis)

        activation_func = self.activation_func
        if activation_func.startswith("maxout"):
            r=int(activation_func.split(" ")[1])
            assert r>=2
            self.n_f /= r

        if activation_func=='prelu':
            b  = self.b[:-self.n_f,0].dimshuffle(pattern)
            b_h = self.b[-self.n_f:,0].dimshuffle(pattern)
            b1 = self.b[:-self.n_f,1].dimshuffle(pattern)
            b1_h = self.b[-self.n_f:,1].dimshuffle(pattern)
        else:
            b   = self.b[:-self.n_f].dimshuffle(pattern)
            b_h = self.b[-self.n_f:].dimshuffle(pattern)
            b1  = None
            b1_h = None

        lin_output = computations.dot(input_tensor, self.w[:, :-self.n_f], self.axis)

        if self.batch_normalisation in ['train', 'fadeout']:
            raise NotImplementedError("Batch normalisation not yet supported  for GRU.")
            mean = computations.apply_except_axis(
                lin_output,self.axis, T.mean).dimshuffle(pattern)
            std = computations.apply_except_axis(
                lin_output,self.axis, T.std).dimshuffle(pattern) + 1e-6
            gamma = self.gamma.dimshuffle(pattern)

            if self.batch_normalisation=='fadeout':
                logger.warning("Batch Normalisation mode 'fadeout' does not "
                               "work for less than 50%...")
                mean = self.gradnet_rate * mean
                std  = self.gradnet_rate * std + (1-self.gradnet_rate) * 1.0
                gamma = self.gradnet_rate * gamma

            self.mean.updates = (self.mean,
                                 0.9995 * self.mean + 0.0005 * T.extra_ops.squeeze(mean))
            self.std.updates  = (self.std,
                                 0.9995 * self.std + 0.0005 * T.extra_ops.squeeze(std))

        elif self.batch_normalisation=='predict':
            raise NotImplementedError("Batch normalisation not yet supported for GRU.")
            mean = self.mean.dimshuffle(pattern)
            std = self.std.dimshuffle(pattern)
            gamma = self.gamma.dimshuffle(pattern)
        else:
            mean = 0
            std = 1
            gamma = 1

        lin_output =  (gamma / std) * lin_output + b - (gamma * mean / std)
        act = computations.apply_activation(lin_output, 'sig', b1)
        slice_obj = [slice(None) for i in range(act.ndim)]
        slice_obj[self.axis] = slice(0, self.n_f)
        z = act[slice_obj]
        slice_obj[self.axis] = slice(self.n_f, None)
        r = act[slice_obj]
        gated_input = T.concatenate([r*memory, parent], axis=self.axis)
        h_tilde = computations.dot(gated_input, self.w[:, -self.n_f:], self.axis)
        h_tilde =  (gamma / std) *  h_tilde  + b_h - (gamma * mean / std)
        h_tilde = computations.apply_activation(h_tilde, activation_func, b1_h)
        act = (1 - z) * memory + z * h_tilde

        self._debug_outputs = [memory, act, z, r,]
        if self.dropout_rate:
            raise NotImplementedError("Dropout not yet supported for GRU.")
            rng = T.shared_randomstreams.RandomStreams(int(time.time()))
            p   = 1 - self.dropout_rate
            dropout_gate = rng.binomial(size=(self.n_f,), n=1, p=p,
                                        dtype=theano.config.floatX)
            dropout_gate *= 1.0 / p
            act =  act * dropout_gate.dimshuffle(('x', 0))

        self.output = act


    def _calc_shape(self):
        """
        Calculate and set self.shape.
        """
        sh = self.parent[0].shape
        if self.flatten:
            self.shape = TaggedShape((sh['b'], self.n_f), 'b,f')
        else:
            self.shape = sh.updateshape('f', self.n_f)


    def _calc_comp_cost(self):
        """
        Calculate and set self.computational_cost.
        """
        n = self.parent[0].shape.stripnone_prod
        self.computational_cost = 3 * n * self.n_f


class LSTM(NeuralLayer):
    """
    Long short term memory layer.

    Using an implementation without peepholes in f, i, o, i.e. weights
    cell state is not taken into account for weights. See
    http://colah.github.io/posts/2015-08-Understanding-LSTMs/.

    Parameters
    ----------
    parent: Node
        The input node.
    memory_states: Node
        Concatenated (initial) feed-back and cell state (one Node!).
    n_f: int
        Number of features.
    activation_func: str
        Activation function name.
    flatten
    batch_normalisation: str or None
        Batch normalisation mode.
        Can be False (inactive), "train" or "fadeout".
    dropout_rate: float
        Dropout rate (probability that a node drops out in a training step).
    name: str
        Layer name.
    print_repr: bool
        Whether to print the node representation upon initialisation.
    w: np.ndarray or T.TensorVariable
        Weight matrix.
        If this is a np.ndarray, its values are used to initialise a
        shared variable for this layer.
        If it is a T.TensorVariable, it is directly used (weight sharing
        with the layer which this variable comes from).
    b: np.ndarray or T.TensorVariable
        Bias vector.
        If this is a np.ndarray, its values are used to initialise a
        shared variable for this layer.
        If it is a T.TensorVariable, it is directly used (weight sharing
        with the layer which this variable comes from).
    gamma
        (For batch normalisation) Initializes gamma parameter.
    mean
        (For batch normalisation) Initializes mean parameter.
    std
        (For batch normalisation) Initializes std parameter.
    gradnet_mode
    """

    def __init__(self, parent, memory_states, n_f, activation_func='tanh',
                 flatten=False, batch_normalisation=False, dropout_rate=0,
                 name="lstm", print_repr=True, w=None, b=None,
                 gamma=None, mean=None, std=None, gradnet_mode=None):
        parent_nodes = (parent, memory_states)
        super(LSTM, self).__init__(parent_nodes, name, print_repr)

        self.n_f = n_f
        self.n_f_memory = memory_states.shape['f']
        self.activation_func = activation_func
        self.batch_normalisation = batch_normalisation
        self.gradnet_mode = gradnet_mode
        self.axis = parent.shape.tag2index('f') #retrieve feature shape's index
        self.spatial_axes = parent.shape.spatial_axes
        self.flatten = flatten

        if flatten:
            raise NotImplementedError("Flatten is not yet supported for LSTM.")
        else:
            n_in = parent.shape['f']

        n_comb = n_f + n_in
        if w != None or b != None:
             raise NotImplementedError("Initial weights are not yet supported for LSTM.")

        if self.n_f_memory != 2*n_f:
            raise ValueError("n_f of memory_states must be 2*n_f.")
        if parent.shape.hastag('r'):
            raise ValueError("Input must not have 'r' axis.")

        w_sh = (n_comb, 4*n_f) # f, i, o, C
        self._setup_params(w_sh, w, b, gamma, mean, std, dropout_rate)


    def _make_output(self):
        """
        Computation of Theano output.
        """
        parent = self.parent[0].output
        memory = self.parent[1].output

        broad_caster_shape = list(parent.shape)
        broad_caster_shape[self.axis] = self.n_f_memory
        broad_caster = T.ones(broad_caster_shape, dtype=memory.dtype)
        memory = memory * broad_caster

        slice_obj = [slice(None) for i in range(len(self.parent[1].shape))]
        slice_obj[self.parent[1].shape.tag2index('f')] = slice(self.n_f)
        feed_back = memory[slice_obj]
        slice_obj[self.parent[1].shape.tag2index('f')] = slice(self.n_f, None)
        cell_state = memory[slice_obj]


        input_tensor = T.concatenate([feed_back, parent] ,
                                     axis=self.axis) #h, x
        pattern = ['x' for i in input_tensor.shape]
        pattern[self.axis] = 0
        activation_func = self.activation_func

        if activation_func.startswith("maxout"):
            r=int(activation_func.split(" ")[1])
            assert r>=2
            self.n_f /= r

        if activation_func=='prelu':
            b  = self.b[:, 0].dimshuffle(pattern)
            b1 = self.b[:, 1]
            b1_f = b1[:self.n_f].dimshuffle(pattern)
            b1_i = b1[self.n_f:2*self.n_f].dimshuffle(pattern)
            b1_o = b1[2*self.n_f:3*self.n_f].dimshuffle(pattern)
            b1_c = b1[3*self.n_f:].dimshuffle(pattern)
        else:
            b   = self.b.dimshuffle(pattern)
            b1_f = None
            b1_i = None
            b1_o = None
            b1_c = None

        lin_output = computations.dot(input_tensor, self.w, self.axis)

        if self.batch_normalisation in ['train', 'fadeout']:
            raise NotImplementedError("Batch normalisation not yet supported for LSTM.")
            mean = computations.apply_except_axis(
                lin_output,self.axis, T.mean).dimshuffle(pattern)
            std = computations.apply_except_axis(
                lin_output,self.axis, T.std).dimshuffle(pattern) + 1e-6
            gamma = self.gamma.dimshuffle(pattern)

            if self.batch_normalisation=='fadeout':
                logger.warning("Batch Normalisation mode 'fadeout' does not "
                               "work for less than 50%...")
                mean = self.gradnet_rate * mean
                std  = self.gradnet_rate * std + (1-self.gradnet_rate) * 1.0
                gamma = self.gradnet_rate * gamma

            self.mean.updates = (self.mean,
                                 0.9995 * self.mean + 0.0005 * T.extra_ops.squeeze(mean))
            self.std.updates  = (self.std,
                                 0.9995 * self.std + 0.0005 * T.extra_ops.squeeze(std))

        elif self.batch_normalisation=='predict':
            raise NotImplementedError("Batch normalisation not yet supported for LSTM.")
            mean = self.mean.dimshuffle(pattern)
            std = self.std.dimshuffle(pattern)
            gamma = self.gamma.dimshuffle(pattern)
        else:
            mean = 0
            std = 1
            gamma = 1

        lin_output =  (gamma / std) * lin_output + b - (gamma * mean / std)
        slice_obj = [slice(None) for i in range(lin_output.ndim)]
        slice_obj[self.axis] = slice(self.n_f)
        f = computations.apply_activation(lin_output[slice_obj], 'sig', b1_f)
        slice_obj[self.axis] = slice(self.n_f, 2*self.n_f)
        i = computations.apply_activation(lin_output[slice_obj], 'sig', b1_i)
        slice_obj[self.axis] = slice(2*self.n_f, 3*self.n_f)
        o = computations.apply_activation(lin_output[slice_obj], 'sig', b1_o)
        slice_obj[self.axis] = slice(3*self.n_f, 4*self.n_f)
        c_tilde = computations.apply_activation(lin_output[slice_obj], activation_func, b1_c)
        cell_out = f * cell_state + i * c_tilde
        lin_output = o * computations.apply_activation(cell_out, activation_func, None)

        if self.dropout_rate:
            raise NotImplementedError("Dropout not yet supported for LSTM.")
            rng = T.shared_randomstreams.RandomStreams(int(time.time()))
            p   = 1 - self.dropout_rate
            dropout_gate = rng.binomial(size=(self.n_f,), n=1, p=p,
                                        dtype=theano.config.floatX)
            dropout_gate *= 1.0 / p
            lin_output =  lin_output * dropout_gate.dimshuffle(('x', 0))

        self.output = T.concatenate([lin_output, cell_out], axis=self.axis)


    def _calc_shape(self):
        """
        Calculate and set self.shape.
        """
        sh = self.parent[0].shape
        if self.flatten:
            self.shape = TaggedShape((sh['b'], 2*self.n_f), 'b,f')
        else:
            self.shape = sh.updateshape('f',2* self.n_f)


    def _calc_comp_cost(self):
        """
        Calculate and set self.computational_cost.
        """
        n = self.parent[0].shape.stripnone_prod
        self.computational_cost = 4 * n * self.n_f


class LRN(Node):
    """
    LRN (Local Response Normalization) layer.

    Parameters
    ----------
    parent: Node
        The input node.
    filter_shape: tuple
    mode: str
        Can be "spatial" or "channel".
    alpha: float
    k: float
    beta: float
    name: str
        Node name.
    print_repr: bool
        Whether to print the node representation upon initialisation.
    """

    def __init__(self, parent, filter_shape, mode='spatial',  alpha=0.0001,
                 k=1, beta=0.75, name="LRN", print_repr=True):
        super(LRN, self).__init__(parent, name, print_repr)

        self.mode = mode
        self.filter_shape = filter_shape
        self.axis = parent.shape.tag2index('f')  # retrieve feature shape's index
        if mode=='spatial':
            self.axis_order = None
            self.spatial_axes = self.parent.shape.spatial_axes
            conv_dim = len(self.spatial_axes)
            x_dim    = len(self.parent.shape)
            if len(self.spatial_axes)!=len(filter_shape):
                raise ValueError("The filter_shape dimensionality (%i) and the number "
                                 "of spatial dimensions in the input (%i)differ! "
                                 "Use filter size 1 on axes which should not be "
                                 "averaged."\
                                 %(len(filter_shape), conv_dim, ))

            n_in = parent.shape['f'] #retrieve feature shape
            fail = False
            if conv_dim==1:
                if x_dim!=3 or self.spatial_axes!=[2]:
                    fail = True
                w_sh = [n_in, n_in] + list(filter_shape)

            elif conv_dim==2:
                if x_dim!=4 or self.spatial_axes!=[2,3]:
                    fail = True
                w_sh = [n_in, n_in] + list(filter_shape)

            elif conv_dim==3:
                if x_dim!=5:
                    fail = True
                if self.spatial_axes==[2,3,4]:
                    self.axis_order = 'dnn'
                    w_sh = [n_in, n_in] + list(filter_shape)
                elif self.spatial_axes==[1,3,4]:
                    self.axis_order = 'theano'
                    w_sh = [n_in, filter_shape[0], n_in] + list(filter_shape[1:])
                else:
                    fail = True

            if fail:
                raise NotImplementedError("Cannot convolve non-standard shapes / axis orders. "
                                          "Implement reshaping before conv"
                                          "and re-reshaping afer!")

            self.conv_dim = conv_dim
            self.w_sh = w_sh
            value = np.zeros(w_sh, dtype=floatX)
            val = 1.0 / np.product(filter_shape)
            for i in range(n_in):
                value[i,i] = val

            self.average_filter = ConstantParam(value, '<%s_filter%s>'%(self.name, tuple(w_sh)))
            self.params['average_filter'] = self.average_filter
        elif mode=='channel':
            assert isinstance(filter_shape, int)
            assert filter_shape%2==1
        else:
            raise ValueError("Unknow mode %s"%mode)

        self.alpha = VariableParam(value=alpha, name="alpha",
                                   dtype=floatX, apply_train=False)

        self.beta = VariableParam(value=beta,name="beta",
                                  dtype=floatX, apply_train=False)

        self.k = VariableParam(value=k,name="k",
                               dtype=floatX, apply_train=False)

        self.params['alpha'] = self.alpha
        self.params['beta'] = self.beta
        self.params['k'] = self.k


    def _make_output(self):
        """
        Computation of Theano output.
        """
        input_tensor = self.parent.output
        input_shape  = list(self.parent.shape)

        if self.mode=='spatial':
            mean_square = computations.conv(T.square(input_tensor), self.average_filter,
                                           self.axis_order, border_mode='same',
                                           x_shape=input_shape, w_shape=self.w_sh)
        else:
            n_f = input_shape[self.axis]
            in_square = T.square(input_tensor)
            half_n = self.filter_shape // 2
            new_sh = list(input_tensor.shape)
            new_sh[self.axis] += 2 * half_n
            in_square_ext = T.zeros(new_sh, floatX)
            slicer = [slice(None)] * input_tensor.ndim
            slicer[self.axis] = slice(half_n,half_n+n_f)
            in_square_ext = T.set_subtensor(in_square_ext[slicer], in_square)
            # pad left
            slicer[self.axis] = slice(0, half_n)
            pad_slicer = [slice(None)] * input_tensor.ndim
            pad_slicer[self.axis] = slice(0, 1)
            in_square_ext = T.set_subtensor(in_square_ext[slicer], in_square[pad_slicer])
            # pad right
            slicer[self.axis] = slice(half_n+n_f, 2*half_n+n_f)
            pad_slicer[self.axis] = slice(n_f-1,n_f)
            in_square_ext = T.set_subtensor(in_square_ext[slicer], in_square[pad_slicer])

            mean_square = 0
            for i in range(self.filter_shape):
                slicer[self.axis] = slice(i,i+n_f)
                mean_square += in_square_ext[slicer]

            mean_square /= self.filter_shape


        divisor = T.power(self.k + self.alpha * mean_square, self.beta)
        self.output = input_tensor / divisor
        self._debug_outputs = [mean_square, divisor]
