# -*- coding: utf-8 -*-
# ELEKTRONN2 Toolkit
# Copyright (c) 2015 Marius Killinger
# All rights reserved

from __future__ import absolute_import, division, print_function
from builtins import filter, hex, input, int, map, next, oct, pow, range, super, zip


import logging
import time

import numpy as np
import theano
from theano import tensor as T, gof
from theano.gradient import disconnected_type

from .. import config
from .. import utils
from .node_basic import Node, GenericInput, FromTensor, model_manager
from .graphutils import TaggedShape, floatX
from .variables import VariableParam

logger = logging.getLogger('elektronn2log')
inspection_logger = logging.getLogger('elektronn2log-inspection')

__all__ = ['GaussianRV', 'SkelLoss',
           'SkelPrior', 'Scan', 'SkelGetBatch', 'SkelLossRec',
           'Reshape', 'SkelGridUpdate']


class GaussianRV(Node):
    """
    Parameters
    ----------
    mu: node
        Mean of the Gaussian density
    sig: node
        Sigma of the Gaussian density
    n_samples: int
        Number of samples to be drawn per instance.
        Special case '0': draw 1 sample but don't' increase rank of tensor!

    The output is a **sample** from separable Gaussians of given mean and
    sigma (but this operation is still differentiable, due to the
    "re-parameterisation trick").

    The output dimension mu.ndim+1 because the samples are accumulated along
    a new axis **right** of 'b' (batch).
    """

    def __init__(self, mu, log_sig, n_samples=0, name="state",
                 print_repr=True):
        super(GaussianRV, self).__init__((mu, log_sig), name, print_repr)

        self.mu = mu
        self.log_sig = log_sig
        self.n_samples = n_samples


    def _make_output(self):
        """ Computation of Theano Output """
        # It is assumed that all other dimensions are matching
        mu  = self.mu.output
        log_sig = self.log_sig.output
        sig = T.exp(log_sig)
        rng = T.shared_randomstreams.RandomStreams(int(time.time()))

        if self.n_samples>0:
            samples = []
            pattern = list(range(mu.type.ndim))
            batch_index = self.parent[0].shape.tag2index('b')
            pattern.insert(batch_index+1, 'x') # +1 because it is right of batch
            for i in range(self.n_samples):
                noise = rng.normal(mu.shape)
                z = mu + sig * noise
                samples.append(z.dimshuffle(pattern))
                z = T.concatenate(samples, axis=batch_index)
        else:
            noise = rng.normal(mu.shape)
            z = mu + sig * noise

        self.output = z

    def _calc_shape(self):
        if self.n_samples>0:
            self.shape = self.parent[0].shape.addaxis('b', self.n_samples, 's')
        else:
            self.shape = self.parent[0].shape.copy()

    def _calc_comp_cost(self):
        n = self.parent[0].shape.stripnone_prod
        if self.n_samples>0:
            self.computational_cost = n * self.n_samples
        else:
            self.computational_cost = n

    def make_priorlayer(self):
        """
        Creates a new Layer that calculates the Auto-Encoding-Variation-Bayes
        (AEVB) prior corresponding to this Layer.
        """
        # In particular the number of samples is the same, otherwise the relative
        # weight of prior and objective are wrong!
        return GaussianAEVBPrior(self.mu, self.log_sig,
                                 self.n_samples, name=self.name+"_prior")


###############################################################################

class GaussianAEVBPrior(GaussianRV):
    """
    Parameters
    ----------
    mu: Layer
        Mean of the Gaussian density
    sig: Layer
        Sigma of the Gaussian density
    n_samples: int
        Number of samples to be drawn per instance.
        Special case '0': draw 1 sample but don't' increase rank of tensor!

    The prior basically puts a L2-norm on mu and a similar constraint on sig
    but transformed such that sig=1 is favoured and for sig --> 0 it goes to inf.

    This Layer should only be created using the dedicated method of ``GaussianRV``.
    """

    def __init__(self, mu, log_sig, n_samples=0, name="prior",
                 print_repr=True):
        super(GaussianAEVBPrior, self).__init__(mu, log_sig, n_samples,
                                                name, print_repr)

    def _make_output(self):
        """ Computation of Theano Output """
        # It is assumed that all other dimensions are matching
        mu  = self.mu.output
        log_sig = self.log_sig.output
        sig = T.exp(log_sig)
        prior = 0.5 * (mu**2 + sig**2 - 1 - 2*log_sig)

        if self.n_samples>0:
            # The prior is replicated along a new sample axis in order to ensure
            # correct relative normalisation
            pattern = list(range(mu.type.ndim))
            batch_index = self.parent[0].shape.tag2index('b')
            pattern.insert(batch_index+1, 'x') # +1 because it is right of batch
            prior = prior.dimshuffle(pattern)
            prior = T.repeat(prior, self.n_samples, axis=-1)

        self.output = prior

###############################################################################

class StereographicMap(Node):
    """
    Interpret pred (bs, 3) as (X,Y,R), map to
    $(x, y, z) = R \cdot \left(\frac{2 X}{1 + X^2 + Y^2}, \frac{2 Y}{1 + X^2 + Y^2}, \frac{-1 + X^2 + Y^2}{1 + X^2 + Y^2}\right)$

    Parameters
    ----------
    pred
    name
    print_repr
    """

    def __init__(self, pred, name="stereo_map", print_repr=True):
        super(StereographicMap, self).__init__(pred, name, print_repr)
        self.pred   = pred.output
        self.pred_shape = pred.shape

    def _make_output(self):
        assert len(self.pred_shape)==2
        assert self.pred_shape[-1] == 3
        assert self.pred_shape[0] == 1

        R, Y, X = self.pred[:,0], self.pred[:,1], self.pred[:,2]
        R = T.nnet.softplus(R)

        z = - R * (-1 + X ** 2 + Y ** 2) / (1 + X ** 2 + Y ** 2)
        y = R * 2 * Y / (1 + X ** 2 + Y ** 2)
        x = R * 2 * X / (1 + X ** 2 + Y ** 2)

        self.output = T.stack((z, y, x)).T

    def _calc_shape(self):
        self.shape = TaggedShape([1,3], ['b','f'])


class SkelPrior(Node):
    """
    pred must be a vector of shape [(1,b),(3,f)] or [(3,f)]
    i.e. only batch_size=1 is supported.

    Parameters
    ----------
    pred
    target_length
    prior_n
    prior_posz
    prior_z
    prior_xy
    name
    print_repr
    """

    def __init__(self, pred, target_length=5.0, prior_n=0.0, prior_posz=0.0, prior_z=0.0, prior_xy=0.0,
                 name="skel_prior", print_repr=True):
        super(SkelPrior, self).__init__(pred, name, print_repr)

        self.pred = pred.output
        self.pred_shape = pred.shape
        self.target_length = VariableParam(value=target_length, name="target_length",
                                          dtype=floatX, apply_train=False)
        self.prior_n = VariableParam(value=prior_n, name="prior_n",
                                          dtype=floatX, apply_train=False)
        self.prior_z = VariableParam(value=prior_z, name="prior_z",
                                     dtype=floatX, apply_train=False)
        self.prior_posz = VariableParam(value=prior_posz, name="prior_posz",
                                     dtype=floatX, apply_train=False)
        self.prior_xy = VariableParam(value=prior_xy, name="prior_xy",
                                     dtype=floatX, apply_train=False)

        self.params['target_length'] = self.target_length
        self.params['prior_n'] = self.prior_n
        self.params['prior_z'] = self.prior_z
        self.params['prior_posz'] = self.prior_posz
        self.params['prior_xy'] = self.prior_xy


    def _make_output(self):
        assert len(self.pred_shape) in [1,2]
        assert self.pred_shape[-1] == 3
        if len(self.pred_shape)==2:
            assert self.pred_shape[0] == 1
            self.pred = self.pred[0]

        #self.pred = self.pred[:3] # just work on first 3 entries, rest might be anything

        aniso      = np.array([2,1,1], dtype=np.float32)
        norm       = T.sqrt(T.sum( (self.pred * aniso)**2))
        prior_n    = self.prior_n    * T.maximum(abs(norm - self.target_length) - 0.75, 0) # norm, maring of +/- 0.75
        prior_posz = self.prior_posz * T.nnet.softplus(-self.pred[0]) # penalise negative z
        prior_z    = self.prior_z    * abs(self.target_length/2-self.pred[0])  # make
        prior_xy   = self.prior_xy   *  T.sqrt(T.sum(self.pred[1:] ** 2)) # penalise larger xy

        self.output = prior_n + prior_posz + prior_z + prior_xy
        self._debug_outputs.extend([norm, prior_n, prior_posz,  prior_z, prior_xy])

    def _calc_shape(self):
        self.shape = TaggedShape([1], ['f'])


class SkelLossOP(theano.Op):
    """
    Parameters
    ----------
    predicted vec

    skel_obj

    Returns
    -------
    scalar loss
    """
    __props__ = ()
    def make_node(self, *inputs, **kwargs):
        inputs = list(inputs)
        inputs[0] = T.as_tensor_variable(inputs[0])
        loss = T.fvector('skel_loss')
        grad = T.fvector('skel_loss_grad')
        outputs = [loss, grad]
        self.loss_kwargs = kwargs
        return gof.Apply(self, inputs, outputs)

    def perform(self, node, inputs, output_storage):
        new_position_c, skel_obj, trafo = inputs
        loss, grad = output_storage

        new_position_l, _ = trafo.cnn_pred2lab_position(new_position_c)
        new_position_s = new_position_l[::-1]
        loss_, grad_s = skel_obj.get_loss_and_gradient(new_position_s,
                                         **self.loss_kwargs) # loss nearest_s
        grad_l = grad_s[::-1]
        grad_c = trafo.lab_coord2cnn_coord(grad_l)

        loss[0], grad[0] = loss_, grad_c
        if config.inspection:
            inspection_logger.info("pred_s: %s, grad_s: %s, pred_c: %s, grad_c: %s"
                                   % (new_position_s.tolist(),
                                      grad_s.tolist(),
                                      new_position_c.tolist(),
                                      grad_c.tolist()))


    def grad(self, inputs, outputs_gradients):
        grad_0 = self(*inputs)[1]
        grad_1 = disconnected_type()
        grad_2 = disconnected_type()
        return [grad_0, grad_1, grad_2]

    def connection_pattern(self, node):
        # there is only a grad of the first output w.r.t. the first input
        return [[True, False],[False, False],[False, False]]


class SkelLossN(Node):
    """
    pred must be a vector of shape [(1,b),(3,f)] or [(3,f)]
    i.e. only batch_size=1 is supported

    Parameters
    ----------
    pred
    skel
    trafo
    loss_kwargs
    name
    print_repr
    """

    def __init__(self, pred, skel, trafo, loss_kwargs, name="skel_loss", print_repr=True):
        super(SkelLossN, self).__init__((pred, skel, trafo), name, print_repr)

        self.skel = skel.output
        self.trafo = trafo.output
        self.pred = pred.output
        self.pred_shape = pred.shape
        self.loss_kwargs = loss_kwargs

    def _make_output(self):
        assert len(self.pred_shape) in [1,2]
        assert self.pred_shape[-1] == 3
        if len(self.pred_shape)==2:
            assert self.pred_shape[0] == 1
            self.pred = self.pred[0]

        loss, grad = SkelLossOP()(self.pred, self.skel,
                                  self.trafo, **self.loss_kwargs)
        self.output = loss

    def _calc_shape(self):
        self.shape = TaggedShape([1], ['f'])


def SkelLoss(pred, loss_kwargs, skel=None, name="skel_loss", print_repr=True):
    if skel is None:
        skel = GenericInput(name='skeleton')  # This is not an argument to this node

    trafo = GenericInput(name='trafo')
    return SkelLossN(pred, skel, trafo, loss_kwargs, name=name, print_repr=print_repr)

###############################################################################

class SkelLossRecOP(theano.Op):
    """
    For Recurrent
    """
    __props__ = ()
    def make_node(self, *inputs, **kwargs):
        inputs = list(inputs)
        inputs[0] = T.as_tensor_variable(inputs[0])
        loss = T.fvector('skel_loss')
        grad = T.fvector('skel_loss_grad')
        outputs = [loss, grad]
        self.loss_kwargs = kwargs
        return gof.Apply(self, inputs, outputs)

    def perform(self, node, inputs, output_storage):
        new_position_c, pred_features, skel_obj = inputs
        new_position_l, tracing_direc_il = skel_obj.trafo.cnn_pred2lab_position(new_position_c)
        new_position_s = new_position_l[::-1]
        tracing_direc_is = tracing_direc_il[::-1]
        if config.inspection:
            inspection_logger.info("LossOP, node pred %s" %(np.array_str(pred_features, precision=2, suppress_small=True),))
            inspection_logger.info("LossOP, new_position_c: %s, new_position_l: %s"%(new_position_c,np.array_str(new_position_l, precision=1, suppress_small=True)))
        loss_, grad_s, nearest_s = skel_obj.step_feedback(new_position_s,
                                                          tracing_direc_is,
                                                          new_position_c,
                                                          pred_features,
                                                          **self.loss_kwargs)
        grad_l = grad_s[::-1]
        grad_c = skel_obj.trafo.lab_coord2cnn_coord(grad_l)
        loss, grad = output_storage
        loss[0] = loss_.astype(np.float32)
        grad[0] = grad_c.astype(np.float32)
        if config.inspection:
            inspection_logger.info("LossOP, loss: %s, grad: %s"%(loss, grad_c))


    def grad(self, inputs, outputs_gradients):
        grad_0 = self(*inputs)[1]
        grad_1 = disconnected_type()
        grad_2 = disconnected_type()
        return [grad_0, grad_1, grad_2]

    def connection_pattern(self, node):
        # there is only a grad of the first output w.r.t. the first input
        return [[True, False],[False, False],[False, False]]

class SkelLossRec(Node):
    """
    pred must be a vector of shape [(1,b),(3,f)] or [(3,f)]
    i.e. only batch_size=1 is supported.

    Parameters
    ----------
    pred
    skel
    loss_kwargs
    name
    print_repr
    """

    def __init__(self, pred, skel, loss_kwargs, name="skel_loss", print_repr=True):
        super(SkelLossRec, self).__init__((pred, skel), name, print_repr)

        self.skel = skel.output
        self.pred = pred.output
        self.pred_shape = pred.shape
        self.loss_kwargs = loss_kwargs

    def _make_output(self):
        assert len(self.pred_shape) in [1,2]
        assert self.pred_shape[-1] == 10
        if len(self.pred_shape)==2:
            assert self.pred_shape[0] == 1
            self.pred = self.pred[0]

        pred = self.pred[:3]
        pred_feat = self.pred[3:]

        loss, grad = SkelLossRecOP()(pred, pred_feat,
                                     self.skel, **self.loss_kwargs)
        self.output = loss

    def _calc_shape(self):
        self.shape = TaggedShape([1], ['f'])

###############################################################################

class SkelGridUpdateOP(theano.Op):
    """
    For Recurrent
    """
    __props__ = ()

    def make_node(self, *inputs, **kwargs):
        inputs = list(inputs)
        best_pred = T.fmatrix('best_pred')
        preds     = T.fmatrix('preds')
        scores    = T.fvector('scores')
        outputs   = [best_pred, preds, scores]
        self.kwargs = kwargs
        return gof.Apply(self, inputs, outputs)

    def perform(self, node, inputs, output_storage):
        grid, skel_obj, radius, bio = inputs
        best_pred_, preds_, scores_ = skel_obj.step_grid_update(grid, radius, bio)
        best_pred, preds, scores = output_storage
        best_pred[0] = best_pred_
        preds[0] = preds_
        scores[0] = scores_


    def grad(self, inputs, outputs_gradients):
        grad_0 = disconnected_type()
        grad_1 = disconnected_type()
        return [grad_0, grad_1]

    def connection_pattern(self, node):
        # there is only a grad of the first output w.r.t. the first input
        return [[True, False, False], [False, False, False]]


class SkelGridUpdateN(Node):
    """
    pred must be a vector of shape [(1,b),(3,f)] or [(3,f)]
    i.e. only batch_size=1 is supported

    Parameters
    ----------
    grid
    skel
    radius
    bio
    name
    print_repr
    """

    def __init__(self, grid, skel, radius, bio, name="grid2pred",
                 print_repr=True):
        super(SkelGridUpdateN, self).__init__((grid, skel, radius, bio), name, print_repr)

        self.skel = skel.output
        self.grid = grid.output
        self.radius = radius.output
        self.bio = bio.output

    def _make_output(self):
        best_pred, preds, scores = SkelGridUpdateOP()(self.grid, self.skel,
                                                      self.radius, self.bio)
        self.output = [best_pred, preds, scores]
        self.output_names = ['tracing', 'tracings', 'scores']

    def _calc_shape(self):
        pred_sh  = TaggedShape((1,3), 'b,f')
        preds_sh = TaggedShape((1, None, 3), 'b,s,f')
        scores   = TaggedShape((1, None), 'b,s')
        self.shape = [pred_sh, preds_sh, scores]

def skelgridupdate_split(parent, name='skelgridupdate_split'):
    args = ()
    kwargs = dict(name=name)
    model_manager.current.register_split(parent, skelgridupdate_split, name, args,
                                         kwargs)

    outs = []
    for out, out_sh, out_name in zip(parent.output, parent.shape,
                                     parent.output_names):
        outs.append(FromTensor(out, out_sh, parent,
                               name=out_name, print_repr=False))

    return outs


def SkelGridUpdate(grid, skel, radius, bio, name="skelgridupdate", print_repr=True):
    preds = SkelGridUpdateN(grid, skel, radius, bio, name=name, print_repr=print_repr)
    outputs = scansplit(preds, name='skelgridupdate_split')
    return outputs

###############################################################################



class SkelGetBatchOP(theano.Op):
    """
    Parameters
    ----------
    predicted vec

    skel_obj

    Returns
    -------
    scalar loss
    """
    __props__ = ()
    def make_node(self, *inputs, **kwargs):
        inputs = list(inputs)
        img         = T.TensorType(theano.config.floatX, (False,) * 5, name='image')()
        target_img  = T.TensorType(theano.config.floatX, (False,) * 5, name='target_img')()
        target_grid = T.TensorType(theano.config.floatX, (False,) * 5, name='target_grid')()
        target_node = T.TensorType(theano.config.floatX, (False,) * 2, name='target_node')()
        outputs = [img, target_img, target_grid, target_node]

        self.get_batch_kwargs = kwargs
        return gof.Apply(self, inputs, outputs)

    def perform(self, node, inputs, output_storage):
        skel_obj, prediction, scale_strenght = inputs
        img, target_img, target_grid, target_node = output_storage
        batch = skel_obj.getbatch(prediction, scale_strenght, **self.get_batch_kwargs)
        img_, target_img_, target_grid_, target_node_ = batch

        img[0]            = img_.astype(np.float32)
        target_img[0]     = target_img_.astype(np.float32)
        target_grid[0]    = target_grid_[None].astype(np.float32)
        target_node[0]    = target_node_[None].astype(np.float32)
        if config.inspection:
            inspection_logger.info("GetBatch: success")

    def grad(self, inputs, outputs_gradients):
        grad = [[disconnected_type(),]*4,]*3
        return grad

    def connection_pattern(self, node):
        return [[False, False, False, False],
                [False, False, False, False],
                [False, False, False, False]]

    def do_constant_folding(self, node):
        return False


class SkelGetBatchN(Node):
    """
    Dummy Node to be used in the split-function.

    Parameters
    ----------
    skel
    aux
    img_sh
    get_batch_kwargs
    scale_strenght
    name
    print_repr
    """

    def __init__(self, skel, aux, img_sh, get_batch_kwargs, scale_strenght=None,
                 name='skel_batch', print_repr=False):
        super(SkelGetBatchN, self).__init__((skel, aux), name, print_repr)
        self.skel = skel
        self.aux  = aux
        self.img_sh = img_sh
        self.get_batch_kwargs = get_batch_kwargs
        scale_strenght = scale_strenght if scale_strenght else 0.0
        self.scale_strenght = VariableParam(value=scale_strenght,
                                            name="scale_strenght",
                                            dtype=floatX,
                                            apply_train=False)

        self.params["scale_strenght"] = self.scale_strenght

    def _make_output(self):
        batch = SkelGetBatchOP()(self.skel.output, self.aux.output,
                                 self.scale_strenght, **self.get_batch_kwargs)
        self.output      = batch[0]
        self.target_img  = batch[1]
        self.target_grid = batch[2]
        self.target_node = batch[3]


    def _calc_shape(self):
        self.shape = TaggedShape(self.img_sh, 'b,f,z,y,x')

    def _calc_comp_cost(self):
        self.computational_cost = 0


def skelgetbatch_split(batch, img_sh, t_img_sh, t_grid_sh, t_node_sh,
                        name='skel_batch_split'):
    args = (img_sh, t_img_sh, t_grid_sh, t_node_sh)
    kwargs = dict(name=name)
    model_manager.current.register_split(batch, skelgetbatch_split, name, args, kwargs)
    # Split the various outputs of the batch
    img         = FromTensor(batch.output, img_sh, batch,
                             name='skel_img', print_repr=False)
    target_img  = FromTensor(batch.target_img, t_img_sh, batch,
                             name='skel_t_img', print_repr=False)
    target_grid = FromTensor(batch.target_grid, t_grid_sh, batch,
                             name='skel_t_grid', print_repr=False)
    target_node = FromTensor(batch.target_node, t_node_sh, batch,
                             name='skel_t_node', print_repr=False)

    return img, target_img, target_grid, target_node


def SkelGetBatch(skel, aux, img_sh, t_img_sh, t_grid_sh, t_node_sh,
                 get_batch_kwargs, scale_strenght=None, name='skel_batch'):
    # The SkelGetBatchN-Node must be created outside of Split because
    # Split is re-called at model restore and hence GetBatch would be create
    # Twice then
    get_batch_kwargs['t_grid_sh'] = t_grid_sh
    batch = SkelGetBatchN(skel, aux, img_sh, get_batch_kwargs,
                          scale_strenght=scale_strenght, name=name)
    return skelgetbatch_split(batch, img_sh, t_img_sh,  t_grid_sh, t_node_sh,
                              name=name+"_split")

class ScanN(Node):
    """
    WARNING: this node may only be used in conjunction with ``scansplit``
    because its ``output`` and ``shape`` attributes are lists which
    will confuse normal nodes. The split wraps the outputs in individual
    Nodes (FromTensor).

    Parameters
    ----------
    step_result: node/list(nodes)
        nodes that represent results of step function
    in_memory: node/list(nodes)
        nodes that indicate at which place in the computational graph
        the memory is feed back into the step function. If ``out_memory``
        is not specified this must contain a node for *every* node in
        ``step_result`` because then the whole result will be fed back.
    out_memory: node/list(nodes)
        (optional) must be subset of ``step_result`` and of same length
        as ``in_memory``, tells which nodes of the result are fed back
        to ``in_memory``. If ``None``, all are fed back.
    in_iterate: node/list(nodes)
        nodes with a leading ``'r'`` axis to be iterated over (e.g.
        time series of shape [(30,r),(100,b),(50,f)]). In every step a slice
        from the first axis is consumed.
    in_iterate_0: node/list(nodes)
        nodes that consume a single slice of the ``in_iterate`` nodes.
        Part of "the inner function" of the scan loop in contrast to
        ``in_iterate``

    n_steps: int
    unroll_scan: bool
    last_only: bool
    name: str
    print_repr: bool
    """

    def __init__(self, step_result, in_memory, out_memory=None,
                 in_iterate=None, in_iterate_0=None, n_steps=None,
                 unroll_scan=True, last_only=False, name="scan", print_repr=True):
        step_result = utils.as_list(step_result)
        in_memory = utils.as_list(in_memory)
        out_memory = utils.as_list(out_memory)
        in_iterate = utils.as_list(in_iterate)
        in_iterate_0 = utils.as_list(in_iterate_0)

        if n_steps is None:
            assert in_iterate is not None
            if unroll_scan==True:
                n_steps = in_iterate[0].shape[0]
        else:
            if in_iterate is not None:
                assert n_steps == in_iterate[0].shape[0]

        if out_memory: assert len(in_memory)==len(out_memory)
        if in_iterate: assert in_iterate_0 and len(in_iterate)==len(in_iterate_0)
        if in_iterate_0: assert in_iterate
        if not out_memory: assert len(step_result)==len(in_memory)
        if out_memory:
            for o_m in out_memory: assert o_m in step_result
            #for o_m, i_m in zip(out_memory, in_memory): assert o_m.shape.shape == i_m.shape.shape
        else:
            for s_r, i_m in zip(step_result, in_memory): assert s_r.shape.shape == i_m.shape.shape

        parents = []
        for pl in [step_result, in_memory, in_iterate]:
            if pl is not None:
                parents.extend(pl)

        if out_memory:
            out_memory_sl = [step_result.index(s_r) for s_r in out_memory]
        else:
            out_memory_sl = list(range(len(step_result)))

        super(ScanN, self).__init__(parents, name, print_repr)
        self.step_result  = step_result
        self.in_memory    = in_memory
        self.out_memory   = out_memory
        self.out_memory_sl= out_memory_sl
        self.in_iterate   = in_iterate
        self.in_iterate_0 = in_iterate_0
        self._iterate     = in_iterate is not None

        self.n_steps      = n_steps
        self.unroll_scan  = unroll_scan
        self.last_only    = last_only
        self.output_names = ["scan_out_"+s_r.name for s_r in step_result]


    def _make_output(self):
        mem_hook = [i_m.output for i_m in self.in_memory]
        out_hook = [s_r.output for s_r in self.step_result]
        if self._iterate:
            it_hook = [i_o.output for i_o in self.in_iterate_0]

        if self.unroll_scan:
            out_accum = []
            # Doe one iteration as is
            new_out = [s_r.output for s_r in self.step_result]
            out_accum.append(new_out)
            # replace the memory input and iterate in every step
            for t in range(1, self.n_steps):
                new_mem = [new_out[i] for i in self.out_memory_sl]
                replacements = dict(zip(mem_hook, new_mem))
                if self._iterate:
                    for i_h, i_i in zip(it_hook, self.in_iterate):
                        replacements[i_h] = i_i.output[t]

                new_out = theano.clone(out_hook, replace=replacements)
                out_accum.append(new_out)
                # for k,v in replacements.items():
                #      print(" Replace ",k.auto_name,k," by ",v.auto_name,v)
                # print('---')

            if self.last_only:
                self.output = new_out
            else:
                self.output = [T.stack(r_accum, axis=0) for r_accum in
                               zip(*out_accum)]

        else:
            if self.n_steps:
                logger.warning("If the number of steps is known, "
                               "you should use 'unroll_scan'!")
            assert self._iterate
            l_it = len(it_hook)
            l_out = len(out_hook)
            l_mem = len(mem_hook)
            mem_map = []
            for s_r in self.step_result:
                try:
                    ix = self.out_memory.index(s_r)
                except:
                    ix = None
                mem_map.append(ix)

            outputs_info = []
            for i, s_r in enumerate(out_hook):
                if mem_map[i] is not None:
                    outputs_info.append(dict(initial=out_hook[mem_map[i]], taps=[-1]))
                else:
                    outputs_info.append(None)

            def step(*args):
                # args: input0 slices, output slices, non_seq
                it_slice     = args[0:l_it]
                mem_new_hook = args[l_it:l_it+l_mem]
                tmp = args[l_it+l_mem:]
                out_hook__   = tmp[0:l_out]
                it_hook__    = tmp[l_out:l_it+l_out]
                mem_hook__   = tmp[l_it+l_out:]
                replacements__ = dict()
                for i, m_i, in enumerate(out_hook__):
                    if mem_map[i] is not None:
                        ix = mem_map[i]
                        replacements__[mem_hook__[ix]] = mem_new_hook[i]

                for i_h, i_i in zip(it_hook__, it_slice): # consume the in_iterate_series slice
                    replacements__[i_h] = i_i

                new_outputs = theano.clone(out_hook__, replace=replacements__)
                updates__ = dict()
                #condition = theano.scan_module.until(False)
                return new_outputs, updates__ #, condition

            results, updates = theano.scan(step,
                                           sequences=[i_i.output[1:] for i_i in self.in_iterate],
                                           outputs_info=outputs_info,
                                           non_sequences=out_hook+it_hook+mem_hook)
            if self.last_only:
                self.output = [r[-1] for r in results]
            else:
                l = []
                results = utils.as_list(results)
                for r_0, r_accum in zip(out_hook, results):
                    pattern = ['x'] + list(range(r_0.ndim))
                    l.append(T.concatenate([r_0.dimshuffle(pattern), r_accum], axis=0))

                self.output = l



    def _calc_shape(self):
        if self.n_steps is None:
            n = self.in_iterate[0].shape[0]
        else:
            n = self.n_steps
        if self.last_only:
            self.shape = [s_r.shape.copy() for s_r in self.step_result]
        else:
            self.shape = [s_r.shape.addaxis(0, n, 'r') for s_r in self.step_result]


    def _calc_comp_cost(self):
        if self.n_steps is None:
            n = self.in_iterate[0].shape[0]
        else:
            n = self.n_steps

        self.computational_cost = (n-1) * self.step_result[0].all_computational_cost

def scansplit(scan, name='scan_split'):
    args = ()
    kwargs = dict(name=name)
    model_manager.current.register_split(scan, scansplit, name, args,
                                         kwargs)

    outs = []
    for out, out_sh, out_name in zip(scan.output, scan.shape, scan.output_names):
        outs.append(FromTensor(out, out_sh, scan,
                     name=out_name, print_repr=False))

    return outs


def Scan(step_result, in_memory, out_memory=None,
         in_iterate=None, in_iterate_0=None, n_steps=None,
         unroll_scan=True, last_only=False, name="scan", print_repr=True):
    """
    Parameters
    ----------
    step_result: node/list(nodes)
        nodes that represent results of step function
    in_memory: node/list(nodes)
        nodes that indicate at which place in the computational graph
        the memory is feed back into the step function. If ``out_memory``
        is not specified this must contain a node for *every* node in
        ``step_result`` because then the whole result will be fed back.
    out_memory: node/list(nodes)
        (optional) must be subset of ``step_result`` and of same length
        as ``in_memory``, tells which nodes of the result are fed back
        to ``in_memory``. If ``None``, all are fed back.
    in_iterate: node/list(nodes)
        nodes with a leading ``'r'`` axis to be iterated over (e.g.
        time series of shape [(30,r),(100,b),(50,f)]). In every step a slice
        from the first axis is consumed.
    in_iterate_0: node/list(nodes)
        nodes that consume a single slice of the ``in_iterate`` nodes.
        Part of "the inner function" of the scan loop in contrast to
        ``in_iterate``

    n_steps: int
    unroll_scan: bool
    last_only: bool
    name: str
    print_repr: bool

    Returns
    -------

    A node for every node in ``step_result`` which either contains the last
    state or the series of states - then it has a leading ``'r'`` axis.
    """
    scan = ScanN(step_result, in_memory, out_memory, in_iterate, in_iterate_0,
                 n_steps, unroll_scan=unroll_scan, last_only=last_only,
                 name=name, print_repr=print_repr)

    outputs = scansplit(scan, name='scan_split')
    if len(outputs)==1:
        outputs = outputs[0]

    return outputs


class Reshape(Node):
    """
    Reshape node.

    Parameters
    ----------
    parent
    shape
    tags
    strides
    fov
    name
    print_repr
    """
    def __init__(self, parent, shape, tags=None, strides=None, fov=None, name="reshape", print_repr=True):

        super(Reshape, self).__init__(parent, name, print_repr)
        self.parent   = parent
        if isinstance(shape, TaggedShape):
            self._shape = shape
        else:
            if tags is None:
                raise ValueError("Tags argument must not be None if "
                                 "shape is not a TaggedShape")
            self._shape = TaggedShape(shape, tags, strides, fov=fov)

        if self._shape.stripbatch_prod != parent.shape.stripbatch_prod:
            raise ValueError("Cannot reshap %s to %s" %(parent.shape, self._shape))

    def _make_output(self):
        new_sh = [x if x is not None else -1 for x in self._shape]
        self.output = self.parent.output.reshape(new_sh)

    def _calc_shape(self):
        self.shape = self._shape

    def _calc_comp_cost(self):
        self.computational_cost = 0


_SkeletonGetBatch = SkelGetBatchN
_skeletongetbatchsplit = skelgetbatch_split
SkeletonLossRec = SkelLossRec
SkeletonPrior = SkelPrior
_Scan = ScanN
_scansplit = scansplit
_SkeletonLoss = SkelLoss

if __name__=="__main__":
    # test wrapping of generic type into graph
    class DummySkel(object):
        def __init__(self):
            self.val = 3.141
            self.grad = np.array([1, 3, 5], dtype=np.float32)

        def get_loss(self, predicted):
            # print("I'm here %s" %predicted)
            return self.val, self.grad * predicted

    skel_var = gof.type.Generic()()
    x = T.TensorType(dtype='float32', broadcastable=(False,))()
    w = theano.shared(np.eye(3, dtype=np.float32))
    pred_var = theano.dot(w, x)

    loss_var, loss_grad = SkelLossOP()(pred_var, skel_var)

    f = theano.function([x, skel_var], [loss_var])
    grad_var = theano.grad(loss_var, [w, ])
    f_grad = theano.function([x, skel_var], grad_var + [loss_var, pred_var])

    test_vals = np.array([1, 2, 3], dtype=np.float32)
    skel = DummySkel()

    print(f(test_vals, skel))
    print(f_grad(test_vals, skel))
    theano.printing.pydotprint(f_grad, '/tmp/step_result.svg')

    ###########################################################################

    from elektronn2 import neuromancer
    import theano
    act = 'tanh'
    data = neuromancer.Input((1, 20), 'b,f', name='data')
    mem_0 = neuromancer.Input((1, 120), 'b,f', name='mem')
    mlp1 = neuromancer.Dot(data, 120, activation_func=act)
    join = neuromancer.Concat([mlp1, mem_0])
    out = neuromancer.Dot(join, 120, activation_func=act)
    out2 = neuromancer.Dot(out, 13, activation_func='lin')
    # recurrent    = neuromancer.Scan(out, in_memory=mem_0, n_steps=10)
    recurrent, out2r = neuromancer.Scan([out, out2], out_memory=out,
                                        in_memory=mem_0, n_steps=7)
    loss = neuromancer.AggregateLoss(recurrent)
    loss = neuromancer.AggregateLoss(out2r)

    grad = theano.grad(loss.output, loss.all_trainable_params.values(),
                       disconnected_inputs='warn')

    recurrent()
    out2r()
    print(len(list(filter(lambda x: x.op.__class__.__name__=="Dot22",
                    recurrent._output_func.func.maker.fgraph.apply_nodes))))

    theano.printing.pydotprint(recurrent._output_func.func,
                               outfile="/tmp/test-comp.png",
                               var_with_name_simple=True)
    theano.printing.pydotprint(out2r._output_func.func,
                               outfile="/tmp/test2-comp.png",
                               var_with_name_simple=True)
    fn = theano.function(out.input_tensors, [recurrent.output, out2r.output])
    x = np.random.rand(1, 20).astype(np.float32)
    m = np.random.rand(1, 120).astype(np.float32)
    y = recurrent(x, m)
    z = out2r(x, m)
    grad_fn = theano.function(out.input_tensors, grad)
    g = grad_fn(x, m)
