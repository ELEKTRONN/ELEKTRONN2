# -*- coding: utf-8 -*-
# ELEKTRONN2 Toolkit
# Copyright (c) 2015 Marius Killinger
# All rights reserved
from __future__ import absolute_import, division, print_function
from builtins import filter, hex, input, int, map, next, oct, pow, range, super, zip


import numpy as np
import theano
from theano import gof
import theano.tensor as T
from theano.gradient import disconnected_type

from . import malis_utils

__all__ = ['malis_weights']

class MalisWeights(theano.Op):
    """
    Computes MALIS loss weights
    
    Roughly speaking the malis weights quantify the impact of an edge in
    the predicted affinity graph on the resulting segmentation.
      
    Parameters
    ----------
    affinity_pred: 4d np.ndarray float32
        Affinity graph of shape (#edges, x, y, z)
        1: connected, 0: disconnected        
    affinity_gt: 4d np.ndarray int16
        Affinity graph of shape (#edges, x, y, z)
        1: connected, 0: disconnected 
    seg_gt: 3d np.ndarray, int (any precision)
        Volume of segmentation IDs        
    nhood: 2d np.ndarray, int
        Neighbourhood pattern specifying the edges in the affinity graph
        Shape: (#edges, ndim)
        nhood[i] contains the displacement coordinates of edge i
        The number and order of edges is arbitrary
    unrestrict_neg: Bool
        Use this to relax the restriction on neg_counts. The restriction
        modifies the edge weights for before calculating the negative counts
        as: ``edge_weights_neg = np.maximum(affinity_pred, affinity_gt)``
        If unrestricted the predictions are used directly.
        
    Returns
    -------
      
    pos_counts: 4d np.ndarray int32
      Impact counts for edges that should be 1 (connect)      
    neg_counts: 4d np.ndarray int32
      Impact counts for edges that should be 0 (disconnect)  
      
      
    Outline
    -------
      
    - Computes for all pixel-pairs the MaxiMin-Affinity
    - Separately for pixel-pairs that should/should not be connected
    - Every time an affinity prediction is a MaxiMin-Affinity its weight is
      incremented by one in the output matrix (in different slices depending
      on whether that that pair should/should not be connected)
    """
    __props__ = ()

    def make_node(self, *inputs):
        inputs = list(inputs)
        if len(inputs)!=5:
            raise ValueError("MalisOp takes 5 inputs: \
          affinity_pred, affinity_gt, seg_gt, nhood, unrestrict_neg")
        
        inputs = list(map(T.as_tensor_variable, inputs))
        
        affinity_pred, affinity_gt, seg_gt, nhood = inputs[:4]
        if affinity_pred.ndim!=4:
            raise ValueError("affinity_pred must be convertible to a\
          TensorVariable of dimensionality 4. This one has ndim=%i" \
                             %affinity_pred.ndim)

        if affinity_gt.ndim!=4:
            raise ValueError("affinity_gt must be convertible to a\
          TensorVariable of dimensionality 4. This one has ndim=%i" \
                             %affinity_gt.ndim)

        if seg_gt.ndim!=3:
            raise ValueError("seg_gt must be convertible to a\
          TensorVariable of dimensionality 3. This one has ndim=%i" \
                             %seg_gt.ndim)

        if nhood.ndim!=2:
            raise ValueError("nhood must be convertible to a\
          TensorVariable of dimensionality 2. This one has ndim=%i" \
                             %nhood.ndim)
                                     
        malis_weights_pos = T.TensorType(
            dtype='uint64',
            broadcastable=(False,)*4)()

        malis_weights_neg = T.TensorType(
            dtype='uint64',
            broadcastable=(False,)*4)()

        outputs = [malis_weights_pos, malis_weights_neg]

        return gof.Apply(self, inputs, outputs)

    def perform(self, node, inputs, output_storage):
        affinity_pred, affinity_gt, seg_gt, nhood, unrestrict_neg = inputs
        pos, neg = output_storage
        pos[0], neg[0] = malis_utils.malis_weights(affinity_pred,
                                                   affinity_gt,
                                                   seg_gt,
                                                   nhood,
                                                   unrestrict_neg)

    def grad(self, inputs, outputs_gradients):
        # The gradient of all outputs is 0 w.r.t. to all inputs
        return [disconnected_type(),]*5

    def connection_pattern(self, node):
        # The gradient of all outputs is 0 w.r.t. to all inputs
        return [[False, False],]*5

def malis_weights(affinity_pred, affinity_gt, seg_gt, nhood, unrestrict_neg=False):
    """
    Computes MALIS loss weights
    
    Roughly speaking the malis weights quantify the impact of an edge in
    the predicted affinity graph on the resulting segmentation.
      
    Parameters
    ----------
    affinity_pred: 4d np.ndarray float32
        Affinity graph of shape (#edges, x, y, z)
        1: connected, 0: disconnected        
    affinity_gt: 4d np.ndarray int16
        Affinity graph of shape (#edges, x, y, z)
        1: connected, 0: disconnected 
    seg_gt: 3d np.ndarray, int (any precision)
        Volume of segmentation IDs        
    nhood: 2d np.ndarray, int
        Neighbourhood pattern specifying the edges in the affinity graph
        Shape: (#edges, ndim)
        nhood[i] contains the displacement coordinates of edge i
        The number and order of edges is arbitrary
    unrestrict_neg: Bool
        Use this to relax the restriction on neg_counts. The restriction
        modifies the edge weights for before calculating the negative counts
        as: ``edge_weights_neg = np.maximum(affinity_pred, affinity_gt)``
        If unrestricted the predictions are used directly.
        
    Returns
    -------
      
    pos_counts: 4d np.ndarray int32
      Impact counts for edges that should be 1 (connect)      
    neg_counts: 4d np.ndarray int32
      Impact counts for edges that should be 0 (disconnect)  
      
      
    Outline
    -------
      
    - Computes for all pixel-pairs the MaxiMin-Affinity
    - Separately for pixel-pairs that should/should not be connected
    - Every time an affinity prediction is a MaxiMin-Affinity its weight is
      incremented by one in the output matrix (in different slices depending
      on whether that that pair should/should not be connected)
    """
    rest = 1 if unrestrict_neg else 0 # Theano cannot bool        
    return MalisWeights()(affinity_pred, affinity_gt, seg_gt, nhood, rest)
