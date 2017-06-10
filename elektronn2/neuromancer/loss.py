# -*- coding: utf-8 -*-
# ELEKTRONN2 Toolkit
# Copyright (c) 2015 Marius Killinger
# All rights reserved

from __future__ import absolute_import, division, print_function
from builtins import filter, hex, input, int, map, next, oct, pow, range, super, zip

import logging

import numpy as np
import theano.tensor as T

from .computations import softmax
from .graphutils import TaggedShape, floatX
from .node_basic import Node, FromTensor
from .variables import VariableParam
from .neural import Conv

logger = logging.getLogger('elektronn2log')
inspection_logger = logging.getLogger('elektronn2log-inspection')

__all__ = ['GaussianNLL', 'BinaryNLL', 'AggregateLoss', 'SquaredLoss',
           'AbsLoss',
           'Softmax', 'MultinoulliNLL', 'MalisNLL', 'Errors', 'BetaNLL',
           'SobelizedLoss', 'BlockedMultinoulliNLL', 'OneHot',
           'EuclideanDistance', 'RampLoss']

xlogy0 = T.xlogx.xlogy0
EPS = 1e-5


class Softmax(Node):
    """
    Softmax node.

    Parameters
    ----------
    parent: Node
        Input node.
    n_class
    n_indep
    name: str
        Node name.
    print_repr: bool
        Whether to print the node representation upon initialisation.
    """

    def __init__(self, parent, n_class='auto', n_indep=1, name="softmax",
                 print_repr=True):

        super(Softmax, self).__init__(parent, name, print_repr)

        n_f = parent.shape['f']

        if hasattr(parent, 'activation_func'):
            if parent.activation_func != 'lin':
                raise ValueError("The parent of a Softmax-node must have a "
                                 "linear activation function.")

        if n_class == 'auto':
            if n_f % n_indep == 0:
                n_class = n_f // n_indep
            else:
                n_class = n_f // n_indep
                raise ValueError("Cannot create %i-fold %i-class softmax "
                                 "from %i features." % (n_indep, n_class, n_f))
        else:
            if n_class * n_indep != n_f:
                raise ValueError("Cannot create %i-fold %i-class softmax ")

        self.n_class = n_class
        self.n_indep = n_indep

    def _make_output(self):
        """ Computation of Theano Output """
        n_class = self.n_class
        n_indep = self.n_indep
        x = self.parent.output
        axis = self.parent.shape.tag2index('f')

        if self.n_indep == 1:
            self.output = softmax(x, axis=axis)
        else:
            y = []
            for i in range(n_indep):
                sl = [slice(None), ] * x.ndim
                sl[axis] = slice(i * n_class, (i + 1) * n_class, 1)
                y_part = softmax(x[tuple(sl)], axis=axis)
                y.append(y_part)

            y = T.concatenate(y, axis=axis)
            self.output = y


class OneHot(Node):
    """
    Onehot node.

    Parameters
    ----------
    target: T.Tensor
        Target tensor.
    n_class: int
    axis
    name: str
        Node name.
    print_repr: bool
        Whether to print the node representation upon initialisation.
    """

    def __init__(self, target, n_class, axis='f', name="onehot",
                 print_repr=True):
        super(OneHot, self).__init__(target, name, print_repr)

        self.target = target
        self.axis = target.shape.tag2index('f')
        self.n_class = n_class

    def _make_output(self):
        """ Computation of Theano Output """
        target = self.target.output

        pattern_exp_class = ['x', ] * target.ndim
        pattern_exp_class[self.axis] = 0

        classes = T.arange(self.n_class)
        classes = classes.dimshuffle(pattern_exp_class)

        target = T.addbroadcast(target, self.axis)
        target = T.eq(target, classes)  # to 1-hot
        target = T.cast(target, floatX)

        self.output = target

    def _calc_shape(self):
        sh = self.parent.shape.updateshape(self.axis, self.n_class)
        self.shape = sh


class MultinoulliNLL(Node):
    """
    Returns the symbolic mean and instance-wise negative log-likelihood of the prediction
    of this model under a given target distribution.

    Parameters
    ----------
    pred: Node
        Prediction node.
    target: T.Tensor
        corresponds to a vector that gives the correct label for each example. Labels < 0 are ignored (e.g. can
        be used for label propagation).
    target_is_sparse: bool
        If the target is sparse.
    class_weights: T.Tensor
        weight vector of float32 of length  ``n_lab``. Values: ``1.0`` (default), ``w < 1.0`` (less important),
        ``w > 1.0`` (more important class).
    example_weights: T.Tensor
        weight vector of float32 of shape ``(bs, z, x, y)`` that can give the individual examples (i.e. labels for
        output pixels) different weights. Values: ``1.0`` (default), ``w < 1.0`` (less important),
        ``w > 1.0`` (more important example). Note: if this is not normalised/bounded it may result in a
        effectively modified learning rate!

    The following refers to lazy labels, the masks are always on a per patch basis, depending on the
    origin cube of the patch. The masks are properties of the individual image cubes and must be loaded
    into CNNData.

    mask_class_labeled: T.Tensor
        shape = (batchsize, num_classes).
        Binary masks indicating whether a class is properly labeled in ``y``. If a class ``k``
        is (in general) present in the image patches **and** ``mask_class_labeled[k]==1``, then
        the labels  **must** obey ``y==k`` for all pixels where the class is present.
        If a class ``k`` is present in the image, but was not labeled (-> cheaper labels), set
        ``mask_class_labeled[k]=0``. Then all pixels for which the ``y==k`` will be ignored.
        Alternative: set ``y=-1`` to ignore those pixels.
        Limit case: ``mask_class_labeled[:]==1`` will result in the ordinary NLL.
    mask_class_not_present: T.Tensor
        shape = (batchsize, num_classes).
        Binary mask indicating whether a class is present in the image patches.
        ``mask_class_not_present[k]==1`` means that the image does **not** contain examples of class ``k``.
        Then for all pixels in the patch, class ``k`` predictive probabilities are trained towards ``0``.
        Limit case: ``mask_class_not_present[:]==0`` will result in the ordinary NLL.
    name: str
        Node name.
    print_repr: bool
        Whether to print the node representation upon initialisation.

    Examples
    --------

    - A cube contains no class ``k``. Instead of labelling the remaining classes they can be
      marked as unlabelled by the first mask (``mask_class_labeled[:]==0``, whether ``mask_class_labeled[k]``
      is ``0`` or ``1`` is actually indifferent because the labels should not be ``y==k`` anyway in this case).
      Additionally ``mask_class_not_present[k]==1`` (otherwise ``0``) to suppress predictions of ``k`` in
      in this patch. The actual value of the labels is indifferent, it can either be ``-1`` or it could be the
      background class, if the background is marked as unlabelled (i.e. then those labels are ignored).

    - Only part of the cube is densely labelled. Set ``mask_class_labeled[:]=1`` for all classes, but set the
      label values in the unlabelled part to ``-1`` to ignore this part.

    - Only a particular class ``k`` is labelled in the cube. Either set all other label pixels to ``-1`` or the
      corresponding flags in ``mask_class_labeled`` for the unlabelled classes.

    ..  Note::
        Using ``-1`` labels or telling that a class is not labelled, is somewhat redundant and just
        supported for convenience.
    """
    # TODO: add comment on normalisation.

    def __init__(self, pred, target, target_is_sparse=False, class_weights=None,
                 example_weights=None, mask_class_labeled=None,
                 mask_class_not_present=None, name="nll", print_repr=True):
        parents = [pred, target]
        if class_weights is not None:
            if isinstance(class_weights, Node):
                parents.append(class_weights)
            else:
                class_weights = np.array(class_weights, dtype=floatX)
                class_weights = VariableParam(value=class_weights,
                                              name="class_weights",
                                              dtype=floatX,
                                              apply_train=False)
        if example_weights is not None:
            parents.append(example_weights)
        if mask_class_labeled is not None:
            parents.append(mask_class_labeled)
        if mask_class_not_present is not None:
            parents.append(mask_class_not_present)

        super(MultinoulliNLL, self).__init__(parents, name, print_repr)

        if isinstance(pred, Softmax):
            parent = pred
        else:
            if isinstance(pred, FromTensor) and isinstance(pred.parent,
                                                           Softmax):
                parent = pred.parent  # split softmax...
            else:
                raise ValueError(
                    "The prob input to a MultinoulliNLL-node must be "
                    "a Softmax-Node.")

        self.target = target
        self.pred = pred
        self.axis = pred.shape.tag2index('f')
        self.n_class = parent.n_class
        self.n_indep = parent.n_indep
        self.target_is_sparse = target_is_sparse
        self.class_weights = class_weights
        self.example_weights = example_weights
        self.mask_class_labeled = mask_class_labeled
        self.mask_class_not_present = mask_class_not_present

    def _make_output(self):
        """ Computation of Theano Output """
        pred = self.pred.output
        target = self.target.output

        pattern_add_class = list(range(pred.ndim - 1))
        pattern_add_class.insert(self.axis, 'x')

        pattern_exp_class = ['x', ] * pred.ndim
        pattern_exp_class[self.axis] = 0

        if self.target_is_sparse:  # convert to 1-hot probabilistic like coding
            classes = T.arange(self.n_class)
            classes = classes.dimshuffle(pattern_exp_class)
            if self.n_indep == 1:  # assuming target (b, ...)
                # target = target.dimshuffle(pattern_add_class)
                target = T.addbroadcast(target, self.axis)
                target = T.eq(target, classes)  # to 1-hot
            else:  # assuming target (b, n_indep, ...)
                t = []
                for i in range(self.n_indep):
                    component = target[:, i:i + 1]
                    component = T.addbroadcast(component, self.axis)
                    t.append(T.eq(component, classes))
                target = T.concatenate(t, axis=self.axis)

        # Target is now a 1-hot encoded bool of shape pred.shape

        if self.class_weights is None:
            class_weights = 1
        else:
            if isinstance(self.class_weights, Node):
                class_weights = self.class_weights.output
            else:
                class_weights = self.class_weights

            class_weights = class_weights.dimshuffle(pattern_exp_class)
            assert class_weights.ndim == pred.ndim

        if self.example_weights is None:
            example_weights = 1
        else:
            example_weights = self.example_weights.output
            example_weights = example_weights.dimshuffle(pattern_add_class)
            assert example_weights.ndim == pred.ndim
        if self.mask_class_labeled is not None:
            m_pattern = ['x', ] * pred.ndim
            m_pattern[self.axis] = 0
            m_pattern[self.pred.shape.tag2index('b')] = 1
            mask_class_labeled = self.mask_class_labeled.output.dimshuffle(
                m_pattern)

            target = target * mask_class_labeled  # this excludes some classes
            # in target (set their row to 0)

        nll_up = -xlogy0(target * class_weights * example_weights, pred + EPS)
        n_labelled_up = target.sum()

        if self.mask_class_not_present is not None:
            m_pattern = ['x', ] * pred.ndim
            m_pattern[self.axis] = 1
            m_pattern[self.pred.shape.tag2index('b')] = 0
            # Expand the mask to the full size, because below we want to sum it
            mask_class_not_present = self.mask_class_not_present.output. \
                                         dimshuffle(m_pattern) * T.ones_like(
                target)
            nll_dn = -xlogy0(
                mask_class_not_present * class_weights * example_weights,
                1.0 - pred + EPS)
            n_labelled_dn = mask_class_not_present.sum()

        else:
            nll_dn = 0.0
            n_labelled_dn = 0.0

        # Scale by n_labelled and n_indep
        # because the x-entropy is the sum across the classes, but this sum
        # is not taken here (so the pred.size is n_class times to big, when
        # given to AggregateLoss)
        n_tot = n_labelled_up + n_labelled_dn
        nll = (nll_up + nll_dn) * pred.size / (
        n_tot + EPS) / self.n_indep / self.n_class
        nll = T.sum(nll, axis=self.axis, keepdims=True)
        self._debug_outputs.extend([n_tot, pred.size])
        self.output = nll

    def _calc_shape(self):
        sh = self.parent[0].shape.updateshape(self.axis, 1)
        self.shape = sh


class BlockedMultinoulliNLL(Node):
    """
    Returns the symbolic mean and instance-wise negative log-likelihood of the prediction
    of this model under a given target distribution.

    Parameters
    ----------
    pred: Node
        Prediction node.
    target: T.Tensor
        corresponds to a vector that gives the correct label for each example. Labels < 0 are ignored (e.g. can
        be used for label propagation).
    blocking_factor: float
        Blocking factor.
    target_is_sparse: bool
        If the target is sparse.
    class_weights: T.Tensor
        weight vector of float32 of length  ``n_lab``. Values: ``1.0`` (default), ``w < 1.0`` (less important),
        ``w > 1.0`` (more important class).
    example_weights: T.Tensor
        weight vector of float32 of shape ``(bs, z, x, y)`` that can give the individual examples (i.e. labels for
        output pixels) different weights. Values: ``1.0`` (default), ``w < 1.0`` (less important),
        ``w > 1.0`` (more important example). Note: if this is not normalised/bounded it may result in a
        effectively modified learning rate!

    The following refers to lazy labels, the masks are always on a per patch basis, depending on the
    origin cube of the patch. The masks are properties of the individual image cubes and must be loaded
    into CNNData.

    mask_class_labeled: T.Tensor
        shape = (batchsize, num_classes).
        Binary masks indicating whether a class is properly labeled in ``y``. If a class ``k``
        is (in general) present in the image patches **and** ``mask_class_labeled[k]==1``, then
        the labels  **must** obey ``y==k`` for all pixels where the class is present.
        If a class ``k`` is present in the image, but was not labeled (-> cheaper labels), set
        ``mask_class_labeled[k]=0``. Then all pixels for which the ``y==k`` will be ignored.
        Alternative: set ``y=-1`` to ignore those pixels.
        Limit case: ``mask_class_labeled[:]==1`` will result in the ordinary NLL.
    mask_class_not_present: T.Tensor
        shape = (batchsize, num_classes).
        Binary mask indicating whether a class is present in the image patches.
        ``mask_class_not_present[k]==1`` means that the image does **not** contain examples of class ``k``.
        Then for all pixels in the patch, class ``k`` predictive probabilities are trained towards ``0``.
        Limit case: ``mask_class_not_present[:]==0`` will result in the ordinary NLL.
    name: str
        Node name.
    print_repr: bool
        Whether to print the node representation upon initialisation.

    Examples
    --------
    - A cube contains no class ``k``. Instead of labelling the remaining classes they can be
      marked as unlabelled by the first mask (``mask_class_labeled[:]==0``, whether ``mask_class_labeled[k]``
      is ``0`` or ``1`` is actually indifferent because the labels should not be ``y==k`` anyway in this case).
      Additionally ``mask_class_not_present[k]==1`` (otherwise ``0``) to suppress predictions of ``k`` in
      in this patch. The actual value of the labels is indifferent, it can either be ``-1`` or it could be the
      background class, if the background is marked as unlabelled (i.e. then those labels are ignored).

    - Only part of the cube is densely labelled. Set ``mask_class_labeled[:]=1`` for all classes, but set the
      label values in the unlabelled part to ``-1`` to ignore this part.

    - Only a particular class ``k`` is labelled in the cube. Either set all other label pixels to ``-1`` or the
      corresponding flags in ``mask_class_labeled`` for the unlabelled classes.

    ..  Note::
        Using ``-1`` labels or telling that a class is not labelled, is somewhat redundant and just
        supported for convenience.
    """

    def __init__(self, pred, target, blocking_factor=0.5,
                 target_is_sparse=False, class_weights=None,
                 example_weights=None, mask_class_labeled=None,
                 mask_class_not_present=None, name="nll", print_repr=True):
        ###TODO add comment on normalisation
        parents = [pred, target]
        if class_weights is not None:
            parents.append(class_weights)
        if example_weights is not None:
            parents.append(example_weights)
        if mask_class_labeled is not None:
            parents.append(mask_class_labeled)
        if mask_class_not_present is not None:
            parents.append(mask_class_not_present)

        super(BlockedMultinoulliNLL, self).__init__(parents, name, print_repr)

        if isinstance(pred, Softmax):
            parent = pred
        else:
            if isinstance(pred, FromTensor) and isinstance(pred.parent,
                                                           Softmax):
                parent = pred.parent  # split softmax...
            else:
                raise ValueError(
                    "The prob input to a MultinoulliNLL-node must be "
                    "a Softmax-Node.")

        self.target = target
        self.pred = pred
        self.axis = pred.shape.tag2index('f')
        self.n_class = parent.n_class
        self.n_indep = parent.n_indep
        self.target_is_sparse = target_is_sparse
        self.class_weights = class_weights
        self.example_weights = example_weights
        self.mask_class_labeled = mask_class_labeled
        self.mask_class_not_present = mask_class_not_present
        self.blocking_factor = VariableParam(blocking_factor,
                                             name='blocking_factor',
                                             apply_train=False,
                                             apply_reg=False)

    def _make_output(self):
        """ Computation of Theano Output """
        pred = self.pred.output
        target = self.target.output

        pattern_add_class = list(range(pred.ndim - 1))
        pattern_add_class.insert(self.axis, 'x')

        pattern_exp_class = ['x', ] * pred.ndim
        pattern_exp_class[self.axis] = 0

        if self.target_is_sparse:  # convert to 1-hot probabilistic like coding
            classes = T.arange(self.n_class)
            classes = classes.dimshuffle(pattern_exp_class)
            if self.n_indep == 1:  # assuming target (b, ...)
                # target = target.dimshuffle(pattern_add_class)
                target = T.addbroadcast(target, self.axis)
                target = T.eq(target, classes)  # to 1-hot
            else:  # assuming target (b, n_indep, ...)
                t = []
                for i in range(self.n_indep):
                    component = target[:, i:i + 1]
                    component = T.addbroadcast(component, self.axis)
                    t.append(T.eq(component, classes))
                target = T.concatenate(t, axis=self.axis)

        # Target is now a 1-hot encoded bool of shape pred.shape

        if self.class_weights is None:
            class_weights = 1
        else:
            class_weights = self.class_weights.output
            class_weights = class_weights.dimshuffle(pattern_exp_class)
            assert class_weights.ndim == pred.ndim
        if self.example_weights is None:
            example_weights = 1
        else:
            example_weights = self.example_weights.output
            example_weights = example_weights.dimshuffle(pattern_add_class)
            assert example_weights.ndim == pred.ndim
        if self.mask_class_labeled is not None:
            m_pattern = ['x', ] * pred.ndim
            m_pattern[self.axis] = 0
            m_pattern[self.pred.shape.tag2index('b')] = 1
            mask_class_labeled = self.mask_class_labeled.output.dimshuffle(
                m_pattern)

            target = target * mask_class_labeled  # this excludes some classes
            # in target (set their row to 0)

        # Blocking
        b_pattern = [slice(None)] * pred.ndim
        b_pattern[self.axis] = slice(1, None)
        new_pred = T.maximum(
            self.blocking_factor * pred[b_pattern].max(axis=self.axis),
            pred[b_pattern])
        T.set_subtensor(pred[b_pattern], new_pred)

        nll_up = -xlogy0(target * class_weights * example_weights, pred + EPS)
        n_labelled_up = target.sum()

        if self.mask_class_not_present is not None:
            m_pattern = ['x', ] * pred.ndim
            m_pattern[self.axis] = 1
            m_pattern[self.pred.shape.tag2index('b')] = 0
            # Expand the mask to the full size, because below we want to sum it
            mask_class_not_present = self.mask_class_not_present.output. \
                                         dimshuffle(m_pattern) * T.ones_like(
                target)
            nll_dn = -xlogy0(
                mask_class_not_present * class_weights * example_weights,
                1.0 - pred + EPS)
            n_labelled_dn = mask_class_not_present.sum()

        else:
            nll_dn = 0.0
            n_labelled_dn = 0.0

        # Scale by n_labelled and n_indep
        # because the x-entropy is the sum across the classes, but this sum
        # is not taken here (so the pred.size is n_class times to big, when
        # given to AggregateLoss)
        n_tot = n_labelled_up + n_labelled_dn
        nll = (nll_up + nll_dn) * pred.size / (
        n_tot + EPS) / self.n_indep / self.n_class
        nll = T.sum(nll, axis=self.axis, keepdims=True)
        self._debug_outputs.extend([new_pred, n_tot, pred.size])
        self.output = nll

    def _calc_shape(self):
        sh = self.parent[0].shape.updateshape(self.axis, 1)
        self.shape = sh


class MalisNLL(Node):
    """
    Malis NLL node. (See https://github.com/TuragaLab/malis)

    Parameters
    ----------
    pred: Node
        Prediction node.
    aff_gt: T.Tensor
    seg_gt: T.Tensor
    nhood: np.ndarray
    unrestrict_neg: bool
    class_weights: T.Tensor
        weight vector of float32 of length  ``n_lab``. Values: ``1.0`` (default), ``w < 1.0`` (less important),
        ``w > 1.0`` (more important class).
    example_weights: T.Tensor
        weight vector of float32 of shape ``(bs, z, x, y)`` that can give the individual examples (i.e. labels for
        output pixels) different weights. Values: ``1.0`` (default), ``w < 1.0`` (less important),
        ``w > 1.0`` (more important example). Note: if this is not normalised/bounded it may result in a
        effectively modified learning rate!
    name: str
        Node name.
    print_repr: bool
        Whether to print the node representation upon initialisation.
    """

    def __init__(self, pred, aff_gt, seg_gt, nhood, unrestrict_neg=True,
                 class_weights=None, example_weights=None,
                 name="nll", print_repr=True):
        parents = [pred, aff_gt, seg_gt]
        if class_weights is not None:
            parents.append(class_weights)
        if example_weights is not None:
            parents.append(example_weights)

        super(MalisNLL, self).__init__(parents, name, print_repr)

        if not isinstance(pred, Softmax):
            raise ValueError("The prob input to a MultinoulliNLL-node must be "
                             "a Softmax-Node.")
        if pred.shape['b'] != 1:
            raise NotImplementedError(
                "Malis can only be used with batch size 1.")

        self.aff_gt = aff_gt
        self.seg_gt = seg_gt
        self.pred = pred
        self.nhood = np.asarray(nhood, dtype=np.int32)
        self.unrestrict_neg = unrestrict_neg
        self.axis = pred.shape.tag2index('f')
        self.n_class = pred.n_class
        self.n_indep = pred.n_indep
        self.class_weights = class_weights
        self.example_weights = example_weights

    def _make_output(self):
        """ Computation of Theano Output """

        from ..malis.malisop import malis_weights

        pred = self.pred.output
        aff_gt = self.aff_gt.output[0]  # strip batch (1)
        seg_gt = self.seg_gt.output[0, 0]  # strip batch (1) and #class (1)

        pattern_add_class = list(range(pred.ndim - 1))
        pattern_add_class.insert(self.axis, 'x')

        pattern_exp_class = ['x', ] * pred.ndim
        pattern_exp_class[self.axis] = 0
        if self.class_weights is None:
            class_weights = 1
        else:
            class_weights = self.class_weights.output
            class_weights = class_weights.dimshuffle(pattern_exp_class)[
                0]  # strip batch dimension
            assert class_weights.ndim == pred[0].ndim
        if self.example_weights is None:
            example_weights = 1
        else:
            example_weights = self.example_weights.output
            example_weights = example_weights.dimshuffle(pattern_add_class)[
                0]  # strip batch dimension
            assert example_weights.ndim == pred[0].ndim

        sl = [slice(None), ] * pred.ndim
        sl[self.axis] = slice(1, None, self.n_class)
        # pred.shape = (bs, 6, x, y, z) 6--> edge1 neg, edge1 pos, edge2 neg...
        affinity_pred = pred[tuple(sl)][0]  # strip batch dimension

        sl = [slice(None), ] * pred.ndim
        sl[self.axis] = slice(0, None, self.n_class)
        disconnect_pred = pred[tuple(sl)][0]  # strip batch dimension

        pos_count, neg_count = malis_weights(affinity_pred,
                                             aff_gt,
                                             seg_gt,
                                             self.nhood,
                                             self.unrestrict_neg)

        pos_weight = pos_count * example_weights * class_weights
        neg_weight = neg_count * example_weights * class_weights
        weighted_pos = xlogy0(pos_weight,
                              affinity_pred + EPS)  # drive up prediction for "connected" here
        weighted_neg = xlogy0(neg_weight,
                              disconnect_pred + EPS)  # drive down prediction for "disconnected" here
        n_pos = T.sum(pos_count)
        n_neg = T.sum(neg_count)
        n_tot = n_pos + n_neg
        nll = -(weighted_pos + weighted_neg)
        # Scale by n_tot, because the counts n_tot are greater
        # than pred.size (~N**2), but the actual value depends on the amount
        # of ECS in the example
        self.output = nll * T.cast(nll.size, 'float32') / (n_tot + EPS)

        # For debug/inspection, take care that those are not in self.output
        false_splits = T.sum((affinity_pred < 0.5) * pos_count)
        false_merges = T.sum((affinity_pred > 0.5) * neg_count)
        rand_index = T.cast(false_splits + false_merges, 'float32') / (
        n_tot + EPS)
        self.rand_index = rand_index
        self.false_splits = false_splits
        self.false_merges = false_merges
        self.pos_count = pos_count
        self.neg_count = neg_count

        # eg 0.0   5187779 4578211 9765990 3439497477 7732598379 1143.9798583984375)
        # return nll, n_pos, n_neg, n_tot, false_splits, false_merges, rand_index, pos_count, neg_count

    def _calc_shape(self):
        sh = self.parent[0].shape.updateshape(self.axis, 1)
        self.shape = sh


class Classification(Node):
    """
    Classification node.

    Parameters
    ----------
    pred: Node
        Prediction node.
    n_class
    n_indep
    name: str
        Node name.
    print_repr: bool
        Whether to print the node representation upon initialisation.
    """

    def __init__(self, pred, n_class='auto', n_indep='auto', name="cls",
                 print_repr=True):
        super(Classification, self).__init__(pred, name, print_repr)
        if not isinstance(pred, Softmax):
            if pred.activation_func in ['sig', 'logistic', 'sigmoid']:
                self.n_class = 2
                self.n_indep = pred.shape['f']
                self.sm_input = False
            else:
                assert n_class != 'auto'
                assert n_indep != 'auto'
                self.n_class = n_class
                self.n_indep = n_indep
                self.sm_input = n_indep != pred.shape['f']
        else:  # pred is softmax node
            self.n_class = pred.n_class
            self.n_indep = pred.n_indep
            self.sm_input = True

        self.pred = pred

    def _make_output(self):
        """ Computation of Theano Output """
        n_class = self.n_class
        n_indep = self.n_indep
        pred = self.pred.output
        axis = self.pred.shape.tag2index('f')
        if self.sm_input:
            if self.n_indep == 1:
                cls = T.argmax(pred, axis=axis, keepdims=True)

            else:
                y = []
                for i in range(n_indep):
                    sl = [slice(None), ] * pred.ndim
                    sl[axis] = slice(i * n_class, (i + 1) * n_class, 1)
                    cls = T.argmax(pred[tuple(sl)], axis=axis, keepdims=True)
                    y.append(cls)

                cls = T.concatenate(y, axis=axis)
        else:
            cls = T.gt(pred, 0.5)

        self.output = cls

    def _calc_shape(self):
        sh = self.parent.shape.updateshape(self.pred.shape.tag2index('f'),
                                           self.n_indep)
        self.shape = sh


class _Errors(Node):
    """
    Errors node.

    Parameters
    ----------
    cls: T.Tensor
    target: T.Tensor
        corresponds to a vector that gives the correct label for each example. Labels < 0 are ignored (e.g. can
        be used for label propagation).
    target_is_sparse: bool
    name: str
        Node name.
    print_repr: bool
        Whether to print the node representation upon initialisation.
    """

    def __init__(self, cls, target, target_is_sparse=False,
                 name="errors", print_repr=True):
        parents = [cls, target]
        super(_Errors, self).__init__(parents, name, print_repr)

        self.n_class = cls.n_class
        self.n_indep = cls.n_indep

        self.target = target
        self.cls = cls
        self.target_is_sparse = target_is_sparse

    def _make_output(self):
        """ Computation of Theano Output """
        n_class = self.n_class
        n_indep = self.n_indep
        target = self.target.output
        axis = self.cls.shape.tag2index('f')

        if not self.target_is_sparse:
            if self.n_indep == 1:
                gt = T.argmax(target, axis=axis, keepdims=True)
            else:
                gt = []
                # This assumes that target is (b,n_class*n_indep,x,y,z)
                for i in range(n_indep):
                    sl = [slice(None), ] * target.ndim
                    sl[axis] = slice(i * n_class, (i + 1) * n_class, 1)
                    t = T.argmax(target[tuple(sl)], axis=axis, keepdims=True)
                    # t = T.argmax(target[:,i*n_class:(i+1)*n_class], axis=axis, keepdims=True)
                    gt.append(t)
                gt = T.concatenate(gt, axis=axis)
        else:
            gt = target

        gt = T.cast(gt, 'int16')

        self.output = T.mean(T.neq(gt, self.cls.output))

    def _calc_shape(self):
        self.shape = TaggedShape([1, ], ['f', ])


def Errors(pred, target, target_is_sparse=False, n_class='auto', n_indep='auto',
           name="errors", print_repr=True):
    if not isinstance(pred, Classification):
        pred = Classification(pred, n_class=n_class, n_indep=n_indep,
                              name='cls for errors', print_repr=False)
    return _Errors(pred, target, target_is_sparse=target_is_sparse,
                   name=name, print_repr=print_repr)


class GaussianNLL(Node):
    """
    Similar to squared loss but "modulated" in scale by the variance.

    Parameters
    ----------
    target: Node
        True value (target), usually directly an input node
    mu: Node
        Mean of the predictive Gaussian density
    sig: Node
        Sigma of the predictive Gaussian density
    sig_is_log: bool
        Whether ``sig`` is actually the ln(sig), then it is
        exponentiated internally


    Computes element-wise:

     .. math::

       0.5 \cdot  ( ln(2  \pi \sigma)) + (target-\mu)^2/\sigma^2 )

    """

    def __init__(self, mu, sig, target, sig_is_log=False, name="g_nll",
                 print_repr=True):
        super(GaussianNLL, self).__init__((mu, sig, target), name, print_repr)

        self.target = target
        self.mu = mu
        self.sig = sig
        self.sig_is_log = sig_is_log

    def _make_output(self):
        """ Computation of Theano Output """
        target = self.target.output
        mu = self.mu.output
        sig = self.sig.output

        # IF there are several samples per instance the target must be made
        # broadcastable along the sample axis
        if 's' in self.mu.shape.tags:
            pattern = list(range(self.target.type.ndim))
            batch_index = self.mu.shape.tag2index('s')
            pattern.insert(batch_index, 'x')
            target = target.dimshuffle(pattern)

        if self.sig_is_log:
            log_sig = sig
            sig = T.exp(sig)
        else:
            log_sig = T.log(sig)

        normalisation = 0.5 * np.log(2 * np.pi) + log_sig
        gauss = 0.5 * ((target - mu) / sig) ** 2
        logpxz = normalisation + gauss

        self.output = logpxz


class BetaNLL(Node):
    """
    Similar to BinaryNLL loss but "modulated" in scale by the variance.

    Parameters
    ----------

    target: Node
        True value (target), usually directly an input node, must be in range [0,1]
    mode: Node
        Mode of the predictive Beta density, must come from linear
        activation function (will be transformed by exp(.) + 2 )
    concentration: node
        concentration of the predictive Beta density


    Computes element-wise:

     .. math::

       0.5 \cdot  2
    """

    def __init__(self, mode, concentration, target, name="beta_nll",
                 print_repr=True):
        super(BetaNLL, self).__init__((mode, concentration, target), name,
                                      print_repr)

        self.target = target
        self.mode = mode
        self.concentration = concentration

    def _make_output(self):
        """ Computation of Theano Output """
        target = self.target.output
        mode = self.mode.output
        concentration = self.concentration.output

        # IF there are several samples per instance the target must be made
        # broadcastable along the sample axis
        if 's' in self.mode.shape.tags:
            pattern = list(range(self.target.type.ndim))
            batch_index = self.mode.shape.tag2index('s')
            pattern.insert(batch_index, 'x')
            target = target.dimshuffle(pattern)

        def log_inv_beta_func(a, b):
            return T.gammaln(a + b) - T.gammaln(a) - T.gammaln(b)

        def log_beta_pdf(x, mode, concentration):
            a = mode * (concentration - 2) + 1
            b = (1 - mode) * (concentration - 2) + 1
            p = log_inv_beta_func(a, b) + (a - 1) * T.log(x + EPS) + (
                                                                     b - 1) * T.log(
                1 - x + EPS)
            return p

        concentration2 = concentration
        self.output = - log_beta_pdf(target, mode,
                                     concentration2) + T.nnet.softplus(
            -concentration)  # sign!!!


class BinaryNLL(Node):
    """
    Binary NLL node. Identical to cross entropy.

    Parameters
    ----------

    pred: Node
        Predictive Bernoulli probability.
    target: Node
        True value (target), usually directly an input node.


    Computes element-wise:

     .. math::

       -(target  ln(pred) + (1 - target) ln(1 - pred))
    """

    def __init__(self, pred, target, subtract_label_entropy=False,
                 name="binary_nll", print_repr=True):
        super(BinaryNLL, self).__init__((pred, target), name, print_repr)

        self.target = target
        self.pred = pred
        self.pred_shape = pred.shape
        self.subtract_label_entropy = subtract_label_entropy

    def _make_output(self):
        target = self.target.output
        pred = self.pred.output
        # IF there are several samples per instance the target must be made
        # broadcastable along the sample axis
        if 's' in self.pred_shape.tags:
            pattern = list(range(target.type.ndim))
            batch_index = self.pred_shape.tag2index('s')
            pattern.insert(batch_index, 'x')
            target = target.dimshuffle(pattern)

        # mask = T.isnan(self.target)
        mask = T.isclose(target, -666.0)
        logger.warning(
            "BinaryNLL: isnan is replaced by 'isclose(target, -666)'")
        n_labelled = (1 - mask).sum()
        n_tot = pred.size
        scale = T.cast(n_tot, 'float32') / (n_labelled + 1)

        # logpxz = T.nnet.binary_crossentropy(pred, target) # This makes NaNs!!!!
        logpxz = -xlogy0(target, pred + EPS) - xlogy0(1.0 - target,
                                                      1.0 - pred + EPS)
        if self.subtract_label_entropy:
            logpxz += -xlogy0(target, target + EPS) - xlogy0(1.0 - target,
                                                             1.0 - target + EPS)

        logpxz = T.set_subtensor(logpxz[mask.nonzero()], 0.0)
        logpxz *= scale
        self.output = logpxz
        self._debug_outputs.extend([n_tot, n_labelled, scale, pred, target])


class SquaredLoss(Node):
    """
    Squared loss node.

    Parameters
    ----------
    pred: Node
        Prediction node.
    target: T.Tensor
        corresponds to a vector that gives the correct label for each example. Labels < 0 are ignored (e.g. can
        be used for label propagation).
    margin: float or None
    scale_correction: float or None
        Downweights absolute deviations for large target scale. The value specifies
        the target value at which the square deviation has half weight compared to target=0
        If the target is twice as large as this value the downweight is 1/3 and so on.
        Note: the smaller this value the stronger the effect. No effect would be
        +inf
    name: str
        Node name.
    print_repr: bool
        Whether to print the node representation upon initialisation.
    """

    def __init__(self, pred, target, margin=None, scale_correction=None,
                 name="se", print_repr=True):
        super(SquaredLoss, self).__init__((pred, target), name, print_repr)

        self.target = target
        self.pred = pred

        if margin:
            margin = VariableParam(value=margin, name="margin", dtype=floatX,
                                   apply_train=False)
            self.params['margin'] = margin

        self.margin = margin

        if scale_correction:
            scale_correction = VariableParam(value=scale_correction,
                                             name="scale_correction",
                                             dtype=floatX,
                                             apply_train=False)
            self.params['scale_correction'] = scale_correction

        self.scale_correction = scale_correction

    def _make_output(self):
        """ Computation of Theano Output """
        target = self.target.output
        pred = self.pred.output

        # IF there are several samples per instance the target must be made
        # broadcastable along the sample axis
        # if 's' in self.mu.shape.tags:
        #     pattern = list(range(self.target.type.ndim))
        #     batch_index = self.mu.shape.tag2index('s')
        #     pattern.insert(batch_index, 'x')
        #     target = target.dimshuffle(pattern)

        # mask = T.isnan(target)
        mask = T.isclose(target, -666.0)
        logger.warning(
            "SquaredLoss: isnan is replaced by 'isclose(target, -666)'")
        n_labelled = (1 - mask).sum()
        n_tot = pred.size
        scale = T.cast(n_tot, 'float32') / (n_labelled + 1)

        if self.margin is not None:
            diff = target - pred
            out = scale * 0.5 * T.square(diff) * T.ge(abs(diff),
                                                      self.margin) - self.margin
        else:
            out = scale * 0.5 * T.square(target - pred)

        if self.scale_correction is not None:
            correction = self.scale_correction / (
            abs(target) + self.scale_correction)
            out *= correction

        out = T.set_subtensor(out[mask.nonzero()], 0.0)
        self.output = T.mean(out, axis=self.pred.shape.tag2index('f'),
                             keepdims=True)
        self._debug_outputs.extend([n_tot, n_labelled, scale, pred, target])

    def _calc_shape(self):
        sh = self.parent[0].shape.updateshape(self.pred.shape.tag2index('f'), 1)
        self.shape = sh


class EuclideanDistance(Node):
    """
    Euclidean distance node.

    Parameters
    ----------
    pred: Node
        Prediction node.
    target: T.Tensor
        corresponds to a vector that gives the correct label for each example. Labels < 0 are ignored (e.g. can
        be used for label propagation).
    margin: float/None
    scale_correction: float/None
        Downweights absolute deviations for large target scale. The value specifies
        the target value at which the square deviation has half weight compared to target=0
        If the target is twice as large as this value the downweight is 1/3 and so on.
        Note: the smaller this value the stronger the effect. No effect would be
        +inf
    name: str
        Node name.
    print_repr: bool
        Whether to print the node representation upon initialisation.
    """  # TODO: Docstring Parameters do not match __init__ parameters.

    def __init__(self, pred, target, name="se", print_repr=True):
        super(EuclideanDistance, self).__init__((pred, target), name,
                                                print_repr)

        self.target = target
        self.pred = pred

    def _make_output(self):
        """ Computation of Theano Output """
        target = self.target.output
        pred = self.pred.output

        # IF there are several samples per instance the target must be made
        # broadcastable along the sample axis

        diff = target - pred
        out = diff.norm(2, axis=self.pred.shape.tag2index('f'))
        mask = T.isnan(out)
        self.output = T.set_subtensor(out[mask.nonzero()], 0.0)
        self._debug_outputs.extend([pred, target])

    def _calc_shape(self):
        sh = self.parent[0].shape.updateshape(self.pred.shape.tag2index('f'), 1)
        self.shape = sh


class RampLoss(Node):
    """
    RampLoss node.

    Parameters
    ----------
    pred: Node
        Prediction node.
    target: T.Tensor
        corresponds to a vector that gives the correct label for each example. Labels < 0 are ignored (e.g. can
        be used for label propagation).
    margin: float/None
    scale_correction: float/None
        downweights absolute deviations for large target scale. The value specifies
        the target value at which the square deviation has half weight compared to target=0
        If the target is twice as large as this value the downweight is 1/3 and so on.
        Note: the smaller this value the stronger the effect. No effect would be
        +inf
    name: str
        Node name.
    print_repr: bool
        Whether to print the node representation upon initialisation.
    """  # TODO: Docstring parameters do not match __init__ parameters.

    def __init__(self, d_low, d_big, name="se", print_repr=True, margin=None):
        super(RampLoss, self).__init__((d_low, d_big), name, print_repr)

        if margin is None:
            margin = 0
        margin = VariableParam(value=margin, name="margin", dtype=floatX,
                               apply_train=False)
        self.params['margin'] = margin

        self.margin = margin

        self.d_low = d_low
        self.d_big = d_big

    def _make_output(self):
        """ Computation of Theano Output """
        d_low = self.d_low.output
        d_big = self.d_big.output

        # IF there are several samples per instance the target must be made
        # broadcastable along the sample axis

        diff = d_low - d_big + self.margin
        neg_mask = diff < 0
        diff = T.set_subtensor(diff[neg_mask.nonzero()], 0.0)
        mask = T.isnan(diff)
        out = T.set_subtensor(diff[mask.nonzero()], 0.0)
        self.output = T.mean(out, axis=self.d_low.shape.tag2index('f'),
                             keepdims=True)
        self._debug_outputs.extend([d_low, d_big])

    def _calc_shape(self):
        sh = self.parent[0].shape.updateshape(self.d_low.shape.tag2index('f'),
                                              1)
        self.shape = sh


class AbsLoss(SquaredLoss):
    """
    AbsLoss node.

    Parameters
    ----------
    pred: Node
        Prediction node.
    target: T.Tensor
        corresponds to a vector that gives the correct label for each example. Labels < 0 are ignored (e.g. can
        be used for label propagation).
    margin: float or None
    scale_correction: float or None
        Boosts loss for large target values: if target=1 the error
        is multiplied by this value (and linearly for other targets)
    name: str
        Node name.
    print_repr: bool
        Whether to print the node representation upon initialisation.
    """

    def __init__(self, pred, target, margin=None, scale_correction=None,
                 name="absloss", print_repr=True):
        super(AbsLoss, self).__init__(pred, target, margin=margin,
                                      scale_correction=scale_correction,
                                      name=name, print_repr=print_repr)

    def _make_output(self):
        """ Computation of Theano Output """
        target = self.target.output
        pred = self.pred.output

        # IF there are several samples per instance the target must be made
        # broadcastable along the sample axis
        # if 's' in self.mu.shape.tags:
        #     pattern = list(range(self.target.type.ndim))
        #     batch_index = self.mu.shape.tag2index('s')
        #     pattern.insert(batch_index, 'x')
        #     target = target.dimshuffle(pattern)

        # mask = T.isnan(target)
        mask = T.isclose(target, -666.0)
        logger.warning(
            "SquaredLoss: isnan is replaced by 'isclose(target, -666)'")
        n_labelled = (1 - mask).sum()
        n_tot = pred.size
        scale = T.cast(n_tot, 'float32') / (n_labelled + 1)

        if self.margin is not None:
            diff = target - pred
            out = scale * abs(diff) * T.ge(abs(diff), self.margin) - self.margin
        else:
            out = scale * abs(target - pred)

        if self.scale_correction is not None:
            correction = self.scale_correction * abs(target) + 1.0
            out *= correction

        out = T.set_subtensor(out[mask.nonzero()], 0.0)
        self.output = T.mean(out, axis=self.pred.shape.tag2index('f'),
                             keepdims=True)
        self._debug_outputs.extend([n_tot, n_labelled, scale, pred, target])


class AggregateLoss(Node):
    """
    This node is used to average the individual losses over a batch
    (and possibly, spatial/temporal dimensions). Several losses can be
    mixed for multi-target training.

    Parameters
    ----------

    parent_nodes: list/tuple of graph or single node
        each component is some (possibly element-wise) loss array
    mixing_weights: list/None
        Weights for the individual costs. If none, then all are weighted
        equally. If mixing weights are used, they can be changed during
        training by manipulating the attribute ``params['mixing_weights']``.
    name: str
        Node name.
    print_repr: bool
        Whether to print the node representation upon initialisation.

    # The following is all wrong, mixing_weights are directly used:

    The losses are first summed per component, and then the component sums
    are summed using the relative weights. The resulting scalar is finally
    normalised such that:
        * The cost does not grow with the number of mixed components
        * Components which consist of more individual losses have more weight
          e.g. If there is a constraint on some hidden representation
          with 20 features and a constraint the reconstruction of 100 features,
          the reconstruction constraint has 5x more impact on the overall loss
          than the constraint on the hidden state (provided those two loss
          are initially on the same scale). If they are intended to have equal
          impact, the weights should be used to upscale the constraint against
          the reconstruction.

    """  # TODO: What about the "all wrong" section?

    def __init__(self, parent_nodes, mixing_weights=None, name="total_loss",
                 print_repr=True):

        if not isinstance(parent_nodes, (tuple, list)):
            parent_nodes = [parent_nodes, ]

        super(AggregateLoss, self).__init__(parent_nodes, name, print_repr)

        if mixing_weights is None:
            mixing_weights = np.ones(len(parent_nodes))

        if isinstance(mixing_weights, (tuple, list, np.ndarray)):
            if len(parent_nodes) != len(mixing_weights):
                stat = "len(parent_nodes)=%i, len(weights)=%i" \
                       % (len(parent_nodes), len(mixing_weights))
                raise ValueError("Mismatch: %s" % (stat,))

            mixing_weights = np.array(mixing_weights, dtype=floatX)
            # mixing_weights *= (len(mixing_weights) / mixing_weights.sum()) # normalise
            mixing_weights = VariableParam(value=mixing_weights,
                                           name="loss_mixing_weights",
                                           dtype=floatX,
                                           apply_train=False)

        else:
            raise ValueError("Unsupported weight format")

        self.params['mixing_weights'] = mixing_weights
        self.mixing_weights = mixing_weights

    def _make_output(self):
        """ Computation of Theano Output """
        # The normalisation is such that:
        # - The cost does not grow with the number of mixed components
        # - Components which consist of more individual losses have more weight
        # inputs = [inp.output for inp in self.parent]
        # sums   = T.stack([inp.sum() for inp in inputs])
        # total_sum = T.sum(sums * self.mixing_weights)
        # sizes = [inp.size for inp in inputs]
        # total_size = T.sum(T.stack(sizes))
        # self.output = total_sum / total_size
        means = []
        for inp in self.parent:
            m = T.mean(inp.output)
            means.append(m)

        means = T.mul(means, self.mixing_weights)
        self.output = T.mean(means)

    def _calc_shape(self):
        self.shape = TaggedShape([1, ], ['f', ])

    def _calc_comp_cost(self):
        self.computational_cost = np.sum(
            [inp.shape.stripnone_prod for inp in self.parent])


def SobelizedLoss(pred, target, loss_type='abs', loss_kwargs=None):
    """
    SobelizedLoss node.

    Parameters
    ----------
    pred: Node
            Prediction node.
    target: T.Tensor
        corresponds to a vector that gives the correct label for each example. Labels < 0 are ignored (e.g. can
        be used for label propagation).
    loss_type: str
        Only "abs" is supported.
    loss_kwargs: dict
        kwargs for the AbsLoss constructor.
    Returns
    -------
    Node:
        The loss node.
    """
    if loss_kwargs is None:
        loss_kwargs = dict()

    dim = pred.shape.ndim
    f = pred.shape['f']

    w_sh = (f * dim, f) + (3,) * dim
    w = np.zeros(w_sh, dtype=floatX)
    b = np.zeros((f * dim), dtype=floatX)
    base_w = np.array([1, 0, -1], dtype=floatX)

    if dim > 1:
        n = np.array([[0.3, 0.4, 0.3]], dtype=np.float32).T
        base_w = np.tile(base_w, [3, 1]) * n
        base_w = np.concatenate([base_w[None], base_w.T[None]], axis=0)

    if dim > 2:
        base_w = np.tile(base_w[0], [3, 1, 1]) * n[:, :, None]
        base_w = np.concatenate([np.transpose(base_w, (2, 1, 0))[None],
                                 base_w[None],
                                 np.transpose(base_w, (1, 2, 0))[None]], axis=0)

    if dim > 3:
        raise NotImplementedError()

    # Now w_base has a filter for each dimension, next we need to take care
    # of the channels in the input
    for i in range(f):
        w[i::f, i] = base_w

    pred_sobel = Conv(pred, f * dim, (3,) * dim, (1,) * dim, conv_mode='same',
                      activation_func='lin', w=[w, 'const'], b=[b, 'const'],
                      name='pred_sobel')

    target_sobel = Conv(target, f * dim, (3,) * dim, (1,) * dim,
                        conv_mode='same',
                        activation_func='lin', w=[w, 'const'], b=[b, 'const'],
                        name='target_sobel')

    if loss_type == 'abs':
        loss = AbsLoss(pred_sobel, target_sobel, **loss_kwargs)
    # elif loss_type=='mnll':
    #    loss = MultinoulliNLL(pred_sobel, target_sobel, **loss_kwargs)
    else:
        raise NotImplementedError()

    return loss


if __name__ == "__main__":
    from elektronn2.neuromancer import Input

    #    pred = Input((2,6,1), 'b,f,x')
    #    pred = Softmax(pred, 3, 2)
    #    lab  = Input((2,2,1), 'b,f,x', name='labels', dtype='int16')
    #    cls  = Classification(pred)
    #    err  = Errors(pred, lab, target_is_sparse=True)
    #
    #    pred_val = np.array([[0.6,0.2,0.2,0.8,0.1,0.1],
    #                         [0.2,0.7,0.1,0.1,0.1,0.8]], dtype=np.float32)[...,None]
    #    lab_val = np.array([[0,0],[1,2]], dtype=np.int16)[...,None]
    #    print cls(pred_val), cls(pred_val).shape
    #    print err(pred_val, lab_val)
    #
    #
    #    lab  = Input((2,6,1), 'b,f,x', name='labels', dtype='int16')
    #    cls  = Classification(pred)
    #    err  = Errors(pred, lab, target_is_sparse=False)
    #
    #    pred_val = np.array([[0.6,0.2,0.2,0.8,0.1,0.1],
    #                         [0.2,0.7,0.1,0.1,0.1,0.8]], dtype=np.float32)[...,None]
    #    lab_val = np.array([[1,0,0,1,0,0],[0,1,0,0,0,1]], dtype=np.int16)[...,None]
    #    print cls(pred_val), cls(pred_val).shape
    #    print err(pred_val, lab_val)


    #    pred = Input((2,6), 'b,f')
    #    pred = Softmax(pred, 3, 2)
    #    lab  = Input((2,2), 'b,f', name='labels', dtype='int16')
    #    nll  = MultinoulliNLL(pred, lab, target_is_sparse=True)
    #
    #    pred_val = np.array([[0.6,0.2,0.2,0.8,0.1,0.1],
    #                         [0.2,0.7,0.1,0.1,0.1,0.8]], dtype=np.float32)
    #    lab_val = np.array([[0,0],[1,2]], dtype=np.int16)
    #    print nll(pred_val, lab_val), pred(pred_val)


    pred = Input((2, 6), 'b,f')
    example_weights = Input((2,), 'b', name='example_weights')
    class_weights = Input((2,), 'b', name='class_weights')
    mask_class_not_present = Input((2, 6), 'b,f', name='mask_class_not_present')
    lab = Input((2, 2), 'b,f', name='labels', dtype='int16')
    pred = Softmax(pred, 3, 2)

    nll = MultinoulliNLL(pred, lab,
                         example_weights=example_weights,
                         class_weights=class_weights,
                         mask_class_not_present=mask_class_not_present,
                         target_is_sparse=True)

    pred_val = np.array([[0.6, 0.2, 0.2, 0.8, 0.1, 0.1],
                         [0.2, 0.7, 0.1, 0.1, 0.1, 0.8]], dtype=np.float32)
    lab_val = np.array([[0, 0], [1, 2]], dtype=np.int16)
    exp_val = np.array([1, 1], dtype=np.int16)
    cls_val = np.array([1, 1, 1, 1, 1, 1], dtype=np.float32)
    not_pres = np.array([[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0]],
                        dtype=np.int16)
    logger.debug(nll(pred_val, lab_val, cls_val, exp_val, not_pres))
