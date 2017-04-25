# -*- coding: utf-8 -*-

# ELEKTRONN2 Toolkit
# Copyright (c) 2015 Marius Killinger
# All rights reserved

from __future__ import absolute_import, division, print_function
from builtins import filter, hex, input, int, map, next, oct, pow, range, \
    super, zip

import logging
import numpy as np


__all__ = ["cnncalculator", "get_cloesest_valid_patch_size",
           "get_valid_patch_sizes"]

logger = logging.getLogger('elektronn2log')


class _Layer(object):
    def __init__(self, patch_size, filter=1, pool=1, stride=1, mfp=True):
        self.field = None
        self.overlap = None

        self.out = patch_size - filter + 1
        if self.out <= 0:
            raise ValueError('CNN has no output for Layer with patch_size',
                             patch_size)
        self.stride = stride
        rest = self.out % pool
        self.pool_out = self.out // pool
        if pool > 1:
            if mfp and rest!=1:
                raise ValueError('mfp fails for Layer with patch_size',
                                 patch_size)
            elif not mfp and rest > 0:
                raise ValueError('Uneven Pools for Layer with patch_size',
                                 patch_size)

    def setfield(self, field):
        self.field = field
        self.overlap = field - self.stride


class _Cnncalculator(object):
    def __init__(self, filters, poolings, desired_patch_size, mfp,
                 force_center, desired_output):
        self.layers = None
        self.pool_out = None
        self.out = None
        self.stride = None
        self.mfp = mfp
        self.fields = self.getFields(filters, poolings)
        fow = self.fields[-1]
        if fow % 2==0:
            if force_center:
                raise ValueError('Receptive Fields are not '
                                 'centered with field of view (%i)' % fow)
            else:
                logger.warning('WARNING: Receptive Fields are not centered '
                               'with even field of view (%i)' % fow)
        self.offset = float(fow) / 2

        valid_patch_sizes = []
        valid_outputs = []
        for inp in range(2, 5000):
            try:
                self.calclayers(inp, filters, poolings, mfp)
                valid_patch_sizes.append(inp)
                valid_outputs.append(self.out[-1])
            except:
                pass

        if desired_output is not None:
            if desired_output in valid_outputs:
                i = valid_outputs.index(desired_output)
                patch_size = valid_patch_sizes[i]
            else:
                valid_outputs = np.array(valid_outputs)
                patch_size = valid_outputs[valid_outputs <= desired_output][-1]
                logger.info("Info: output size requires patch_size>5000, "
                            "next smaller output (%i) is used" % patch_size)
                valid_outputs = list(valid_outputs)

            # patch_size corresponding to that output
            i = valid_outputs.index(desired_output)
            patch_size = valid_patch_sizes[i]

        elif desired_patch_size in valid_patch_sizes:
            patch_size = desired_patch_size
        elif desired_patch_size is None:
            patch_size = valid_patch_sizes[-1]
        else:
            valid_patch_sizes = np.array(valid_patch_sizes)
            if desired_patch_size < valid_patch_sizes[0]:
                patch_size = valid_patch_sizes[0]
                logger.info("patch_size (%i) changed to (%i) "
                            "(size too small)" % (
                                desired_patch_size, patch_size))
            else:
                patch_size = valid_patch_sizes[
                    valid_patch_sizes <= desired_patch_size][-1]
                logger.info("patch_size (%i) changed to (%i) (size not "
                            "possible)" % (desired_patch_size, patch_size))
                valid_patch_sizes = list(valid_patch_sizes)

        self.valid_patch_sizes = valid_patch_sizes
        self.calclayers(patch_size, filters, poolings, mfp)
        self.patch_size = patch_size
        self.pred_stride = self.layers[-1].stride
        for lay, field in zip(self.layers, self.fields):
            lay.setfield(field)

        self.overlap = [l.overlap for l in self.layers]

    def calclayers(self, patch_size, filters, poolings, mfp):
        stride = poolings[0]
        lay0 = _Layer(patch_size, filters[0], poolings[0], stride, mfp=mfp[0])
        self.layers = [lay0,]
        for i in range(1, len(filters)):
            stride = np.multiply(stride, poolings[i])
            lay = _Layer(self.layers[i - 1].pool_out, filters[i], poolings[i],
                         stride, mfp[i])
            self.layers.append(lay)

        self.pool_out = [l.pool_out for l in self.layers]
        self.out = [l.out for l in self.layers]
        self.stride = [l.stride for l in self.layers]

    def __repr__(self):
        if not isinstance(self.pool_out[0], list):
            ls = self.pool_out[::-1]
        else:
            ls = list(zip(*self.pool_out))[::-1]
        if not isinstance(self.out[0], list):
            out = self.out[::-1]
        else:
            out = list(zip(*self.out))[::-1]
        if not isinstance(self.fields[0], list):
            fields = self.fields[::-1]
        else:
            fields = list(zip(*self.fields))[::-1]
        if not isinstance(self.stride[0], list):
            stride = self.stride[::-1]
        else:
            stride = list(zip(*self.stride))[::-1]
        if not isinstance(self.overlap[0], list):
            overlap = self.overlap[::-1]
        else:
            overlap = list(zip(*self.overlap))[::-1]

        s = "patch_size: " + repr(
            self.patch_size) + "\nLayer/Fragment sizes:\t" + repr(ls)\
            + "\nUnpooled Layer sizes:\t" + repr(out)\
            + "\nReceptive fields:\t" + repr(fields)\
            + "\nStrides:\t\t" + repr(stride)\
            + "\nOverlap:\t\t" + repr(overlap) + "\nOffset:\t\t"\
            + repr(self.offset) + "\nIf offset is non-int: output neurons " \
            + "lie centered on patch_size neurons,they have an odd FOV\n"
        return s

    @staticmethod
    def getFields(filter, pool):
        def recFields_helper(filter, pool):
            rf = [None] * (len(filter) + 1)
            rf[-1] = 1
            for i in range(len(filter), 0, -1):
                rf[i - 1] = rf[i] * pool[i - 1] + filter[i - 1] - 1
            return rf[0]

        fields = []
        for i in range(1, len(filter) + 1):
            fields.append(recFields_helper(filter[:i], pool[:i]))

        return fields


class _Multi_cnncalculator(_Cnncalculator):
    """ Adaptor Class to unify multiple CNNCalculators"""

    def __init__(self, calcs):
        self.fields = []
        self.offset = []
        self.valid_patch_sizes = []
        self.patch_size = []
        self.pred_stride = []
        self.stride = []
        self.pool_out = []
        self.out = []
        self.overlap = []
        for c in calcs:
            self.fields.append(c.fields)
            self.offset.append(c.offset)
            self.valid_patch_sizes.append(c.valid_patch_sizes)
            self.patch_size.append(c.patch_size)
            self.pred_stride.append(c.pred_stride)
            self.overlap.append(c.overlap)
            self.pool_out.append(c.pool_out)
            self.out.append(c.out)
            self.stride.append(c.stride)


def cnncalculator(filters, poolings, desired_patch_size=None, mfp=False,
                  force_center=False, desired_output=None, ndim=1):
    """
    Helper to calculate CNN architectures

    This is a *function*, but it returns an *object* that has various
    architecture values as attributes.
    Useful is also to simply print 'd' as in the example.

    Parameters
    ----------

    filters: list
      Filter shapes (for anisotropic filters the shapes are again a list)
    poolings: list
      Pooling factors
    desired_patch_size: int or list of int
      Desired patch_size size(s). If ``None`` a range of suggestions can be
      found in the attribute ``valid_patch_sizes``
    mfp: list of int/{0,1}
      Whether to apply Max-Fragment-Pooling in this Layer and check
      compliance with max-fragment-pooling
      (requires other patch_size sizes than normal pooling)
    force_center: Bool
      Check if output neurons/pixel lie at center of patch_size
      neurons/pixel (and not in between)
    desired_output: int or list of int
      Alternative to ``desired_patch_size``
    ndim: int
      Dimensionality of CNN

    Examples
    --------

    Calculation for anisotropic "flat" 3d CNN with mfp in the first layers only::

      >>> desired_patch_size   = [211, 211, 20]
      >>> filters         = [[6,6,1], [4,4,4], [2,2,2], [1,1,1]]
      >>> pool            = [[2,2,1], [2,2,2], [2,2,2], [1,1,1]]
      >>> mfp             = [1,        1,       0,       0,   ]
      >>> ndim=3
      >>> d = cnncalculator(filters, pool, desired_patch_size, mfp=mfp, force_center=True, desired_output=None, ndim=ndim)
      Info: patch_size (211) changed to (210) (size not possible)
      Info: patch_size (211) changed to (210) (size not possible)
      Info: patch_size (20) changed to (22) (size too small)
      >>> print(d)
      patch_size: [210, 210, 22]
      Layer/Fragment sizes:	[[102, 49, 24, 24], [102, 49, 24, 24], [22, 9, 4, 4]]
      Unpooled Layer sizes:	[[205, 99, 48, 24], [205, 99, 48, 24], [22, 19, 8, 4]]
      Receptive fields:	[[7, 15, 23, 23], [7, 15, 23, 23], [1, 5, 9, 9]]
      Strides:		[[2, 4, 8, 8], [2, 4, 8, 8], [1, 2, 4, 4]]
      Overlap:		[[5, 11, 15, 15], [5, 11, 15, 15], [0, 3, 5, 5]]
      Offset:		[11.5, 11.5, 4.5].
          If offset is non-int: floor(offset).
          Select labels from within img[offset-x:offset+x]
          (non-int means, output neurons lie centered on patch_size neurons,
          i.e. they have an odd field of view)
    """  # TODO: The example is still ELEKTRONN1-specific. Check the whole function for incompatibilities.

    assert len(poolings)==len(filters)

    if mfp is False:
        mfp = [False, ] * len(filters)

    if ndim==1:  # not hasattr(filters[0], '__len__') :
        return _Cnncalculator(filters, poolings, desired_patch_size, mfp,
                              force_center, desired_output)
    else:
        if desired_patch_size is None:
            desired_patch_size = (None,) * ndim
        elif not hasattr(desired_patch_size, '__len__'):
            desired_patch_size = (desired_patch_size,) * ndim
        if desired_output is None:
            desired_output = (None,) * ndim
        elif not hasattr(desired_output, '__len__'):
            desired_output = (desired_output,) * ndim
        if not hasattr(poolings[0], '__len__'):
            poolings = [[p, ] * ndim for p in poolings]
        if not hasattr(filters[0], '__len__'):
            filters = [[f, ] * ndim for f in filters]
        if not hasattr(mfp[0], '__len__'):
            mfp = [[m, ] * ndim for m in mfp]

        assert len(mfp)==len(filters)

        filters = [list(l) for l in zip(*filters)]
        poolings = [list(l) for l in zip(*poolings)]
        mfp = [list(l) for l in zip(*mfp)]

        calcs = []
        for f, p, d, o, mfp in zip(filters, poolings, desired_patch_size,
                                   desired_output, mfp):
            c = _Cnncalculator(f, p, d, mfp, force_center, o)
            calcs.append(c)

        return _Multi_cnncalculator(calcs)


def get_valid_patch_sizes(filters, poolings, desired_patch_size=100,mfp=False,
                          ndim=1):
    calc = cnncalculator(filters, poolings, desired_patch_size, mfp=mfp,
                         ndim=ndim)
    return calc.valid_patch_sizes


def get_cloesest_valid_patch_size(filters, poolings, desired_patch_size=100,
                                  mfp=False, ndim=1):
    calc = cnncalculator(filters, poolings, desired_patch_size, mfp=mfp,
                         ndim=ndim)
    return calc.patch_size


if __name__=="__main__":
    logger.debug("Testing cnncalculator")
    desired_patch_size = 200
    mfp = False
    filters = [6, 5, 4, 4, 1, 4, 4, 4, 4, 2]
    pool = [1, 2, 2, 1, 1, 1, 1, 1, 1, 1]

    #    filters         = [1,1,1,1, 4,3,3,2, 2,1]
    #    pool            = [1,1,1,1, 2,1,1,1, 1,1]

    ndim = 1
    d = cnncalculator(filters, pool, desired_patch_size, mfp=mfp,
                      force_center=True, desired_output=None, ndim=ndim)
    print(d)
