# -*- coding: utf-8 -*-
# ELEKTRONN2 Toolkit
# Copyright (c) 2015 Marius F. Killinger
# All rights reserved

from __future__ import absolute_import, division, print_function
from builtins import filter, hex, input, int, map, next, oct, pow, range, super, zip

from collections import OrderedDict
import sys
import time
import logging

import numpy as np
import theano

logger = logging.getLogger('elektronn2log')

if sys.version_info.major >= 3:
    unicode = str

__all__ = ['TaggedShape', 'floatX', 'as_floatX', 'make_func',
           'getinput_for_multioutput']

floatX = theano.config.floatX

class TaggedShape(object):
    """
    Object to manage shape and associated tags uniformly.
    The ``[]``-operator can be used get shape values by either index (``int``) or tag (``str``)

    Parameters
    ----------
    shape: list/tuple of int
        shape of array, unspecified shapes are ``None``
    tags: list/tuple of strings or comma-separated string
        tags indicate which purpose the dimensions of the tensor serve. They are
        sometimes used to decide about reshapes. The maximal tensor has tags:
        "r, b, s, f, z, y, x, s" which denote:
        * r: perform recurrence along this axis
        * b: batch size
        * s: samples of the same instance (over which expectations are calculated)
        * f: features, filters, channels
        * z: convolution no. 3 (slower than 1,2)
        * y: convolution no. 1
        * x: convolution no. 2
    strides:
        list of strides, only for spatial dimensions, so it is 1-3 long
    mfp_offsets:
    """

    def __init__(self, shape, tags, strides=None, mfp_offsets=None, fov=None,):
        self._shape = list(shape) # copy
        self._tags  = self._check_tags(tags) # is copied too
        if len(self._shape)!=len(self._tags):
            raise ValueError("Shape %s and tags %s must have same length"\
                             %(self._shape, self._tags))

        if strides is None:
            self._strides = np.ones(len(self.spatial_axes), np.int)
        else:
            self._strides = np.array(strides, np.int)

        if mfp_offsets is None:
            self._mfp_offsets = np.zeros((1, len(self.spatial_axes)), np.int)
        else:
            self._mfp_offsets = np.atleast_2d((np.array(mfp_offsets, np.int)))

        if fov is None:
            self._fov = np.ones(len(self.spatial_axes), np.int)
        else:
            self._fov = np.array(fov, np.int)

    def __repr__(self):
        r = "["
        for s,t  in zip(self._shape, self._tags):
            r += "(%s,%s), "%(s,t)

        r = r[:-2] # remove last 2 chars
        r += "]"
        return r

    @property
    def ext_repr(self):
        s = repr(self)
        s += '\n'
        s += 'fov=%s, offsets=%s, strides=%s, spatial shape=%s'\
             %(self.fov, self.offsets, self.strides, self.spatial_shape)
        return s


    @staticmethod
    def _check_tags(tags):
        if tags is None:
            return None

        if not isinstance(tags, (list, tuple)):
            if isinstance(tags, (str, unicode)):
                tags = tags.split(',')
                tags = [x.strip() for x in tags]
            else:
                raise ValueError("Tags must be either list/tuple of comma-separated string, not %s" %tags)

        allowed_tags = ['r', 'b', 'z', 'f', 'x', 'y', 's']
        for t in tags:
            if t not in allowed_tags:
                raise ValueError("Unknown tag %s" %(t,))

        return tags

    def __getitem__(self, slice):
        if isinstance(slice, (str, unicode)):
            return self._shape[self.tag2index(slice)]
        else:
            return self._shape[slice]

    def __len__(self):
        return len(self._shape)

    @property
    def shape(self):
        return self._shape

    @property
    def tags(self):
        return self._tags

    @property
    def strides(self):
        return self._strides

    @property
    def mfp_offsets(self):
        return self._mfp_offsets

    @property
    def fov(self):
        return list(self._fov)

    @property
    def fov_all_centered(self):
        fov = self.fov
        return np.all(np.mod(fov, 2)==1)

    @property
    def offsets(self):
        return [i//2 for i in self.fov]

    @property
    def spatial_axes(self):
        spatial_axes = [tag for tag in ['z', 'x', 'y'] if self.hastag(tag)]
        spatial_axes = [self.tag2index(tag) for tag in spatial_axes]
        spatial_axes.sort()
        return list(spatial_axes)

    @property
    def ndim(self):
        return len(self.spatial_axes)

    @property
    def spatial_shape(self):
        return [self.shape[i] for i in self.spatial_axes]

    @property
    def spatial_size(self):
        return int(np.prod(self.spatial_shape))

    @property
    def stripnone(self):
        """
        Return the shape but with all None elements removed (e.g. if batch size
        is unspecified)
        """
        new_sh = []
        for s in self._shape:
            if s is not None:
                new_sh.append(s)

        return new_sh

    @property
    def stripbatch_prod(self):
        """
        Calculate product excluding batch dimension
        """
        new_sh = []
        for s,t in zip(self._shape, self._tags):
            if t is not 'b':
                new_sh.append(s)

        return np.prod(new_sh)

    @property
    def stripnone_prod(self):
        """
        Return the product of the shape but with all None elements removed
        (e.g. if batch size is unspecified)
        """
        return np.prod(self.stripnone)

    def tag2index(self, target_tag):
        """
        Finds the index of the desired tag
        """
        try:
            i = self._tags.index(target_tag)
        except ValueError:
            raise ValueError("Shape does not have tag %s, only"
                             "tags %s" %(target_tag, self._tags))

        return i

    def hastag(self, tag):
        has = False
        try:
            self.tag2index(tag)
            has = True
        except ValueError:
            pass

        return has


    def updateshape(self, axis, new_size, mode=None):
        """
        Create new TaggedShape with ``new_size`` on ``axis`` .
        Modes for updating: ``None`` (override), ``'add'``, ``'mult'``
        """
        if isinstance(axis, int):
            i = axis
        else:
            i = self.tag2index(axis)

        sh = list(self._shape) # copy

        if mode is None:
            sh[i] = new_size
        else:
            if sh[i] is None:
                logger.debug("Updating shape %s at index %i carrying value None, "
                             "will stay None" %(self, i))
                sh[i] = sh[i]
            elif mode=='add':
                sh[i] += new_size
            elif mode=='mult':
                sh[i] *= new_size

        ret = self.copy()
        ret._shape = sh

        return ret


    def updatefov(self, axis, new_fov):
        """
        Create new TaggedShape with ``new_fov`` on ``axis``. Axis is given as
        index of the spatial axes (not matching the absolute index of sh).
        """
        ret = self.copy()
        ret._fov[axis] = new_fov

        return ret

    def updatestrides(self, strides):
        ret = self.copy()
        ret._strides = strides
        return ret

    def updatemfp_offsets(self, mfp_offsets):
        ret = self.copy()
        ret._mfp_offsets = mfp_offsets
        return ret

    def addaxis(self, axis, size, tag):
        """
        Create new TaggedShape with new axis inserted at ``axis`` of size ``size``
        tagged ``tag``. If axis is a tag, the new axis is **right** of that tag
        """
        if isinstance(axis, int):
            i = axis
        else:
            i = self.tag2index(axis) + 1

        sh = list(self._shape) # copy
        sh.insert(i, size)
        tags = list(self._tags)
        tags.insert(i, tag)

        return TaggedShape(sh, tags, self._strides, self._mfp_offsets, self._fov)

    def delaxis(self, axis):
        """
        Create new TaggedShape with new axis inserted at ``axis`` of size ``size``
        tagged ``tag``. If axis is a tag, the new axis is **right** of that tag
        """
        if isinstance(axis, int):
            i = axis
        else:
            i = self.tag2index(axis) + 1

        sh = list(self._shape) # copy
        sh.pop(i)
        tags = list(self._tags)
        tags.pop(i)
        return TaggedShape(sh, tags, self._strides, self._mfp_offsets, self._fov)


    def copy(self):
        return TaggedShape(self._shape, self._tags,
                           self._strides, self._mfp_offsets, self.fov)



class make_func(object):
    """
    Wrapper for compiled theano functions. Features:

    * The function is compiled on demand (i.e. no wait at initialisation)
    * Singleton return values are returned directly, multiple values as list
    * The last execution time can inspected in the attribute ``last_exec_time``
    * Functions can be timed: ``profile_execution`` is an ``int`` that specifies
      the number of runs to average. The average time is printed then.
    * In/Out values can have a ``borrow`` flag which might overwrite the numpy
      arrays but might speed up execution (see theano doc)
    """

    def __init__(self, tt_input, tt_output, updates=None, name='Unnamed Function',
                 borrow_inp=False, borrow_out=False, profile_execution=False):
        self.name = name
        self.func = None
        self.profile = profile_execution
        self.last_exec_time = None
        self.updates = updates
        if borrow_inp:
            tt_input = [theano.In(x, borrow=True) for x in tt_input]

        self.tt_input = tt_input

        self.single_return = False
        if not isinstance(tt_output, (list, tuple)):
            tt_output = [tt_output,]
            self.single_return = True

        if borrow_out:
            tt_output = [theano.Out(x, borrow=True) for x in tt_output]

        self.tt_output = tt_output

    def __call__(self, *args):
        if self.func is None:
            self.compile()

        if len(args)==0: # calling foo() now just compiles the function
            return

        if self.profile:
            t0 = time.time()
            if self.profile>1:
                for i in range(self.profile-1):
                    self.func(*args)

            ret = self.func(*args)
            t = time.time() - t0
            logger.info("Function <%s> took %.5f s averaged over %i execs" % (self.name, t/self.profile, self.profile))

        else:
            t0 = time.time()
            ret =  self.func(*args)
            t = time.time() - t0
            self.last_exec_time = t

        if self.single_return:
            ret = ret[0]
        return ret

    def compile(self, profile=False):
        t_init = time.time()
        logger.info("Compiling %s, inputs=%s" % (self.name, self.tt_input))
        tf = theano.function(self.tt_input,
                             self.tt_output,
                             updates=self.updates,
                             name=self.name,
                             on_unused_input='warn',
                             profile=profile)

        logger.info(" Compiling done  - in %.2f s" % (time.time() - t_init))
        self.func = tf


def getinput_for_multioutput(outputs):
    """
    For list of several output layers return a list of required
    input tensors (without duplicates) to compute these outputs.
    """
    inputs = [out.input_tensors for out in outputs]
    ret = OrderedDict()
    for inp in inputs:
        for i in inp:
            ret[i] = True

    return list(ret.keys())


def as_floatX(x):
    if not hasattr(x, '__len__'):
        return np.array(x, dtype=floatX)
    return np.ascontiguousarray(x, dtype=floatX)


