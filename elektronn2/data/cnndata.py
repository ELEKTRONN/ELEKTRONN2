# -*- coding: utf-8 -*-
# ELEKTRONN2 Toolkit
# Copyright (c) 2015 Marius Killinger
# All rights reserved

from __future__ import absolute_import, division, print_function
from builtins import filter, hex, input, int, map, next, oct, pow, range, super, zip

__all__ = ['AgentData', 'BatchCreatorImage', 'GridData']

import gc
import logging
import os
import sys
import time
import getpass

try:
    from importlib import reload
except ImportError:
    pass  # reload() is built-in in Python 2

import h5py
import numpy as np
import tqdm

from . import transformations
from .image import make_affinities, make_nhood_targets
from .. import utils
from ..config import config

logger = logging.getLogger('elektronn2log')
inspection_logger = logging.getLogger('elektronn2log-inspection')

user_name = getpass.getuser()

###############################################################################


def greyAugment(d, channels, rng):
    """
    Performs grey value (histogram) augmentations on ``d``. This is only
    applied to ``channels`` (list of channels indices), ``rng`` is a random
    number generator
    """
    if channels == []:
        return d
    else:
        k = len(channels)
        d = d.copy()  # d is still just a view, we don't want to change the original data so copy it
        alpha = 1 + (rng.rand(k) - 0.5) * 0.3 # ~ contrast
        c     = (rng.rand(k) - 0.5) * 0.3 # mediates whether values are clipped for shadows or lights
        gamma = 2.0 ** (rng.rand(k) * 2 - 1) # sample from [0.5,2] with mean 0

        d[channels] = d[channels] * alpha[:,None,None] + c[:,None,None]
        d[channels] = np.clip(d[channels], 0, 1)
        d[channels] = d[channels] ** gamma[:,None,None]
    return d


def border_treatment(data_list, ps, border_mode, ndim):
    def treatArray(data):
        if border_mode=='keep':
            return data

        sh = data.shape[1:] # (z,y,x)/(x,y)


        if border_mode=='crop':
            excess = [int((x[0] - x[1])//2) for x in zip(sh, ps)]
            if ndim == 3:
                data = data[:,
                            excess[0]:excess[0]+ps[0],
                            excess[1]:excess[1]+ps[1],
                            excess[2]:excess[2]+ps[2]]
            elif ndim==2:
                data = data[:,
                            :,
                            excess[0]:excess[0]+ps[0],
                            excess[1]:excess[1]+ps[1]]

        else:
            excess_l = [int(np.ceil(float(x[0] - x[1])/2)) for x in zip(ps, sh)]
            excess_r = [int(np.floor(float(x[0] - x[1])/2)) for x in zip(ps, sh)]
            if ndim == 3:
                pad_with = [(0,0),
                            (excess_l[0],excess_r[0]),
                            (excess_l[1],excess_r[1]),
                            (excess_l[2],excess_r[2])]
            else:
                pad_with = [(0,0),
                            (0,0),
                            (excess_l[0],excess_r[0]),
                            (excess_l[1],excess_r[1])]

            if border_mode=='mirror':
                data = np.pad(data, pad_with, mode='symmetric')

            if border_mode=='0-pad':
                data = np.pad(data, pad_with, mode='constant', constant_values=0)

        return data

    return [treatArray(d) for d in data_list]


class BatchCreatorImage(object):
    def __init__(self, input_node, target_node=None, d_path=None, l_path=None,
                 d_files=None, l_files=None, cube_prios=None, valid_cubes=None,
                 border_mode='crop', aniso_factor=2, target_vec_ix=None,
                 target_discrete_ix=None, h5stream=False):

        assert (d_path and l_path and d_files and l_files)
        if len(d_files)!=len(l_files):
            raise ValueError("d_files and l_files must be lists of same length!")
        d_path = os.path.expanduser(d_path)
        l_path = os.path.expanduser(l_path)

        self.h5stream = h5stream
        self.d_path = d_path
        self.l_path = l_path
        self.d_files = d_files
        self.l_files = l_files
        self.cube_prios = cube_prios
        self.valid_cubes = valid_cubes if valid_cubes is not None else []
        self.aniso_factor  = aniso_factor
        self.border_mode = border_mode
        self.target_vec_ix = target_vec_ix
        self.target_discrete_ix = target_discrete_ix

        # Infer geometric info from input/target shapes
        self.ndim = input_node.shape.ndim
        self.patch_size = np.array(input_node.shape.spatial_shape, dtype=np.int)
        self.strides = np.array(target_node.shape.strides, dtype=np.int)
        self.offsets = np.array(target_node.shape.offsets, dtype=np.int)
        self.target_ps = self.patch_size - self.offsets * 2
        self.t_dtype = target_node.output.dtype
        if input_node.shape.ndim==target_node.shape.ndim:
            self.mode = 'img-img'
        else:
            if target_node.shape.ndim > 1:
                raise ValueError() ###TODO would it work to map 'img-vect'?
            self.mode = 'img-scalar'

        # The following will be inferred when reading data
        self.n_labelled_pixel = 0
        self.n_f     = None # number of channels/feature in input
        self.t_n_f  = None # the shape of the returned label batch at index 1

        # Actual data fields
        self.valid_d = []
        self.valid_l = []
        self.valid_extra = []

        self.train_d = []
        self.train_l = []
        self.train_extra = []

        # Setup internal stuff
        self.rng = np.random.RandomState(np.uint32((time.time() * 0.0001 -
                                                    int(
                                                        time.time() * 0.0001)) * 4294967295))
        self.pid = os.getpid()
        self.gc_count = 1

        self._sampling_weight = None
        self._training_count = None
        self._valid_count = None
        self.n_successful_warp = 0
        self.n_failed_warp = 0

        self.load_data()
        logger.info('{}\n'.format(self.__repr__()))

    def __repr__(self):
        s = "{0:,d}-target Data Set with {1:,d} input channel(s):\n"+\
        "#train cubes: {2:,d} and #valid cubes: {3:,d}, {4:,d} labelled "+\
        "pixels."
        s = s.format(self.t_n_f, self.n_f, self._training_count,
                                  self._valid_count, self.n_labelled_pixel)
        return s

    @property
    def warp_stats(self):
        return "Warp stats: successful: %i, failed %i, quota: %.1f" %(
            self.n_successful_warp, self.n_failed_warp,
            float(self.n_successful_warp)/(self.n_failed_warp+self.n_successful_warp))

    def _reseed(self):
        """Reseeds the rng if the process ID has changed!"""
        current_pid = os.getpid()
        if current_pid!=self.pid:
            self.pid = current_pid
            self.rng.seed(np.uint32((time.time()*0.0001 -
                                     int(time.time()*0.0001))*4294967295+self.pid))
            reload(transformations) # needed for numba GIL released stuff to work
            logger.debug("Reseeding RNG in Process with PID: {}".format(self.pid))

    def _allocbatch(self, batch_size):
        images = np.zeros((batch_size, self.n_f,)+tuple(self.patch_size), dtype='float32')
        sh = self.patch_size - self.offsets * 2
        target  = np.zeros((batch_size, self.t_n_f)+tuple(sh), dtype=self.t_dtype)
        return images, target


    def getbatch(self, batch_size=1, source='train',
                 grey_augment_channels=None, warp=False, warp_args=None,
                 ignore_thresh=False, force_dense=False,
                 affinities=False, nhood_targets=False, ret_ll_mask=False):
        """
        Prepares a batch by randomly sampling, shifting and augmenting
        patches from the data

        Parameters
        ----------
        batch_size: int
            Number of examples in batch (for CNNs often just 1)
        source: str
            Data set to draw data from: 'train'/'valid'
        grey_augment_channels: list
            List of channel indices to apply grey-value augmentation to
        warp: bool or float
            Whether warping/distortion augmentations are applied to examples
            (slow --> use multiprocessing). If this is a float number,
            warping is applied to this fraction of examples e.g. 0.5 --> every
            other example.
        warp_args: dict
            Additional keyword arguments that get passed through to
            elektronn2.data.transformations.get_warped_slice()
        ignore_thresh: float
            If the fraction of negative targets in an example patch exceeds this
            threshold, this example is discarded (Negative targets are ignored
            for training [but could be used for unsupervised target propagation]).
        force_dense: bool
            If True the targets are *not* sub-sampled according to the CNN output\
            strides. Dense targets requires MFP in the CNN!
        affinities
        nhood_targets
        ret_ll_mask: bool
            If True additional information for reach batch example is returned.
            Currently implemented are two ll_mask arrays to indicate the targeting mode.
            The first dimension of those arrays is the batch_size!

        Returns
        -------
        data: np.ndarray
            [bs, ch, x, y] or [bs, ch, z, y, x] for 2D and 3D CNNS
        target: np.ndarray
            [bs, ch, x, y] or [bs, ch, z, y, x]
        ll_mask1: np.ndarray
            (optional) [bs, n_target]
        ll_mask2: np.ndarray
            (optional) [bs, n_target]
        """
        # This is especially required for multiprocessing
        if grey_augment_channels is None:
            grey_augment_channels = []
        self._reseed()
        images, target = self._allocbatch(batch_size)
        ll_masks = []
        patch_count = 0
        while patch_count < batch_size: # Loop to fill up batch with examples
            d, t, ll_mask = self._getcube(source) # get cube randomly

            try:
                if self.mode=='img-img':
                    d, t = self.warp_cut(d, t, warp, warp_args)
                else:
                    d, _ = self.warp_cut(d, None, warp, warp_args)
                self.n_successful_warp += 1

            except transformations.WarpingOOBError:
                    self.n_failed_warp += 1
                    continue

            # Check only if a ignore_thresh is set and the cube is labelled
            if (ignore_thresh is not False) and (not np.any(ll_mask[1])):
                if (t<0).mean() > ignore_thresh:
                    continue # do not use cubes which have no information

            if source == "train": # no grey augmentation for testing
                d = greyAugment(d, grey_augment_channels, self.rng)

            target[patch_count] = t
            images[patch_count] = d
            ll_masks.append(ll_mask)
            patch_count += 1

        # Patch / Nhood target stuff
        if nhood_targets is not None and nhood_targets is not False:
            if isinstance(nhood_targets, np.ndarray):
                nhood = nhood_targets
            else:
                nhood = np.array([[0, 0, 0],
                                  [1, 0, 0],
                                  [-1, 0, 0],
                                  [0, 1, 0],
                                  [0, -1, 0],
                                  [0, 0, 1],
                                  [0, 0, -1]], dtype=np.int)

            if target.shape[1]==1:
                target = make_nhood_targets(target, nhood)
            else:  # Nhood targets for first feature only, keep other targets
                tmp = make_nhood_targets(target[:, 0:1], nhood)
                target = np.concatenate([tmp, target[:, 1:]], axis=1)


        # Final modification of targets: striding and replacing nan
        if not (force_dense or np.all(self.strides==1)):
            target = self._stridedtargets(target)

        ret = [images, target]  # The "normal" batch

        # MALIS stuff
        if affinities == 'malis':
            # create aff from seg IDs (i.e. shape[1] must be 1)
            aff, seg = make_affinities(target[:, 0])  # [bs, 3, z, y, x]
            seg = seg[:, None]  # add 'class' dimension
            ret = [images, aff, seg]
        elif affinities == 'affinity':
            aff, seg = make_affinities(target)  # [bs, z, y, x, 3]
            ret = [images, aff]

        # Lazy Labels stuff
        if ret_ll_mask:  # ll_mask is now a list(bs) of tuples(2)
            ll_masks = np.atleast_3d(np.array(ll_masks, dtype=np.int16))  # (bs, 2, 5)
            ll_mask1 = ll_masks[:, 0]
            ll_mask2 = ll_masks[:, 1]
            ret += [ll_mask1, ll_mask2]

        target[np.isnan(target)] = -666  # hack because theano does not support nan


        self.gc_count += 1
        if self.gc_count%1000==0:
            gc.collect()

        return tuple(ret)

    def warp_cut(self, img, target, warp, warp_params):
        """
        (Wraps :py:meth:`elektronn2.data.transformations.get_warped_slice()`)
        
        Cuts a warped slice out of the input and target arrays.
        The same random warping transformation is each applied to both input
        and target.
        
        Warping is randomly applied with the probability defined by the ``warp``
        parameter (see below).
        
        Parameters
        ----------
        img: np.ndarray
            Input image
        target: np.ndarray
            Target image
        warp: float or bool
            False/True disable/enable warping completely.
            If ``warp`` is a float, it is used as the ratio of inputs that
            should be warped.
            E.g. 0.5 means approx. every second call to this function actually
            applies warping to the image-target pair.
        warp_params: dict
            kwargs that are passed through to
            :py:meth:`elektronn2.data.transformations.get_warped_slice()`.
            Can be empty.
        
        Returns
        -------
        d: np.ndarray
            (Warped) input image slice
        t: np.ndarray
            (Warped) target slice
        """
        if (warp is True) or (warp==1): # always warp
            do_warp=True
        elif (0 < warp < 1): # warp only a fraction of examples
            do_warp=True if (self.rng.rand()<warp) else False
        else: # never warp
            do_warp=False

        if not do_warp:
            warp_params = dict(warp_params)
            warp_params['warp_amount'] = 0

        d, t = transformations.get_warped_slice(img, self.patch_size,
                            aniso_factor=self.aniso_factor, target=target,
                            target_ps=self.target_ps, target_vec_ix=self.target_vec_ix,
                            target_discrete_ix=self.target_discrete_ix,
                                                rng=self.rng, **warp_params)

        return d, t


    def _getcube(self, source):
        """
        Draw an example cube according to sampling weight on training data,
        or randomly on valid data
        """
        if source=='train':
            p = self.rng.rand()
            i = np.flatnonzero(self._sampling_weight <= p)[-1]
            d, t, ll_mask = self.train_d[i], self.train_l[i], self.train_extra[i]
        elif source == "valid":
            if len(self.valid_d) == 0:
                logger.info("Validation Set empty. Disable testing on validation set.")
                raise ValueError("No validation set")

            i = self.rng.randint(0, len(self.valid_d))
            d = self.valid_d[i]
            t = self.valid_l[i]
            ll_mask = self.valid_extra[i]
        else:
            raise ValueError("Unknown data source")

        return d, t, ll_mask


    def _stridedtargets(self, lab):
        if self.ndim == 3:
            return lab[:, :, ::self.strides[0], ::self.strides[1], ::self.strides[2]]
        elif self.ndim == 2:
            return lab[:, :, ::self.strides[0], ::self.strides[1]]

    def load_data(self):
        """
        Parameters
        ----------

        d_path/l_path: string
          Directories to load data from
        d_files/l_files: list
          List of data/target files in <path> directory (must be in the same order!).
          Each list element is a tuple in the form
          **(<Name of h5-file>, <Key of h5-dataset>)**
        cube_prios: list
          (not normalised) list of sampling weights to draw examples from
          the respective cubes. If None the cube sizes are taken as priorities.
        valid_cubes: list
          List of indices for cubes (from the file-lists) to use as validation
          data and exclude from training, may be empty list to skip performance
          estimation on validation data.
        """
        # returns lists of cubes, ll_mask is a tuple per cube
        data, target, extras = self.read_files()

        if self.mode=='img-scalar':
            data = border_treatment(data, self.patch_size, self.border_mode,
                                    self.ndim)

        default_extra = (np.ones(self.t_n_f), np.zeros(self.t_n_f))
        extras = [default_extra if x is None else x for x in extras]

        prios = []
        # Distribute Cubes into training and valid list
        for k, (d, t, e) in enumerate(zip(data, target, extras)):
            if k in self.valid_cubes:
                self.valid_d.append(d)
                self.valid_l.append(t)
                self.valid_extra.append(e)
            else:
                self.train_d.append(d)
                self.train_l.append(t)
                self.train_extra.append(e)
                # If no priorities are given: sample proportional to cube size
                prios.append(t.size)

        if self.cube_prios is None:
            prios = np.array(prios, dtype=np.float)
        else: # If priorities are given: sample irrespective of cube size
            prios = np.array(self.cube_prios, dtype=np.float)

        # sample example i if: batch_prob[i] < p
        self._sampling_weight = np.hstack((0, np.cumsum(prios / prios.sum())))
        self._training_count = len(self.train_d)
        self._valid_count = len(self.valid_d)


    def check_files(self):
        """
        Check if file paths in the network config are available.
        """
        notfound = False
        give_neuro_data_hint = False
        fullpaths = [os.path.join(self.d_path, f) for f, _ in self.d_files] +\
                    [os.path.join(self.l_path, f) for f, _ in self.l_files]
        for p in fullpaths:
            if not os.path.exists(p):
                print('{} not found.'.format(p))
                notfound = True
                if 'neuro_data_zxy' in p:
                    give_neuro_data_hint = True
        if give_neuro_data_hint:
            print('\nIt looks like you are referencing the neuro_data_zxy dataset.\n'
                  'To install the neuro_data_xzy dataset to the default location, run:\n'
                  '  $ wget http://elektronn.org/downloads/neuro_data_zxy.zip\n'
                  '  $ unzip neuro_data_zxy.zip -d ~/neuro_data_zxy')
        if notfound:
            print('\nPlease fetch the necessary dataset and/or '
                  'change the relevant file paths in the network config.')
            sys.stdout.flush()
            sys.exit(1)

    def read_files(self):
        """
        Image files on disk are expected to be in order (ch,x,y,z) or (x,y,z)
        But image stacks are returned as (z,ch,x,y) and target as (z,x,y,)
        irrespective of the order in the file. If the image files have no
        channel this dimension is extended to a singleton dimension.
        """
        self.check_files()
        data, target, extras = [], [], []
        pbar = tqdm.tqdm(total=len(self.d_files), ncols=120, leave=False)


        for (d_f, d_key), (l_f, l_key) in zip(self.d_files, self.l_files):
            pbar.write('Loading %s and %s' % (d_f,l_f))
            if self.h5stream:
                d = h5py.File(os.path.join(self.d_path, d_f), 'r')[d_key]
                t = h5py.File(os.path.join(self.l_path, l_f), 'r')[l_key]
                assert d.compression==t.compression==None
                assert len(d.shape)==len(t.shape)==4
                assert d.dtype==np.float32
                assert t.dtype==self.t_dtype

            else:
                d = utils.h5load(os.path.join(self.d_path, d_f), d_key)
                t = utils.h5load(os.path.join(self.l_path, l_f), l_key)

            try:
                ll_mask_1 = utils.h5load(os.path.join(self.l_path, l_f), 'll_mask')
                extras.append(ll_mask_1)
            except KeyError:
                extras.append(None)

            if self.mode == 'img-scalar':
                assert t.ndim == 1, "Scalar targets must be 1d"

            if len(d.shape) == 4: # h5 dataset has no ndim
                self.n_f = d.shape[0]
            elif len(d.shape) == 3:  # We have no channels in data
                self.n_f = 1
                d = d[None]  # add (empty) 0-axis


            if len(t.shape) == 3: # If labels not empty add first axis
                t = t[None]

            if self.t_n_f is None:
                self.t_n_f = t.shape[0]
            else:
                assert self.t_n_f == t.shape[0]


            self.n_labelled_pixel += t[0].size

            # determine normalisation depending on int or float type
            if d.dtype in [np.int, np.int8, np.int16, np.int32, np.uint32,
                           np.uint, np.uint8, np.uint16, np.uint32, np.uint32]:
                m = 255
                d = np.ascontiguousarray(d, dtype=np.float32) / m

            if (np.dtype(self.t_dtype) is not np.dtype(t.dtype)) and \
                self.t_dtype not in ['float32']:
                m = t.max()
                M = np.iinfo(self.t_dtype).max
                if m  > M:
                    raise ValueError("Loading of data: targets must be cast "
                                     "to %s, but %s cannot store value %g, "
                                     "maximum allowed value: %g. You may try "
                                     "to renumber targets." %(self.t_dtype,
                                                             self.t_dtype, m, M))
            if not self.h5stream:
                d = np.ascontiguousarray(d, dtype=np.float32)
                t = np.ascontiguousarray(t, dtype=self.t_dtype)

            pbar.write('Shapes: data %s, targets %s' % (d.shape, t.shape))

            data.append(d)
            target.append(t)
            gc.collect()
            pbar.update()

        return data, target, extras

"""
3d Label kinds:
k: number of classes
n_c: number of categories (each of which has n_k classes)

normal single category multiclass classification (e.g. bg,syn,mito,ves):
    (x,y,z)   encoded_targets==True, n_targets==k
    (1,x,y,z) encoded_targets==True, n_targets==k
    (k,x,y,z) encoded_targets==False, n_targets==k
multicategory multiclass classification (e.g. affinities):
    (n_c, x,y,z)  encoded_targets==True, n_targets=k
    (n_c*k,x,y,z) encoded_targets==False, n_targets=k,
single target regression (e.g. image reconstruction of gray img):
    (x,y,z)   encoded_targets indiff, n_targets=1
    (1,x,y,z) encoded_targets indiff, n_targets=1
multi target regression (e.g. image reconstruction of rgb img):
    (k,x,y,z) n_targets=k

--> l.shape[1] = ?
"""


##############################################################################################################
class DummySkel(object):
    def __init__(self):
        self.grad = np.array([0,0,0], dtype=np.float32)
        self.last_slicing_params = None
        self.lost_track = True # don't train sequentially on this
        self.debug_traces = []
        self.debug_traces_current = []
        self.debug_grads = []
        self.debug_grads_current = []

    def _grad(self, pred_zxy):
        # L= 1/2(x**2 + y**2)
        # dL/dp = [0, x, y]
        return np.array([0,pred_zxy[1],pred_zxy[2]], dtype=np.float32)

    def get_loss_and_gradient(self, prediction_c, **kwargs):
        loss = abs(prediction_c[1:]).sum() # sum of x,y magnitude
        loss = np.array([loss, ], dtype=np.float32)

        self.debug_traces_current.append(None)
        self.debug_grads_current.append(None)

        return loss, self._grad(prediction_c)

class AgentData(BatchCreatorImage):
    """
    Load raw_cube, vec_prob_obj_cube and skelfiles + rel.offset

    """
    def __init__(self, input_node, side_target_node, path_prefix=None,
                 raw_files=None, skel_files=None, vec_files=None, valid_skels=None,
                 target_vec_ix=None, target_discrete_ix=None,
                 abs_offset=None, aniso_factor=2):

        assert (path_prefix and raw_files and skel_files and vec_files and abs_offset)
        abs_offset = np.array(abs_offset, dtype=np.int)


        path_prefix = os.path.expanduser(path_prefix)
        self.path_prefix = path_prefix
        self.raw_files = raw_files
        self.skel_files = skel_files
        self.vec_files = vec_files
        self.valid_skels = valid_skels if valid_skels is not None else []
        self.abs_offset = abs_offset
        self.aniso_factor  = aniso_factor
        self.target_vec_ix = target_vec_ix
        self.target_discrete_ix = target_discrete_ix

        # Infer geometric info from input/target shapes
        self.ndim = input_node.shape.ndim
        self.patch_size = np.array(input_node.shape.spatial_shape, dtype=np.int)
        self.strides = np.array(side_target_node.shape.strides, dtype=np.int)
        self.offsets = np.array(side_target_node.shape.offsets, dtype=np.int)
        self.target_ps = self.patch_size - self.offsets * 2
        #self.n_ch_model = input_node.shape['f']
        self.n_f       = None # number of channels/feature in input
                              # will be inferred when reading data
        self.t_n_f  = None # the shape of the returned label batch at index 1
        self.t_dtype = side_target_node.output.dtype
        # Need to infer this when reading labels and with n_target/encoded_targets

        # Setup internal stuff
        self.rng = np.random.RandomState(np.uint32((time.time()*0.0001 -
                                            int(time.time()*0.0001))*4294967295))
        self.pid = os.getpid()
        self.gc_count = 1
        self.n_failed_warp = 0
        self.n_successful_warp = 0

        self._sampling_weight = None
        self._training_count  = None
        self._valid_count     = None

        # Actual data fields
        self.train_d = []
        self.train_l = []
        self.valid_s = []
        self.train_s = []

        assert len(raw_files)==len(skel_files)==len(vec_files)==1, "not supported atm, skeletons " \
                                                                   "from different cubes are concated, " \
                                                                   "only one abs offset"
        self.load_data()
        logger.info('{}\n'.format(self.__repr__()))

    def __repr__(self):
        s = "Data Set with {0:,d} input channel(s):\n"+\
        "#train skels: {1:,d} and #valid skels: {2:,d}"
        s = s.format(self.n_f, len(self.train_s),
                     len(self.valid_s))
        return s

    def _allocbatch(self, batch_size):
        images, target = super(AgentData, self)._allocbatch(batch_size)
        skel_i = np.zeros(batch_size, dtype='int16')
        slicing_params = np.zeros([batch_size,20], dtype='float32')
        return images, target, skel_i, slicing_params

    def getbatch(self, batch_size=1, source='train', aniso=True,
                  z_shift=0, gamma=0,
                  grey_augment_channels=None,
                  r_max_scale=0.9,
                  tracing_dir_prior_c=0.5,
                  force_dense=False, flatfield_p=1e-3):

        # This is especially required for multiprocessing
        if grey_augment_channels is None:
            grey_augment_channels = []
        self._reseed()
        images, target, skel_i, slicing_params = self._allocbatch(batch_size)
        patch_count = 0
        while patch_count < batch_size: # Loop to fill up batch with examples
            if (self.rng.rand() > flatfield_p) or source!='train':
                skel, i = self.getskel(source) # get skel randomly
                position_s = skel.sample_tube_point(self.rng, r_max_scale=r_max_scale)
                local_direc_is = skel.sample_local_direction_iso(position_s, n_neighbors=6)
                tracing_direc_is = skel.sample_tracing_direction_iso(self.rng, local_direc_is, c=tracing_dir_prior_c)

                position_l = position_s[::-1]  # from lab to data frame (xyz) -> (zxy)
                position_d = position_l - self.abs_offset
                tracing_direc_il = tracing_direc_is[::-1]         # from lab to data frame (xyz) -> (zxy)

                try:
                    if config.inspection > 2: # mark position by opaque cube
                        self.train_d[0][:,
                            position_d[0] - 1:position_d[0] + 2,
                            position_d[1] - 2:position_d[1] + 3,
                            position_d[2] - 2:position_d[2] + 3] = 1.0

                    image, vec, M = transformations.get_tracing_slice(
                    self.train_d[0], self.patch_size, position_d, z_shift=z_shift,
                    aniso_factor=self.aniso_factor, sample_aniso=aniso,
                    gamma=gamma, direction_iso=tracing_direc_il,
                    target=self.train_l[0], target_ps=self.target_ps,
                    target_vec_ix=self.target_vec_ix,
                    target_discrete_ix=self.target_discrete_ix, rng=self.rng)
                    self.n_successful_warp += 1

                    trafo = transformations.Transform(M, position_l=position_l,
                                                      aniso_factor=self.aniso_factor)

                except transformations.WarpingOOBError:
                    self.n_failed_warp += 1
                    continue

                if source == "train": # no grey augmentation for testing
                    image    = greyAugment(image, grey_augment_channels, self.rng)

            else:
                vec_sh = (self.train_l[0].shape[0],)+tuple(self.target_ps)
                vec = np.ones(vec_sh, dtype=self.t_dtype) * np.nan
                img_sh = (self.train_d[0].shape[0],)+tuple(self.patch_size)
                image = np.zeros(img_sh, dtype=np.float32)
                i = len(self.train_s) - 1
                M = np.eye(4)
                position_l = np.zeros(3, np.int)
                position_d = position_l
                local_direc_is = position_l
                tracing_direc_il = position_l
                trafo = transformations.Transform(M, position_l=position_l,
                                                  aniso_factor=self.aniso_factor)

            target[patch_count] = vec
            images[patch_count] = image
            skel_i[patch_count] = i
            slicing_params[patch_count] = trafo.to_array()
            patch_count += 1

        target[np.isnan(target)] = -666 # Stupid hack because theano does not support nan
        if not (force_dense or np.all(self.strides==1)):
            target = self._stridedtargets(target)

        ret = [images, target, skel_i, slicing_params]

        if config.inspection > 2:
            debug_img = self.train_d[0][:,
                        position_d[0] - 11:position_d[0] + 12,
                        position_d[1] - 21:position_d[1] + 22,
                        position_d[2] - 21:position_d[2] + 22]
            if i == len(self.train_s) - 1:
                i = "DummySkel"

            local_direc_il = local_direc_is[::-1]
            inspection_logger.info("skeleton: %s" % i)
            inspection_logger.info("pos_d: %s, pos_l: %s, ldir_il: %s, tdir_il: %s"
                                   % (position_d.tolist(),
                                      position_l.tolist(),
                                      local_direc_il.tolist(),
                                      tracing_direc_il.tolist()))

            inspection_logger.info("params: %s" % (trafo.to_array().tolist()))

            ret = [images, target, skel_i, slicing_params, debug_img]

        self.gc_count += 1
        if self.gc_count%1000==0:
            gc.collect()

        return tuple(ret)


    def get_newslice(self, position_l, direction_il, batch_size=1, source='train',
                     aniso=True, z_shift=0, gamma=0, grey_augment_channels=None,
                     r_max_scale=0.9, tracing_dir_prior_c=0.5, force_dense=False,
                     flatfield_p=1e-3, scale=1.0, last_ch_max_interp=False):
        assert batch_size==1
        if grey_augment_channels is None:
            grey_augment_channels = []
        if source!='train':
            raise ValueError

        images, target, skel_i, slicing_params = self._allocbatch(batch_size)

        position_d = position_l - self.abs_offset
        if config.inspection > 2:  # mark position by opaque cube
            self.train_d[0][:,
            position_d[0] - 2:position_d[0] + 3,
            position_d[1] - 4:position_d[1] + 5,
            position_d[2] - 4:position_d[2] + 5] = 1.0

        image, vec, M = transformations.get_tracing_slice(
            self.train_d[0], self.patch_size, position_d,
            z_shift=z_shift, aniso_factor=self.aniso_factor, sample_aniso=aniso,
            gamma=gamma, scale_factor=scale, direction_iso=direction_il,
            target=self.train_l[0], target_ps=self.target_ps,
            target_vec_ix=self.target_vec_ix,
            target_discrete_ix=self.target_discrete_ix, rng=self.rng,
            last_ch_max_interp=last_ch_max_interp)

        trafo = transformations.Transform(M, position_l=position_l,
                                           aniso_factor=self.aniso_factor)

        image = greyAugment(image, grey_augment_channels, self.rng)

        if vec is not None: # allow for using this with no vec data
            target[0] = vec

        images[0] = image

        target[np.isnan(target)] = -666  # Stupid hack because theano does not support nan
        if not (force_dense or np.all(self.strides==1)):
            target = self._stridedtargets(target)

        ret = [images, target, trafo]

        if config.inspection > 2:
            debug_img = self.train_d[0][:,
                        position_d[0] - 11:position_d[0] + 12,
                        position_d[1] - 21:position_d[1] + 22,
                        position_d[2] - 21:position_d[2] + 22]

            # inspection_logger.info("skeleton: 999")
            # inspection_logger.info("pos_d: %s, pos_l: %s, ldir_il: %s, tdir_il: %s"
            #                        % (position_d.tolist(),
            #                           position_l.tolist(),
            #                           direction_il.tolist(),
            #                           direction_il.tolist()))
            # inspection_logger.info("params: %s" % (trafo.to_array().tolist()))
            image_ref, _,_= transformations.get_tracing_slice(
                self.train_d[0], self.patch_size, position_d,
                z_shift=z_shift, aniso_factor=self.aniso_factor,
                sample_aniso=aniso,
                gamma=gamma, scale_factor=1.0, direction_iso=direction_il,
                target=self.train_l[0], target_ps=self.target_ps,
                target_vec_ix=self.target_vec_ix,
                target_discrete_ix=self.target_discrete_ix, rng=self.rng)
            i = np.random.randint(0, 1014)
            utils.h5save(image, '/tmp/%s_img%i-%.2f.h5'%(user_name, i,scale))
            utils.h5save(image_ref, '/tmp/%s_img%i-ref.h5' % (user_name, i))
            #utils.h5save(debug_img, '/tmp/dbg-img-%.2f.h5' % scale)
            ret = [images, target, trafo, debug_img]

        return tuple(ret)


    def getskel(self, source):
        """
        Draw an example skeleton according to sampling weight on training data,
        or randomly on valid data
        """
        if source=='train':
            p = self.rng.rand()
            i = np.flatnonzero(self._sampling_weight <= p)[-1]
            skel = self.train_s[i]
        elif source == "valid":
            if len(self.valid_s) == 0:
                logger.info("Validation Set empty. Disable testing on validation set.")
                return []
            i = self.rng.randint(0, len(self.valid_s))
            skel = self.valid_s[i]
        else:
            raise ValueError("Unknown data source")
        return skel, i


    def load_data(self):
        """
        Parameters
        ----------

        d_path/l_path: string
          Directories to load data from
        d_files/l_files: list
          List of data/target files in <path> directory (must be in the same order!).
          Each list element is a tuple in the form
          **(<Name of h5-file>, <Key of h5-dataset>)**
        cube_prios: list
          (not normalised) list of sampling weights to draw examples from
          the respective cubes. If None the cube sizes are taken as priorities.
        valid_cubes: list
          List of indices for cubes (from the file-lists) to use as validation
          data and exclude from training, may be empty list to skip performance
          estimation on validation data.
        """
        # returns lists of cubes, ll_mask is a tuple per cube
        data, vec, skeletons = self.read_files()
        self.train_d = data
        self.train_l = vec
        prios = []
        # Distribute Cubes into training and valid list
        for skel_list, valid_skels in zip(skeletons, self.valid_skels):
            for skel_i, skel in enumerate(skel_list):
                if skel_i in valid_skels:
                    self.valid_s.append(skel)
                else:
                    self.train_s.append(skel)
                    # If no priorities are given: sample proportional to cube size
                    prios.append(len(skel.all_nodes))


        prios = np.array(prios, dtype=np.float)
        # sample example i if: batch_prob[i] < p
        self._sampling_weight = np.hstack((0, np.cumsum(prios / prios.sum())))
        self.train_s.append(DummySkel())


    def read_files(self):
        """
        Image files on disk are expected to be in order (ch,x,y,z) or (x,y,z)
        But image stacks are returned as (z,ch,x,y) and target as (z,x,y,)
        irrespective of the order in the file. If the image files have no
        channel this dimension is extended to a singleton dimension.
        """
        logger.info("Loading Data")
        data, vec, skeletons = [], [], []

        for (d_f, d_key), (l_f, l_key), s_f in zip(self.raw_files, self.vec_files, self.skel_files):
            d = utils.h5load(os.path.join(self.path_prefix, d_f), d_key)
            if d.ndim == 4:
                self.n_f = d.shape[0]
            elif len(d.shape) == 3:  # We have no channels in data
                self.n_f = 1
                d = d[None]  # add (empty) 0-axis



            # determine normalisation depending on int or float type
            if d.dtype in [np.int, np.int8, np.int16, np.int32, np.uint32,
                           np.uint, np.uint8, np.uint16, np.uint32, np.uint32]:
                m = 255
            else:
                m  = 1

            d = np.ascontiguousarray(d, dtype=np.float32) / m

            try:
                t = utils.h5load(os.path.join(self.path_prefix, l_f), l_key)
                if len(t.shape) == 3: # If labels not empty add first axis
                    t = t[None]

                assert t.ndim==4
                self.t_n_f = t.shape[0]
                if (self.t_dtype is not t.dtype) and self.t_dtype not in ['float32']:
                    m = t.max()
                    M = np.iinfo(self.t_dtype).max
                    if m  > M:
                        raise ValueError("Loading of data: targets must be cast "
                                         "to %s, but %s cannot store value %g, "
                                         "maximum allowed value: %g. You may try "
                                         "to renumber targets." %(self.t_dtype,
                                                                 self.t_dtype, m, M))

                t = np.ascontiguousarray(t, dtype=self.t_dtype)
            except:
                t = None
                self.t_n_f = 9

            gc.collect()
            data.append(d)
            vec.append(t)
            s = utils.pickleload(os.path.join(self.path_prefix, s_f))
            skeletons.append(s)
            for skel in s: # backward fixes of attributes that are not pickled
                skel.lost_track = False
                skel.position_s = None
                skel.position_l = None
                skel.direction_il = None
                skel.start_new_training = True
                skel.prev_batch = None
                skel.trafo = None
                skel.training_traces = []
                skel.current_trace = None
                skel.linked_data = self
                skel._hull_point_bg = dict()
                skel.background_processes = config.background_processes
                skel.cnn_grid = None

        return data, vec, skeletons

class GridData(AgentData):
    def __init__(self, *args, **kwargs):
        super(GridData, self).__init__(*args, **kwargs)
        #self.mean_grid = utils.pickleload("~/CNN_Training/mean.pkl")

    def getbatch(self, **get_batch_kwargs):
        self._reseed()
        try:
            skel_example, skel_index = self.getskel('train')
            skel_example.start_new_training = True
            batch = skel_example.getbatch(0, 0.0, **get_batch_kwargs)
            img, target_img, target_grid, target_node= batch
            target_grid = target_grid[None]
            #target_grid -= self.mean_grid[None,None]
            self.last_skel = skel_example #DBG

            return img, target_grid
        except transformations.WarpingOOBError:
            return self.getbatch(**get_batch_kwargs)



