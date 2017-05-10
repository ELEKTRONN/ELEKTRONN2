# -*- coding: utf-8 -*-
"""
Copyright (c) 2015 Marius Killinger, Sven Dorkenwald, Philipp Schubert
All rights reserved
"""
from __future__ import absolute_import, division, print_function
from builtins import filter, hex, input, int, map, next, oct, pow, range, super, zip

__all__ = ['Data', 'MNISTData', 'PianoData', 'PianoData_perc']

import hashlib
import logging
import os
import time
try:
    import urllib.request as urllib2
except ImportError:
    import urllib2

import numpy as np


logger = logging.getLogger('elektronn2log')

import sklearn

from ..utils import pickleload

class Data(object):
    """
    Load and prepare data, Base-Obj
    """
    def __init__(self, n_lab=None):
        self._pos           = 0
        # self.train_d = None
        # self.train_l = None
        # self.valid_d = None
        # self.valid_l = None
        # self.test_d = None
        # self.test_l = None

        if isinstance(self.train_d, np.ndarray):
            self._training_count = self.train_d.shape[0]
            if n_lab is None:
                self.n_lab = np.unique(self.train_l).size
            else:
                self.n_lab = n_lab
        elif isinstance(self.train_d, list):
            self._training_count = len(self.train_d)
            if n_lab is None:
                unique = [np.unique(l) for l in self.train_l]
                self.n_lab = np.unique(np.hstack(unique)).size
            else:
                self.n_lab = n_lab

        if self.example_shape is None:
            self.example_shape = self.train_d[0].shape
        self.n_ch = self.example_shape[0]

        self.rng = np.random.RandomState(np.uint32((time.time()*0.0001 - int(time.time()*0.0001))*4294967295))
        self.pid = os.getpid()
        logger.info(self.__repr__())
        self._perm = self.rng.permutation(self._training_count)


    def _reseed(self):
        """Reseeds the rng if the process ID has changed!"""
        current_pid = os.getpid()
        if current_pid!=self.pid:
            self.pid = current_pid
            self.rng.seed(np.uint32((time.time()*0.0001 - int(time.time()*0.0001))*4294967295+self.pid))
            logger.debug("Reseeding RNG in Process with PID: {}".format(self.pid))


    def __repr__(self):
        return "%i-class Data Set: #training examples: %i and #validing: %i" \
        %(self.n_lab, self._training_count, len(self.valid_d))


    def getbatch(self, batch_size, source='train'):
        if source=='train':
            if (self._pos+batch_size) < self._training_count:
                self._pos += batch_size
                slice = self._perm[self._pos-batch_size:self._pos]
            else: # get new permutation
                self._perm = self.rng.permutation(self._training_count)
                self._pos = 0
                slice = self._perm[:batch_size]

            if isinstance(self.train_d, np.ndarray):
                return (self.train_d[slice], self.train_l[slice])

            elif isinstance(self.train_d, list):
                data  = np.array([self.train_d[i] for i in slice])
                label = np.array([self.train_l[i] for i in slice])
                return (data, label)

        elif source=='valid':
            data  = self.valid_d[:batch_size]
            label = self.valid_l[:batch_size]
            return (data, label)

        elif source=='test':
            data  = self.test_d[:batch_size]
            label = self.test_l[:batch_size]
            return (data, label)

    def createCVSplit(self, data, label, n_folds=3, use_fold=2, shuffle=False, random_state=None):
        try:  # sklearn >=0.18 API
            # (see http://scikit-learn.org/dev/whats_new.html#model-selection-enhancements-and-api-changes)
            import sklearn.model_selection
            kfold = sklearn.model_selection.KFold(
                n_splits=n_folds, shuffle=shuffle, random_state=random_state
            )
            cv = kfold.split(data)
        except:  # sklearn <0.18 API # TODO: We can remove this after a while.
            import sklearn.cross_validation
            cv = sklearn.cross_validation.KFold(
                len(data), n_folds, shuffle=shuffle, random_state=random_state
            )
        for fold, (train_i, valid_i) in enumerate(cv):
            if fold==use_fold:
                self.valid_d = data[valid_i]
                self.valid_l = label[valid_i]
                self.train_d = data[train_i]
                self.train_l = label[train_i]

##########################################################################################

def _augmentMNIST(data, label, crop=2, factor=4):
    """
    Creates new data, by cropping/shifting data.
    Control blow-up by factor and maximum offset by crop
    """
    n = data.shape[-1]
    new_size    = (n-crop)
    new_data    = np.zeros((0,1,new_size,new_size), dtype=np.float32) # store new data in here
    new_label   = np.zeros((0,), dtype=np.int16)
    pos         = [(i%crop, int(i/crop)%crop) for i in range(crop**2)] # offests of different positions
    perm        = np.random.permutation(range(crop**2))

    for i in range(factor): # create <factor> new version of data
        ix =pos[perm[i]]
        new = (data[:, :, ix[0]:ix[0]+new_size, ix[1]:ix[1]+new_size])
        new_data = np.concatenate((new_data, new), axis=0)
        new_label= np.concatenate((new_label, label), axis=0)

    return new_data, new_label


class MNISTData(Data):
    def __init__(self, input_node, target_node,  path=None, convert2image=True,
                 warp_on=False, shift_augment=True, center=True):
        if path is None:
            (self.train_d, self.train_l), (self.valid_d, self.valid_l), (
            self.test_d, self.test_l) = self.download()

        else:
            path = os.path.expanduser(path)
            (self.train_d, self.train_l), (self.valid_d, self.valid_l), (self.test_d, self.test_l) = pickleload(path)

        self.warp_on = warp_on
        self.shif_augment = shift_augment
        self.return_flat  = not convert2image
        self.test_l  = self.test_l.astype(np.int16)
        self.train_l = self.train_l.astype(np.int16)
        self.valid_l = self.valid_l.astype(np.int16)
        self.example_shape = None

        if center:
            self.test_d  -= self.test_d.mean()
            self.train_d -= self.train_d.mean()
            self.valid_d -= self.valid_d.mean()

        self.convert_to_image()
        if self.shif_augment:
            self._stripborder(1)
            self.train_d, self.train_l = _augmentMNIST(self.train_d, self.train_l, crop=2, factor=4)

        self.train_l = self.train_l[:, None]
        self.test_l = self.test_l[:, None]
        self.valid_l = self.valid_l[:, None]
        super(MNISTData, self).__init__()
        if not convert2image:
            self.example_shape = self.train_d[0].size


        logger.info("MNIST data is converted/augmented to shape {}".format(self.example_shape))

    @staticmethod
    def download():
        if os.name == 'nt':
            dest = os.path.join(os.environ['APPDATA'], 'ELEKTRONN2')
        else:
            xdg_cache_home = os.environ.get('XDG_CACHE_HOME', os.path.expanduser('~/.cache'))
            dest = os.path.join(xdg_cache_home, 'ELEKTRONN2')

        if not os.path.exists(dest):
            os.makedirs(dest)

        dest = os.path.join(dest, 'mnist.pkl.gz')

        if os.path.exists(dest):
            print("Found existing MNIST data at {}".format(dest))
        else:
            download_url = "http://www.elektronn.org/downloads/mnist.pkl.gz"
            print("Downloading MNIST data from {}".format(download_url))
            # TODO: We could use a progress bar here.
            response = urllib2.urlopen(download_url)
            data = response.read()
            print("Saving data to {}".format(dest))
            with open(dest, "wb") as f:
                f.write(data)
            response.close()
            # Check if download is complete
            with open(dest, "rb") as f:
                mnist_sha256_abbr = '4b950cea3877c03ea8db5cb4'
                response_sha256 = hashlib.sha256(f.read()).hexdigest()
                if not response_sha256.startswith(mnist_sha256_abbr):
                    raise IOError('{} is corrupted. Please delete it and retry.'.format(dest))

        return pickleload(dest)

    def convert_to_image(self):
        """For MNIST / flattened 2d, single-Layer, square images"""

        valid_size  = self.valid_l.size
        test_size   = self.test_l.size
        data        = np.vstack((self.valid_d, self.test_d, self.train_d))
        size        = data[0].size
        n = int(np.sqrt(size))
        assert abs(n**2-size) < 1e-6 , '<convertToImage> data is not square'
        count = data.shape[0]
        data = data.reshape((count, 1, n, n))
        self.valid_d  = data[:valid_size]
        self.test_d = data[valid_size:valid_size+test_size]
        self.train_d = data[valid_size+test_size:]

    def getbatch(self,batch_size, source='train'):
        if source=='valid':
            ret = super(MNISTData, self).getbatch(batch_size, 'valid')
        if source=='test':
            ret =  super(MNISTData, self).getbatch(batch_size, 'test')
        else:
            d, l = super(MNISTData, self).getbatch(batch_size, source)
            if self.warp_on:
                d = self._warpaugment(d)
            ret = d, l

        if self.return_flat:
            ret = (ret[0].reshape((batch_size, -1)), ret[1])

        return ret

    def _stripborder(self, pix=1):
        s = self.train_d.shape[-1]
        self.valid_d = self.valid_d[:, :, pix:s-pix, pix:s-pix]
        self.test_d  = self.test_d [:, :, pix:s-pix, pix:s-pix]

    def _warpaugment(self, d, amount=1):
        rot_max     = 5  * amount
        shear_max   = 7   * amount
        scale_max   = 1.15 * amount
        stretch_max = 0.25 * amount

        shear = shear_max * 2 * (np.random.rand()-0.5)
        twist   = rot_max * 2 * (np.random.rand()-0.5)
        rot     = 0 # min(rot_max - abs(twist), rot_max  * (np.random.rand()))
        scale   = 1 + (scale_max-1) * np.random.rand(2)
        stretch =  stretch_max * 2 * (np.random.rand(4)-0.5)

        ps = (d.shape[0],)+d.shape[2:]
        raise ValueError("Warping is suspended, reimplement using warp.py")
        #w =  warping.warp3dFast(d, ps, rot, shear, (scale[0], scale[1], 1), stretch, twist)
        return w


class PianoData(Data):
  def __init__(self, input_node, target_node,
               path='/home/mkilling/devel/data/PianoRoll/Nottingham_enc.pkl', n_tap=20, n_lab=58):
    path = os.path.expanduser(path)
    (self.train_d, self.valid_d, self.test_d)  = pickleload(path)
    super(PianoData, self).__init__(n_lab=n_lab)
    self.example_shape = self.train_d[0].shape[-1]
    self.n_taps = n_tap
    self.n_lab = n_lab


  def getbatch(self, batch_size, source='train'):
    if source=='train':
      if (self._pos+batch_size) < self._training_count:
        self._pos += batch_size
        slice = self._perm[self._pos-batch_size:self._pos]
      else: # get new permutation
        self._perm = self.rng.permutation(self._training_count)
        self._pos = 0
        slice = self._perm[:batch_size]

      data  = [self.train_d[i] for i in slice]

    elif source=='valid':
      data  = self.valid_d[:batch_size]

    elif source=='test':
      data  = self.test_d[:batch_size]

    lengths = np.array(map(len, data))
    start_t = np.round(np.random.rand(batch_size)*(lengths-self.n_taps-1)).astype(np.int)
    x = np.array([d[t:t+self.n_taps].astype(np.float32) for d,t in zip(data, start_t)])
    y = np.array([d[t+self.n_taps]                      for d,t in zip(data, start_t)])
    return x, y


class PianoData_perc(PianoData):
  def __init__(self, input_node, target_node,
               path='/home/mkilling/devel/data/PianoRoll/Nottingham_enc.pkl', n_tap=20, n_lab=58):
    super(PianoData_perc, self).__init__(input_node, target_node, path='/home/mkilling/devel/data/PianoRoll/Nottingham_enc.pkl', n_tap=20, n_lab=58)


  def getbatch(self, batch_size, source='train'):
    x, y = super(PianoData_perc, self).getbatch(batch_size, source)
    x = x.swapaxes(0, 1)
    return x.reshape((x.shape[0], -1)), y
