# -*- coding: utf-8 -*-
# ELEKTRONN2 Toolkit
# Copyright (c) 2015 Marius Killinger
# All rights reserved

from __future__ import absolute_import, division, print_function
from builtins import filter, hex, input, int, map, next, oct, pow, range, \
    super, zip

__all__ = ['CircularBuffer', 'AccumulationArray', 'DynamicKDT', 'KDT',
           'pickleload', 'picklesave', 'h5load', 'h5save', 'pretty_string_ops',
           'import_variable_from_file', 'timeit', 'Timer',
           'cache', 'pretty_string_time', 'unique_rows',
           'get_free_cpu_count', 'parallel_accum', 'makeversiondir', 'as_list']

from builtins import filter, hex, input, int, map, next, oct, pow, range, super, zip

import os
import re
import time
import logging
from functools import reduce

try:
    import cPickle as pkl
except:
    import pickle as pkl


import numba
import psutil
from multiprocessing import Pool
import gzip
import h5py
import numpy as np
import sys
import importlib
import functools
from scipy.spatial.distance import cdist
import sklearn
from sklearn.neighbors import NearestNeighbors as NearestNeighbors_


# TODO: Why are there two __all__ assignments? Delete the first one?
__all__ = ['get_free_cpu_count', 'parallel_accum',
           'timeit', 'cache', 'CircularBuffer', 'AccumulationArray', 'KDT',
           'DynamicKDT', 'import_variable_from_file', 'pickleload', 'picklesave',
           'h5save', 'h5load', 'pretty_string_ops', 'pretty_string_time',
           'makeversiondir', 'Timer', 'unique_rows', 'as_list']

logger = logging.getLogger('elektronn2log')


def get_free_cpu_count():
    m = psutil.cpu_count()
    if m<=2:
        return 1
    else:
        load = float(psutil.cpu_percent(interval=0.4)) / 100
        free = 1 - load
        n = min(max(3, m * free * 0.8), m - 1)
        return int(n)


def parallel_accum(func, n_ret, var_args, const_args, proc=-1, debug=False):
    if proc==-1:
        proc = get_free_cpu_count()
    args = []
    for a in var_args:
        try:
            v = tuple(a)
        except TypeError:
            v = (a,)

        arg = v + tuple(const_args)
        args.append(arg)

    assert len(args)>0
    if not debug:
        p = Pool(proc)
        try:
            ret = p.map(func, args)
            p.close()
            p.join()
        except KeyboardInterrupt:
            p.terminate()
            p.join()
            raise KeyboardInterrupt
    else:
        ret = map(func, args)

    accums = [list() for i in range(n_ret)]
    for tmp in ret:
        if n_ret==1:
            try:
                accums[0].extend(tmp)
            except TypeError:
                accums[0].append(tmp)
        else:
            for i in range(n_ret):
                x = tmp[i]
                try:
                    accums[i].extend(x)
                except TypeError:
                    accums[i].append(x)

    if n_ret==1:
        return accums[0]
    else:
        return tuple(accums)


### Decorator Collection ###

class DecoratorBase(object):
    """
    If used as
    ``@DecoratorBase``
    this initialiser receives only the function to be wrapped (no wrapper args)
    Then ``__call__`` receives the arguments for the underlying function.

    Alternatively, if used as
    ``@DecoratorBase(wrapper_print=True, n_times=10)``
    this initialiser receives wrapper args, the function is passed to ``__call__``
    and ``__call__`` returns a wrapped function.

    This base class completely ignores all wrapper arguments.
    """

    def __init__(self, *args, **kwargs):
        self.func = None
        self.dec_args = None
        self.dec_kwargs = None
        if len(args)==1 and not len(kwargs):
            assert hasattr(args[0], '__call__')
            func = args[0]
            self.func = func
            self.__call__.__func__.__doc__ = func.__doc__
            self.__call__.__func__.__name__ = func.__name__
        else:
            self.dec_args = args
            self.dec_kwargs = kwargs

    def __call__(self, *args, **kwargs):
        # The decorator was initialised with the func, it now has apply the decoration itself
        if not self.func is None:
            # do something with args
            ret = self.func(*args, **kwargs)
            # do something with kwargs
            return ret

        # The decorator was initialised with args, it now returns a wrapped function
        elif len(args)==1 and not len(kwargs):
            assert hasattr(args[0], '__call__')
            func = args[0]

            @functools.wraps(func)
            def decorated(*args0, **kwargs0):
                # do something with args0, read the decorator arguments
                # print(self.dec_args)
                # print(self.dec_kwargs)
                ret = func(*args0, **kwargs0)
                # do something with ret
                return ret

            return decorated
        else:
            raise ValueError()


class timeit(DecoratorBase):
    def __call__(self, *args, **kwargs):
        # The nor args for the decorator --> n=1
        if not self.func is None:
            t0 = time.time()
            ret = self.func(*args, **kwargs)
            t = time.time() - t0
            print("Function <%s> took %.5g s" % (self.func.__name__, t))
            return ret

        # The decorator was initialised with args, it now returns a wrapped function
        elif len(args)==1 and not len(kwargs):
            assert hasattr(args[0], '__call__')
            func = args[0]
            n = self.dec_kwargs.get('n', 1)

            @functools.wraps(func)
            def decorated(*args0, **kwargs0):
                t0 = time.time()
                if n>1:
                    for i in range(n - 1):
                        func(*args0, **kwargs0)

                ret = func(*args0, **kwargs0)
                t = time.time() - t0
                print("Function <%s> took %.5g s averaged over %i execs" % (
                    func.__name__, t / n, n))

                return ret

            return decorated

        else:
            raise ValueError()


class cache(DecoratorBase):
    def __init__(self, *args, **kwargs):
        super(cache, self).__init__(*args, **kwargs)
        self.memo = {}
        self.default = None

    @staticmethod
    def hash_args(args):
        tmp = []
        for arg in args:
            if isinstance(arg, np.ndarray):
                tmp.append(hash(arg.tostring()))
            elif isinstance(arg, (list, tuple)):
                tmp.append(reduce(lambda x, y: x + hash(y), arg, 0))
            else:
                tmp.append(hash(arg))

        return reduce(lambda x, y: x + y, tmp, 0)

    def __call__(self, *args, **kwargs):
        # The nor args for the decorator --> n=1
        if not self.func is None:
            if len(args)==0 and len(kwargs)==0:
                if self.default is None:
                    self.default = self()
                return self.default()
            else:
                key1 = self.hash_args(args)
                key2 = self.hash_args(kwargs.values())
                key = key1 + key2
                if not key in self.memo:
                    self.memo[key] = self.func(*args, **kwargs)
                return self.memo[key]

        # The decorator was initialised with args, it now returns a wrapped function
        elif len(args)==1 and not len(kwargs):
            assert hasattr(args[0], '__call__')
            func = args[0]

            @functools.wraps(func)
            def decorated(*args0, **kwargs0):
                if len(args0)==0 and len(kwargs0)==0:
                    if self.default is None:
                        self.default = self()
                    return self.default()
                else:
                    key1 = self.hash_args(args0)
                    key2 = self.hash_args(kwargs0.values())
                    key = key1 + key2
                    if not key in self.memo:
                        self.memo[key] = func(*args0, **kwargs0)
                    return self.memo[key]

            return decorated

        else:
            raise ValueError()


### Custom Data Structures ###

class CircularBuffer(object):
    def __init__(self, buffer_len):
        self.length = 0
        self._buffer = np.zeros(buffer_len)

    def append(self, data):
        self._buffer = np.roll(self._buffer, 1)
        self._buffer[0] = data
        self.length += 1

    @property
    def data(self):
        return self._buffer[:self.length]

    def mean(self):
        if self.length:
            return self.data.mean()
        else:
            return 0.0

    def setvals(self, val):
        self._buffer[:] = val

    def __len__(self):
        return self.length

    def __getitem__(self, slc):
        return self._buffer[self.length][slc]

    def __repr__(self):
        return repr(self.data)


class AccumulationArray(object):
    def __init__(self, right_shape=(), dtype=np.float32, n_init=100, data=None,
                 ema_factor=0.95):
        if isinstance(dtype, dict) and right_shape!=():
            raise ValueError("If dict is used as dtype, right shape must be"
                             "unchanged (i.e it is 1d)")

        if data is not None and len(data):
            n_init += len(data)
            right_shape = data.shape[1:]
            dtype = data.dtype

        self._n_init = n_init
        if isinstance(right_shape, int):
            self._right_shape = (right_shape,)
        else:
            self._right_shape = tuple(right_shape)
        self.dtype = dtype
        self.length = 0
        self._buffer = self._alloc(n_init)
        self._min = +np.inf
        self._max = -np.inf
        self._sum = 0
        self._ema = None
        self._ema_factor = ema_factor

        if data is not None and len(data):
            self.length = len(data)
            self._buffer[:self.length] = data
            self._min = data.min(0)
            self._max = data.max(0)
            self._sum = data.sum(0)

    def __repr__(self):
        return repr(self.data)

    def _alloc(self, n):
        if isinstance(self._right_shape, (tuple, list, np.ndarray)):
            ret = np.zeros((n,) + tuple(self._right_shape), dtype=self.dtype)
        elif isinstance(self.dtype, dict):  # rec array
            ret = np.zeros(n, dtype=self.dtype)
        else:
            raise ValueError("dtype not understood")
        return ret

    def append(self, data):
        # data = self.normalise_data(data)
        if len(self._buffer)==self.length:
            tmp = self._alloc(len(self._buffer) * 2)
            tmp[:self.length] = self._buffer
            self._buffer = tmp

        if isinstance(self.dtype, dict):
            for k, val in enumerate(data):
                self._buffer[self.length][k] = data[k]
        else:
            self._buffer[self.length] = data
            if self._ema is None:
                self._ema = self._buffer[self.length]
            else:
                f = self._ema_factor
                fc = 1 - f
                self._ema = self._ema * f + self._buffer[self.length] * fc

        self.length += 1

        self._min = np.minimum(data, self._min)
        self._max = np.maximum(data, self._max)
        self._sum = self._sum + np.asanyarray(data)

    def add_offset(self, off):
        self.data[:] += off
        if off.ndim>np.ndim(self._sum):
            off = off[0]
        self._min += off
        self._max += off
        self._sum += off * self.length

    def clear(self):
        self.length = 0
        self._min = +np.inf
        self._max = -np.inf
        self._sum = 0

    def mean(self):
        return np.asarray(self._sum, dtype=np.float32) / self.length

    def sum(self):
        return self._sum

    def max(self):
        return self._max

    def min(self):
        return self._min

    def __len__(self):
        return self.length

    @property
    def data(self):
        return self._buffer[:self.length]

    @property
    def ema(self):
        return self._ema

    def __getitem__(self, slc):
        return self._buffer[:self.length][slc]


class KDT(NearestNeighbors_):
    warning_shown = False

    @functools.wraps(NearestNeighbors_.__init__)
    def __init__(self, n_neighbors=5, radius=1.0, algorithm='auto',
                 leaf_size=30, metric='minkowski', p=2, metric_params=None,
                 n_jobs=1, **kwargs):
        if sklearn.__version__=="0.16.1":
            if not KDT.warning_shown:
                logger.warning("sklearn version does not support MP, try to upgrade it.")
                KDT.warning_shown = True

            if "n_jobs" in kwargs:
                kwargs.pop("n_jobs")
        else:
            kwargs['n_jobs'] = n_jobs

        super(KDT, self).__init__(n_neighbors=n_neighbors, radius=radius,
                                  algorithm=algorithm, leaf_size=leaf_size,
                                  metric=metric, p=p,
                                  metric_params=metric_params, **kwargs)

    __init__.__doc__ = NearestNeighbors_.__init__.__doc__


@numba.jit(nopython=True, looplift=True, cache=True)
def _merge(distances, indices, coordinates, pairwise_dist, sort_ix, new_points,
           k, query_points):
    q = len(query_points)
    dim = query_points.shape[1:]
    distances_new = np.zeros((q, k), dtype=np.float32)  # (q,k)
    indices_new = np.zeros((q, k), dtype=np.int64)  # (q,k)
    coordinates_new = np.zeros((q, k,) + dim, dtype=np.float32)  # (q,k,2/3)
    kdt_pointer = np.zeros(q, dtype=np.int64)  # (q) should be maximal k+1
    new_pointer = np.zeros(q, dtype=np.int64)  # (q) should be maximal m+1

    for p in range(q):  # over query points
        for c in range(k):  # over #NNs
            new_ix = sort_ix[p, new_pointer[p]]
            d_new = pairwise_dist[p, new_ix]
            d_kdt = distances[p, kdt_pointer[p]]
            if d_kdt>d_new:
                distances_new[p, c] = d_new
                indices_new[p, c] = -666
                coordinates_new[p, c] = new_points[new_ix]
                new_pointer[p] += 1
            else:
                distances_new[p, c] = d_kdt
                indices_new[p, c] = indices[p, kdt_pointer[p]]
                coordinates_new[p, c] = coordinates[p, kdt_pointer[p]]
                kdt_pointer[p] += 1

    return distances_new, indices_new, coordinates_new


class DynamicKDT(object):
    def __init__(self, points=None, k=1, n_jobs=-1, rebuild_thresh=100,
                 aniso_scale=None):
        self._kdt = None
        self._new_points = []
        self._static_points = []
        self._k = k
        self._jobs = n_jobs
        self._rebuild_thresh = rebuild_thresh
        self.aniso_scale = 1
        if aniso_scale is not None:
            if isinstance(aniso_scale, (list, tuple)):
                self.aniso_scale = np.atleast_2d(np.array(aniso_scale))
            elif isinstance(aniso_scale, np.ndarray):
                self.aniso_scale = np.atleast_2d(aniso_scale)
            else:
                raise ValueError("aniso_scale not understood")

        if points is not None:
            if len(points)<=k:
                raise ValueError("points must be longer than k")
            self._kdt = KDT(n_neighbors=k, n_jobs=n_jobs, algorithm='kd_tree',
                            leaf_size=20)

            self._kdt.fit(points * self.aniso_scale)
            self._static_points = points

    def append(self, point):
        point = np.asarray(point)
        if self._new_points==[]:
            self._new_points = AccumulationArray(right_shape=point.shape,
                                                 n_init=self._rebuild_thresh)

        if len(self._new_points)==self._rebuild_thresh:
            if self._static_points==[]:
                self._static_points = self._new_points.data.copy()
            else:
                self._static_points = np.concatenate(
                    [self._static_points, self._new_points.data], axis=0)
            self._new_points.clear()
            self._kdt = KDT(n_neighbors=self._k, n_jobs=self._jobs,
                            algorithm='kd_tree', leaf_size=20)
            self._kdt.fit(self._static_points * self.aniso_scale)

        self._new_points.append(point)

    def get_knn(self, query_points, k=None):
        if k is None:
            k = self._k

        if k>(len(self._new_points) + len(self._static_points)):
            raise ValueError("The requested number of neighbours is larger "
                             "than the number of stored points")
        if query_points.ndim==1:
            query_points = query_points[None]
        q = len(query_points)

        if len(self._static_points):
            # assert k==self._kdt.n_neighbors
            distances, indices = self._kdt.kneighbors(
                query_points * self.aniso_scale, n_neighbors=k)
            # Add inf for stopping
            distances = np.hstack([distances, np.ones((q, 1)) * np.inf])
        else:
            distances = np.ones((q, 1)) * np.inf
            indices = np.zeros((q, 1), dtype=np.int)

        if len(self._new_points):
            new_points = self._new_points.data
            pairwise_dist = cdist(query_points * self.aniso_scale,
                                  new_points * self.aniso_scale, p=2)
            # Add inf for stopping
            pairwise_dist = np.hstack(
                [pairwise_dist, np.ones((q, 1)) * np.inf])
        else:
            new_points = np.zeros((0, 1), dtype=query_points.dtype)
            pairwise_dist = np.ones((q, 1)) * np.inf  # (q,1)

        if k==1:
            indices = indices[:, 0]
            distances = distances[:, 0].astype(np.float32)
            if len(self._static_points):
                coordinates = self._static_points[indices]
            else:
                coordinates = new_points[indices]
            # Override found neighbours if a closer neighbour is in new_points
            replace_by_new = pairwise_dist.min(axis=1)<distances
            distances[replace_by_new] = pairwise_dist[replace_by_new]
            new_index = pairwise_dist.argmin(axis=1)
            if np.any(replace_by_new):
                coordinates[replace_by_new] = new_points[
                    new_index[replace_by_new]]
                indices[replace_by_new] = new_index[replace_by_new] + len(
                    self._static_points)  # -666 # This is just a dummy, atm indices is not used anyway
                distances[replace_by_new] = pairwise_dist[
                    replace_by_new, new_index[replace_by_new]]
        else:
            if len(self._static_points):
                coordinates = self._static_points[indices]
            else:
                coordinates = np.zeros_like(new_points)

            sort_ix = pairwise_dist.argsort(axis=1)  # (q,n) this is ascending
            distances, indices, coordinates = _merge(distances, indices,
                                                     coordinates,
                                                     pairwise_dist, sort_ix,
                                                     new_points, k,
                                                     query_points)

        if q==1:
            distances = distances[0]
            indices = indices[0]
            coordinates = coordinates[0]

        return distances, indices, coordinates

    def get_radius_nn(self, query_points, radius):
        raise NotImplementedError()


### Various Simple Utils ###

def import_variable_from_file(file_path, class_name):
    directory = os.path.dirname(file_path)
    sys.path.append(directory)
    mod_name = os.path.split(file_path)[1]
    if mod_name[-3:]=='.py':
        mod_name = mod_name[:-3]

    mod = importlib.import_module(mod_name)
    sys.path.pop(-1)
    cls = getattr(mod, class_name)
    return cls


def picklesave(data, file_name):
    """
    Writes one or many objects to pickle file

    data:
      single objects to save or iterable of objects to save.
      For iterable, all objects are written in this order to the file.
    file_name: string
      path/name of destination file
    """
    file_name = os.path.expanduser(file_name)
    with open(file_name, 'wb') as f:
        pkl.dump(data, f, protocol=2)


def pickleload(file_name):
    """
    Loads all object that are saved in the pickle file.
    Multiple objects are returned as list.
    """
    file_name = os.path.expanduser(file_name)
    ret = []
    try:
        with open(file_name, 'rb') as f:
            try:
                while True:
                    # Python 3 needs explicit encoding specification,
                    # which Python 2 lacks:
                    if sys.version_info.major>=3:
                        ret.append(pkl.load(f, encoding='latin1'))
                    else:
                        ret.append(pkl.load(f))
            except EOFError:
                pass

        if len(ret)==1:
            return ret[0]
        else:
            return ret

    except pkl.UnpicklingError:
        with gzip.open(file_name, 'rb') as f:
            try:
                while True:
                    # Python 3 needs explicit encoding specification,
                    # which Python 2 lacks:
                    if sys.version_info.major>=3:
                        ret.append(pkl.load(f, encoding='latin1'))
                    else:
                        ret.append(pkl.load(f))
            except EOFError:
                pass

        if len(ret)==1:
            return ret[0]
        else:
            return ret


def h5save(data, file_name, keys=None, compress=True):
    """
    Writes one or many arrays to h5 file

    data:
      single array to save or iterable of arrays to save.
      For iterable all arrays are written to the file.
    file_name: string
      path/name of destination file
    keys: string / list thereof
      For single arrays this is a single string which is used as a name
      for the data set.
      For multiple arrays each dataset is named by the corresponding key.
      If keys is ``None``, the dataset names created by enumeration: ``data%i``
    compress: Bool
      Whether to use lzf compression, defaults to ``True``. Most useful for
      label arrays.
    """
    file_name = os.path.expanduser(file_name)
    compr = 'lzf' if compress else None
    f = h5py.File(file_name, "w")
    if isinstance(data, list) or isinstance(data, tuple):
        if keys is not None:
            assert len(keys)==len(data)
        for i, d in enumerate(data):
            if keys is None:
                f.create_dataset(str(i), data=d, compression=compr)
            else:
                f.create_dataset(keys[i], data=d, compression=compr)
    else:
        if keys is None:
            f.create_dataset('data', data=data, compression=compr)
        else:
            f.create_dataset(keys, data=data, compression=compr)
    f.close()


def h5load(file_name, keys=None):
    """
    Loads data sets from h5 file

    file_name: string
      destination file
    keys: string / list thereof
      Load only data sets specified in keys and return as list in the order
      of ``keys``
      For a single key the data is returned directly - not as list
      If keys is ``None`` all datasets that are listed in the keys-attribute
      of the h5 file are loaded.
    """
    file_name = os.path.expanduser(file_name)
    ret = []
    try:
        f = h5py.File(file_name, "r")
    except IOError:
        raise IOError("Could not open h5-File %s" % (file_name))

    if keys is not None:
        try:
            if isinstance(keys, str):
                ret.append(f[keys].value)
            else:
                for k in keys:
                    ret.append(f[k].value)
        except KeyError:
            raise KeyError("Could not read h5-dataset named %s. Available "
                           "datasets: %s" % (keys, list(f.keys())))
    else:
        for k in f.keys():
            ret.append(f[k].value)

    f.close()

    if len(ret)==1:
        return ret[0]
    else:
        return ret


def pretty_string_ops(n):
    """
    Return a humanized string representation of a large number.
    """
    abbrevs = [(1000000000000, 'Tera Ops'),
               (1000000000, 'Giga Ops'),
               (1000000, 'Mega Ops'),
               (1000, 'kilo Ops')]
    for factor, suffix in abbrevs:
        if n>=factor:
            break
    return "%.1f %s" % (float(n) / factor, suffix)


def makeversiondir(path, dir_name=None, cd=False):
    path = os.path.expanduser(path)
    if dir_name:
        path = os.path.join(path, dir_name)
    while True:
        if os.path.exists(path):
            try:
                num = re.findall(r"-v(\d+)$", path)[0]
                num = int(num)
                i = 2 + np.int(np.log10(num) + 1)
                num = "-v" + str(int(num) + 1)
                path = path[:-i] + num
            except:
                path = path + "-v1"
        else:
            break

    os.makedirs(path, mode=0o755)
    if cd:
        os.chdir(path)
    return path


class Timer(object):
    def __init__(self, silent_all=False):
        self.last_t = time.time()
        self.total = 0
        self.checktimes = []
        self.checknames = []
        self.accumulator = {}
        self.silent_all = silent_all

    def check(self, name=None, silent=False):
        t = time.time()
        dt = t - self.last_t
        self.total += dt
        self.last_t = t
        name = name if name is not None else ""
        if not silent and not self.silent_all:
            print("%s\tdt=%.3g s,\tt=%.3g s" % (name, dt, self.total))

        self.checknames.append(name)
        self.checktimes.append(dt)
        if name is not None:
            accum = self.accumulator.get(name, 0)
            self.accumulator[name] = accum + dt

    def plot(self, accum=False):
        # I don't want this import every time utils is used
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        if accum:
            times = list(self.accumulator.values())
            names = list(self.accumulator.keys())
        else:
            times = self.checktimes
            names = self.checknames

        ind = np.arange(len(times))
        ax.bar(ind, times)
        ax.set_xticks(ind + 0.5)
        ax.set_xticklabels(names)
        plt.show()

    def summary(self, silent=False, print_func=None):
        s = "Total t: %.3gs\n" % self.total
        if len(self.accumulator):
            ix = np.argsort(self.accumulator.values())
            for i in ix:
                name, dt = self.accumulator.items()[i]
                s += "%s:\t\t%.3gs\t%.3g%%\n"\
                     %(name, dt, dt/self.total*100.0)
        else:
            ix = np.argsort(self.checktimes)
            for i in ix:
                name, dt = self.checknames[i], self.checktimes[i]
                s += "%s:\t\t%.3gs\t%.3g%%\n"\
                     % (name, dt. dt/self.total*100.0)
        if silent:
            return s
        else:
            if print_func:
                print_func(s)
            else:
                print(s)


def pretty_string_time(t):
    """Custom printing of elapsed time"""
    if t>4000:
        s = 't=%.1fh' % (t / 3600)
    elif t>300:
        s = 't=%.0fm' % (t / 60)
    else:
        s = 't=%.0fs' % (t)
    return s


def unique_rows(a):
    # removes duplicates from a
    a = np.ascontiguousarray(a)
    unique_a, index = np.unique(a.view([('', a.dtype)] * a.shape[1]),
                                return_index=True)
    ret = unique_a.view(a.dtype).reshape(
        (unique_a.shape[0], a.shape[1])), index
    print("Removed %i of %i (new %i)" % (
        len(a) - len(index), len(a), len(index)))
    return ret


def as_list(var):
    if var is None:
        return var
    elif isinstance(var, (list, tuple)):
        return list(var)
    else:
        return [var, ]
