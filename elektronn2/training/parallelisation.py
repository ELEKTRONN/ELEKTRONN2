# -*- coding: utf-8 -*-
# ELEKTRONN2 Toolkit
# Copyright (c) 2015 Marius Killinger
# All rights reserved

from __future__ import absolute_import, division, print_function
from builtins import filter, hex, input, int, map, next, oct, pow, range, \
    super, zip


import multiprocessing as mp
import ctypes
import logging
import time
from collections import deque
import numpy as np


__all__ = ['BackgroundProc', 'SharedQ', 'TimeoutError']

logger = logging.getLogger('elektronn2log')


class TimeoutError(RuntimeError):
    def __init__(self, *args, **kwargs):
        super(TimeoutError, self).__init__(*args, **kwargs)


class SharedMem(object):
    """Utilities to share np.arrays between processes"""
    _ctypes_to_numpy = {ctypes.c_int8: np.dtype(np.int8),
                        ctypes.c_int16: np.dtype(np.int16),
                        ctypes.c_uint16: np.dtype(np.uint16),
                        ctypes.c_int32: np.dtype(np.int32),
                        ctypes.c_uint32: np.dtype(np.uint32),
                        ctypes.c_int64: np.dtype(np.int64),
                        ctypes.c_uint64: np.dtype(np.uint64),

                        ctypes.c_byte: np.dtype(np.int8),
                        ctypes.c_ubyte: np.dtype(np.uint8),
                        ctypes.c_short: np.dtype(np.int16),
                        ctypes.c_ushort: np.dtype(np.uint16),
                        ctypes.c_int: np.dtype(np.int32),
                        ctypes.c_uint: np.dtype(np.uint32),
                        ctypes.c_long: np.dtype(np.int32),
                        ctypes.c_ulong: np.dtype(np.uint32),
                        ctypes.c_longlong: np.dtype(np.int64),
                        ctypes.c_ulonglong: np.dtype(np.int64),

                        ctypes.c_float: np.dtype(np.float32),
                        ctypes.c_double: np.dtype(np.float64)}

    _numpy_to_ctypes = dict(
        list(zip(_ctypes_to_numpy.values(), _ctypes_to_numpy.keys())))

    @staticmethod
    def shm2ndarray(mp_array, shape=None):
        """
        Parameters
        ----------

        mp_array: a mp.Array
        shape: (optional) the returned np.ndarray
                is reshaped to this shape, flat otherwise

        Returns
        -------

        array: np.ndarray
         That can be normally used but changes are reflected in shared mem

        Note: the returned array is still pointing to the sharedmem,
              data might be changed by another process!
        """
        dtype = SharedMem._ctypes_to_numpy[mp_array._type_]
        result = np.frombuffer(mp_array, dtype)

        if shape is not None:
            result = result.reshape(shape)

        return np.asarray(result)

    @staticmethod
    def ndarray2shm(np_array, lock=False):
        """
        Parameters
        ----------

        np_array: np.ndarray
          array of arbitrary shape
        lock: Bool
          Whether to create a multiprocessing.Lock

        Returns
        -------

        handle: mp.Array:
          flat with data from ndarray copied to it
        """
        array1d = np_array.ravel(order='A')

        try:
            c_type = SharedMem._numpy_to_ctypes[array1d.dtype]
        except KeyError:
            c_type = SharedMem._numpy_to_ctypes[np.dtype(array1d.dtype)]

        result = mp.Array(c_type, array1d.size, lock=lock)
        SharedMem.shm2ndarray(result)[:] = array1d
        return result

    def puthandle(self, dtype, shape, data=None, lock=False):
        """
        Creates new shared memory and puts it on the queue.
        Other sub-processes can write to it.

        Parameters
        ----------
        dtype: np.dtype
          Type of data to store in array
        shape: tuple
          Properties of shared mem to be created
        data: np.ndarray
         (optional) values to fill shared array with
        lock: Bool
          Whether to create a multiprocessing.Lock on the shared variable

        Returns
        -------
        sharedmem handle: mp.array
        """
        t0 = time.clock()
        size = int(np.prod(shape))
        try:
            c_type = SharedMem._numpy_to_ctypes[dtype]
        except KeyError:
            c_type = SharedMem._numpy_to_ctypes[np.dtype(dtype)]

        shm = mp.Array(c_type, size, lock=lock)
        t1 = time.clock()

        if data is not None:
            SharedMem.shm2ndarray(shm)[:] = data.astype(dtype).ravel(order='A')
        t2 = time.clock()

        if self.profile:
            t_alloc = t1 - t0
            t_write = t2 - t1
            self.logger.info('SharedMemAlloc %g ms, WriteInitialData %g ms' % (
                t_alloc * 1000, t_write * 1000))

        return shm


class Proc(mp.Process):
    """
    A *reusable* and *configurable* background process, that does the
    same job every time ``events['new']`` is set and signals that is has
    finished one iteration by setting ``events['ready']``
    """

    def __init__(self, mp_arrays, shapes, events, target, target_args,
                 target_kwargs, profile):
        super(Proc, self).__init__()
        self.events = events
        self.target = target
        self.target_args = target_args
        self.target_kwargs = target_kwargs
        self.arrays = []  # shm "wrapped" as np.array-objs
        self.profile = profile

        if profile:
            self.logger = mp.log_to_stderr(logging.INFO)

        for shm, shp in zip(mp_arrays, shapes):
            self.arrays.append(SharedMem.shm2ndarray(shm, shp))

    def run(self):
        while True:
            try:
                # wait till host has fetched data from shm and demands new data from this proc
                self.events['new'].wait()
                self.events['new'].clear()
                t0 = time.clock()
                result = self.target(*self.target_args, **self.target_kwargs)
                if isinstance(result, np.ndarray):
                    result = [result, ]
                for a, r in zip(self.arrays, result):
                    a[:] = r
                # signal host that task is done and data is ready in shm
                self.events['ready'].set()
                t1 = time.clock()
                if self.profile:
                    t_exec = t1 - t0
                    self.logger.info(
                        'Executing Target and writing to shm %g ms' % (
                            t_exec * 1000))
            except KeyboardInterrupt:
                pass


class BackgroundProc(SharedMem):
    """
    Data structure to manage repeated background tasks by reusing a fixed
    number of *initially* created background process with the same arguments
    at every time. (E.g. retrieving an augmented batch) Remember to call
    ``BackgroundProc.shutdown`` after use to avoid zombie process and RAM clutter.

    Parameters
    ----------

    dtypes:
      list of dtypes of the target return values
    shapes:
      list of shapes of the target return values
    n_proc: int
      number of background procs to use
    target: callable
      target function for background proc. Can even be a method of an object,
      if object data is read-only (then data will not be copied in RAM and the
      new process is lean). If several procs use random modules, new seeds
      must be created inside target because they have the same random state
      at the beginning.
    target_args:  tuple
      Proc args (constant)
    target_kwargs: dict
      Proc kwargs (constant)
    profile: Bool
      Whether to print timing results in to stdout

    Examples
    --------

    Use case to retrieve batches from a data structure ``D``:

    >>> data, label = D.getbatch(2, strided=False, flip=True, grey_augment_channels=[0])
    >>> kwargs = {'strided': False, 'flip': True, 'grey_augment_channels': [0]}
    >>> bg = BackgroundProc([np.float32, np.int16], [data.shape,label.shape], \
                            D.getbatch, n_proc=2, target_args=(2,), \
                            target_kwargs=kwargs, profile=False)
    >>> for i in range(100):
    >>>    data, label = bg.get()

    """

    def __init__(self, target, dtypes=None, shapes=None, n_proc=1,
                 target_args=(), target_kwargs={}, profile=False):
        self.dtypes = dtypes
        self.shapes = shapes
        self.target = target
        self.n_proc = n_proc
        self.i = 0  # index of next item to consume
        self.mp_arrays = []
        self.procs = []
        self.events = []
        self.profile = profile

        if (dtypes is None) or (shapes is None):
            ret = target(*target_args, **target_kwargs)
            if isinstance(ret, np.ndarray):
                ret = [ret, ]
            dtypes = [b.dtype for b in ret if b is not None]
            shapes = [b.shape for b in ret if b is not None]
            self.dtypes = dtypes
            self.shapes = shapes

        if profile:
            self.logger = mp.log_to_stderr(logging.INFO)

        # create a list of mp-arrays for each process
        for k in range(n_proc):
            a = []
            for dtype, shape in zip(dtypes, shapes):
                a.append(self.puthandle(dtype, shape))

            self.mp_arrays.append(a)
            self.events.append({'new': mp.Event(), 'ready': mp.Event()})

        # initialise the procs and give them their mp-arrays
        for shm, e in zip(self.mp_arrays, self.events):
            p = Proc(shm, shapes, e, target, target_args, target_kwargs, profile)
            p.start()
            e['new'].set()
            self.procs.append(p)

    def get(self, timeout=False):
        """
        This gets the next result from a background process and blocks
        until the corresponding proc has finished.
        """
        t0 = time.time()
        while True:  # go trough queue until a ready process is found
            k = self.i
            self.i = (self.i + 1) % self.n_proc  # advance index of next item
            if self.events[k]['ready'].is_set():
                break
            time.sleep(0.0001)
            t = time.time() - t0
            if timeout and t > timeout:
                raise TimeoutError

        result = []
        t0 = time.clock()
        self.events[k]['ready'].wait()
        self.events[k]['ready'].clear()
        t1 = time.clock()
        for shm, shp in zip(self.mp_arrays[k], self.shapes):
            # copy! Otherwise a proc will write to result
            result.append(SharedMem.shm2ndarray(shm, shp).copy())

        self.events[k]['new'].set()
        t2 = time.clock()
        if self.profile:
            t_wait = t1 - t0
            t_write = t2 - t1
            self.logger.info(
                'Waiting for subprocess %g ms, converting to numpy %g ms' % (
                    t_wait * 1000, t_write * 1000))

        return tuple(result)

    def shutdown(self):
        """**Must be called to free memory** if the background tasks are no longer needed"""
        for p in self.procs:
            p.terminate()

    def reset(self):
        """
        Should be called after an exception (e.g. by pressing ctrl+c) was raised.
        """
        for e in self.events:
            e['new'].set()

    def __del__(self):
        self.shutdown()


class SharedQ(SharedMem):
    """
    FIFO Queue to process np.ndarrays in the background
    (also pre-loading of data from disk)

    procs must accept list of ``mp.Array`` and make items ``np.ndarray``
    using ``SharedQ.shm2ndarray``, for this the shapes are required as too.
    The target requires the signature::

       >>> target(mp_arrays, shapes, *args, **kwargs)

    Whereas mp_array and shape are *automatically* added internally

    All parameters are optional:

    Parameters
    ----------
    n_proc: int
      If larger than 0, a message is printed if to few processes are running
    profile: Bool
      Whether to print timing results in terminal

    Examples
    --------
    Automatic use:

    >>> Q = SharedQ(n_proc=2)
    >>> Q.startproc(target=, shape= args=, kwargs=)
    >>> Q.startproc(target=, shape= args=, kwargs=)
    >>> for i in range(5):
    >>>     Q.startproc(target=, shape= args=, kwargs=)
    >>>     item = Q.get() # starts as many new jobs as to maintain n_proc
    >>>     dosomethingelse(item) # processes work in background to pre-fetch data for next iteration
    """

    def __init__(self, n_proc=0, profile=False):

        self.data = deque()  # items of type [shm, shape, proc]
        self.len = 0
        self.n_proc = n_proc
        self.profile = profile

    def startproc(self, dtypes, shapes, target, target_args=(),
                  target_kwargs={}):
        """
        Starts a new process

        procs must accept list of ``mp.Array`` and make items ``np.ndarray``
        using ``SharedQ.shm2ndarray``, or this the shapes are required as
        too. The target requires the signature::

           target(mp_arrays, shapes, *args, **kwargs)

        Whereas mp_array  and shape are *automatically* added internally
        """

        data = target_kwargs.get('data')

        mp_arrays = []
        for dtype, shape in zip(dtypes, shapes):
            mp_arrays.append(self.puthandle(dtype, shape, data))

        t0 = time.clock()
        _args = (mp_arrays, shapes) + target_args
        proc = mp.Process(target=target, args=_args, kwargs=target_kwargs)
        proc.daemon = True
        proc.start()

        self.data.append([mp_arrays, shapes, proc])
        self.len += 1
        t1 = time.clock()
        if self.profile:
            t_start = t1 - t0
            self.logger.info('Start Process %g ms' % (t_start * 1000))

    def get(self):
        """
        This gets the first results in the queue and blocks until the
        corresponding proc has finished. If a n_proc value is defined this
        then new procs must be started *before* to avoid a warning message.
        """
        mp_arrays, shapes, proc = self.data.popleft()
        self.len -= 1
        missing = self.n_proc - self.len
        if missing > 0:
            self.logger.warning(
                "You should have started %i new workes before Q.get()" % missing)

        t0 = time.clock()
        proc.join()
        t1 = time.clock()
        result = []
        for shm, shp in zip(mp_arrays, shapes):
            result.append(SharedMem.shm2ndarray(shm, shp))

        t2 = time.clock()

        if self.profile:
            t_join = t1 - t0
            t_conv = t2 - t1
            self.logger.info('Join %g ms, Shared2Numpy %g ms' % (
                t_join * 1000, t_conv * 1000))

        return result


### Testing etc. ##############################################################
# Prerequisites
if __name__=="__main__":
    import gc
    import h5py


    def load():
        f = h5py.File('~/devel/data/MPI/raw_center_cube_mag1_v3.h5', 'r')
        d = f['raw'].value
        f.close()
        return d[0]


    t0 = time.time()
    D = load()
    t1 = time.time()
    lt = t1 - t0
    logger.info('REAL LOAD TIME %.2f  sec' % lt)


    def CPU():
        a = np.random.rand(1160 * 480)
        for i in range(50):
            np.sin(a)


    t0 = time.time()
    for i in range(3):
        CPU()

    t1 = time.time()
    rt = (t1 - t0) / 3
    logger.info('REAL CPU TASK TIME %.2f  sec' % rt)
    serial = 20 * rt + 20 * lt
    D = None
    gc.collect()


    def IO(mp_array, shape):
        t0 = time.time()
        d = load()
        SharedQ.shm2ndarray(mp_array, shape)[:] = d
        t = time.time() - t0
        logger.info('LOADED data in %.2f  sec' % t)

