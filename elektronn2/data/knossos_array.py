# -*- coding: utf-8 -*-
# ELEKTRONN2 Toolkit
# Copyright (c) 2015 Marius Killinger
# All rights reserved

# TODO: Python 3 compatibility


__all__ = ['KnossosArray', 'KnossosArrayMulti']

import glob
import itertools
import re
import os
import sys
import traceback
import ctypes
import multiprocessing as mp
from collections import deque

import numpy as np

from .. import utils

if sys.version_info[:2] != (2, 7):
    raise ImportError(
        '\nSorry, this module only supports Python 2.7.'
        '\nYour current Python version is {}\n'.format(sys.version)
    )


def our_glob(s):
    l = []
    for g in glob.glob(s):
        l.append(g.replace(os.path.sep,"/"))
    return l

def get_remaining_size(fd):
    try:
        fn = fd.fileno()
    except AttributeError:
        return os.path.getsize(fd.name) - fd.tell()
    st = os.fstat(fn)
    size = st.st_size - fd.tell()
    return size

def load_rawfile(path, out, offset=0):
    fd = open(path, 'rb')
    if (offset > 0):
        fd.seek(offset, 1)
    size = get_remaining_size(fd)
    itemsize = out.itemsize
    shapeprod = out.size
    nbytes = shapeprod * itemsize
    if nbytes > size:
        raise ValueError(
                "Not enough bytes left in file for specified shape and type")

    nbytesread = fd.readinto(out.data)
    if nbytesread != nbytes:
        raise IOError("Didn't read as many bytes as expected")

    fd.close()
    return out

class LoadProc(mp.Process):
    def __init__(self, mp_array, path, dtype, status):
        super(LoadProc, self).__init__()
        self.path = path
        self.mp_array = mp_array
        self.status = status
        self.array = np.frombuffer(mp_array.get_obj(), dtype)

    def run(self):
        try:
            self.mp_array.acquire()
            a_buffer = np.empty(len(self.array), np.uint8)
            load_rawfile(self.path, a_buffer)
            self.array[:] = a_buffer
            self.array /= 255
            self.mp_array.release()
        except:
            self.mp_array.release()
            traceback.print_exc()
            self.status.set()


class KnossosArray(object):
    """
    Interfaces with knossos cubes, all  axes are in zxy order!
    """
    def __init__(self, path, max_ram=1000, n_preload=2, fixed_mag=1):
        path = os.path.expanduser(path)
        # Knossos attributes
        self._shape = np.zeros(3, dtype=np.int)
        self._shape_spatial = np.zeros(3, dtype=np.int)
        self.scale = np.zeros(3, dtype=np.float)
        self._knossos_path = None
        self._experiment_name = None
        self._cube_boundary = None
        self._cube_length = 128
        self._cube_sh = [self._cube_length,]*3
        self._mag = []

        self._n_f = 1

        # slicing/loading attributes
        self.dtype = np.dtype(np.float32)
        _n_preload = np.ones(3, dtype=np.float)*n_preload
        _n_preload[0] /= 2
        self.n_preload = np.ceil(_n_preload).astype(np.int)

        max_bytes = max_ram * 2**20
        self.cache_size = int(max_bytes / (self._cube_length**3 * self.dtype.itemsize))
        if self.cache_size<np.prod(self.n_preload*2+1):
            ram = np.prod(self.n_preload*2+1) * (self._cube_length**3 * self.dtype.itemsize) // 2**20
            raise ValueError("For %s preload cubes more RAM is required: %i MB" %(n_preload, ram))

        self.mpa_used = {}
        self.mpa_empty = []
        self.cubes = {}
        self.load_q = deque()
        self.loading = {}

        for i in range(self.cache_size):
            mp_array = mp.Array(ctypes.c_float, int(self._cube_length**3), lock=True)
            mp_array.acquire()
            self.mpa_empty.append(mp_array)
        self._initialize_from_knossos_path(path, fixed_mag)
        # self.track = []

    @property
    def shape(self):
        return self._shape

    @property
    def n_f(self):
        return self._n_f

    def __repr__(self):
        s = "<KnossosArray> %s\n"%(self.shape,)
        s += "%i cubes in cache\n" %len(self.cubes)
        s += "%i mpa in use\n" %len(self.mpa_used)
        s += "%i mpa empty\n" %len(self.mpa_empty)
        s += "%i mpa loading\n" %len(self.loading)
        s += "%i loading q\n" %len(self.load_q)
        return s

    def _initialize_from_knossos_path(self, path, fixed_mag=0):
        """ Initialises the dataset by parsing the knossos.conf in path + "mag1"

        :param path: str
            forward-slash separated path to the dataset folder - not .../mag !
        :param fixed_mag: int
            fixes available mag to one specific value

        :return:
            nothing
        """
        self._knossos_path = path
        all_mag_folders = our_glob(os.path.join(path,"*mag*"))

        if fixed_mag > 0:
            self._mag.append(fixed_mag)
        else:
            for mag_test_nb in range(32):
                for mag_folder in all_mag_folders:
                    if "mag"+str(2**mag_test_nb) in mag_folder:
                        self._mag.append(2**mag_test_nb)
                        break

        mag_folder = our_glob(os.path.join(path,"*mag*"))[0].split("/")
        if len(mag_folder[-1]) > 1:
            mag_folder = mag_folder[-1]
        else:
            mag_folder = mag_folder[-2]

        self._name_mag_folder = \
            mag_folder[:-len(re.findall("[\d]+", mag_folder)[-1])]

        try:
            p = our_glob(os.path.join(path,"*mag1"))[0]+"/knossos.conf"
            print("Reading %s" %p)
            f = open(p)

            lines = f.readlines()
            f.close()
        except:
            raise NotImplementedError("Could not find/read *mag1/knossos.conf")

        parsed_dict = {}
        for line in lines:
            try:
                match = re.search(r'(?P<key>[A-Za-z _]+)'
                                  r'((?P<numeric_value>[0-9\.]+)'
                                  r'|"(?P<string_value>[A-Za-z0-9._/-]+)");',
                                  line).groupdict()

                if match['string_value']:
                    val = match['string_value']
                elif '.' in match['numeric_value']:
                    val = float(match['numeric_value'])
                elif match['numeric_value']:
                    val = int(match['numeric_value'])
                else:
                    raise Exception('Malformed knossos.conf')

                parsed_dict[match["key"]] = val
            except:
                print("Unreadable line in knossos.conf - ignored.")

        self._shape[2] = parsed_dict['boundary x ']
        self._shape[1] = parsed_dict['boundary y ']
        self._shape[0] = parsed_dict['boundary z ']
        self._shape_spatial[:] = self._shape
        self._cube_boundary = np.ceil(self._shape.astype(np.float) / self._cube_length).astype(np.int) # the boundary is the first for which no cube is available!
        self.scale[2] = parsed_dict['scale x ']
        self.scale[1] = parsed_dict['scale y ']
        self.scale[0] = parsed_dict['scale z ']

        self._experiment_name = parsed_dict['experiment name ']
        if self._experiment_name.endswith("mag1"):
            self._experiment_name = self._experiment_name[:-5]

    def _request_cube(self, coords):
        assert (coords not in self.cubes) and (coords not in self.loading)
        assert len(self.mpa_empty)
        # self.track.append(coords)
        coords_xyz = [coords[2], coords[1], coords[0]] # knossos has xyz order, we have zyx
        mag = self._mag[0]
        path = self._knossos_path+self._name_mag_folder+\
               str(mag) + "/x%04d/y%04d/z%04d/" \
               % (coords_xyz[0], coords_xyz[1], coords_xyz[2]) + \
               self._experiment_name + '_mag' + str(mag) + \
               "_x%04d_y%04d_z%04d.raw" \
               % (coords_xyz[0], coords_xyz[1], coords_xyz[2])
        if not os.path.exists(path):
            raise ValueError("Slice out of bounce, no cube %s "
            "[Note: not all cubes exist within bounding box!].)" %(coords,))

        mp_array = self.mpa_empty.pop()
        status = mp.Event() # is not set
        mp_array.release()
        proc = LoadProc(mp_array, path, self.dtype, status)
        proc.start()
        self.load_q.append([status, proc, mp_array, coords])
        self.loading[coords]=True

    def _clear_q(self, wait_for):
        if wait_for is None:
            states = [q[1].is_alive() for q in self.load_q]
            try:
                max_i = states.index(True)
            except: # all have terminated already
                max_i = len(self.load_q)

        elif wait_for=='all':
            max_i = len(self.load_q)
        else:
            max_i = len(self.load_q)

        i = 0
        while i<max_i:
            status, proc, mp_array, coords = self.load_q.popleft()
            if proc.is_alive():
                print("WAITING for cube %s" %(coords,))
            proc.join()

            if status.is_set():
                raise Exception("An error happened in one of the loading processes")

            mp_array.acquire()
            np_array = np.frombuffer(mp_array.get_obj(), self.dtype).reshape(self._cube_sh)
            assert coords not in self.cubes, "%s"%(coords,)
            self.cubes[coords] = np_array
            self.mpa_used[coords] = mp_array
            self.loading.pop(coords)
            #print("cube %s loaded" %(coords,))
            if coords==wait_for:
                break
            i += 1

    def preload(self, position, start_end=None, sync=False):
        """
        preloads around position preload distance but at least to cover start-end
        """
        position = np.array(position, dtype=np.int)
        center_cube = np.floor((position)/self._cube_length).astype(np.int)
        start = np.maximum(center_cube - self.n_preload, 0)
        end   = np.minimum(center_cube + self.n_preload + 1, self._cube_boundary)
        # preload indices
        ijk_pre = list(itertools.product(range(start[0], end[0]),
                                         range(start[1], end[1]),
                                         range(start[2], end[2])))

        # requested indices:
        if start_end is not None:
            start_r, end_r = start_end
            if np.any(end_r > self._cube_boundary):
                print("Warning: Slice at %s might be out of bounce (no cube %s)" %(position, end_r-1))

            ijk_req = list(itertools.product(range(start_r[0], end_r[0]),
                                             range(start_r[1], end_r[1]),
                                             range(start_r[2], end_r[2])))
            ijk_req_set = set((ijk_req))
            ijk_pre = list(filter(lambda x: x not in ijk_req_set, ijk_pre))
            ijk = ijk_req + ijk_pre
        else:
            ijk = ijk_pre

        fetch = []
        keep  = {}
        for coords in ijk:
            if (coords in self.cubes):
                keep[coords] = True
            elif (coords not in self.loading):
                fetch.append(coords)

        if len(fetch) > len(self.mpa_empty): # need to release arrays first
            n = len(fetch) - len(self.mpa_empty)
            if len(ijk) > self.cache_size:
                f = float(len(ijk)) / self.cache_size
                raise ValueError("The requested array+preload margin is %.2f x"
                " too large for the cache limit" %f)
            self._release_cubes(center_cube, keep, n)

        for i,j,k in fetch:
            self._request_cube((i,j,k))

        if sync:
            self._clear_q("all")

        if start_end is not None:
            return ijk_req

    def _release_cube(self, coords):
        self.cubes.pop(coords)
        mp_array = self.mpa_used.pop(coords)
        self.mpa_empty.append(mp_array)
        #print("cube %s deleted" %(coords,))

    def _release_cubes(self, center_cube, keep, n):
        release = []
        for coords in self.cubes:
            if coords not in keep:
                dist = np.linalg.norm((center_cube - coords) * [2,1,1])
                release.append((coords, dist))

        release.sort(key=lambda x: x[1], reverse=True)

        if len(release)< n: # not enough cubes available, they are still in loading q
            for i in range(len(release)): # release the possible cubes first
                self._release_cube(release[i][0])

            n_left = n - len(release)
            # now clear  the q
            try: # without waiting
                self._clear_q(None)
                self._release_cubes(center_cube, keep, n_left)
            except: # if it doesn't work, wait for all
                self._clear_q('all')
                self._release_cubes(center_cube, keep, n_left)
            return


        for i in range(n):
            self._release_cube(release[i][0])


    def _fetch_cube(self, coords):
        """
        Fetch a cube that is in cache or in loading q. i.e. this cube must have been preloaded
        Don't use this to request/preload a cube!
        """
        if coords not in self.cubes:
            assert coords in self.loading
            self._clear_q(coords)

        return self.cubes[coords]

    #@utils.timeit
    def cut_slice(self, shape, offset, out=None):
        #tt = utils.Timer()
        shape = np.array(shape, dtype=np.int)
        offset = np.array(offset, dtype=np.int)
        start = np.floor((offset)/self._cube_length).astype(np.int)
        end   = np.floor((offset+shape-1)/self._cube_length).astype(np.int) + 1
        position = offset + shape//2

        #tt.check("setup", True)
        ijk = self.preload(position=position, start_end=(start, end), sync=False)
        #tt.check("preload", True)
        if out is None:
            out = np.empty(shape, self.dtype)
        else:
            assert out.dtype==self.dtype
            assert np.allclose(out.shape, shape)

        l = self._cube_length
        for i,j,k in ijk: # Flat triple-nested loop
            values = self._fetch_cube((i,j,k))
            #tt.check("fetch", True)
            xl = max(offset[0]-i*l, 0)
            xh = min(offset[0]-i*l+shape[0], l)
            yl = max(offset[1]-j*l, 0)
            yh = min(offset[1]-j*l+shape[1], l)
            zl = max(offset[2]-k*l, 0)
            zh = min(offset[2]-k*l+shape[2], l)
            cut = values[xl:xh, yl:yh, zl:zh]
            #tt.check("cut", True)
            xl1 = max(i*l-offset[0], 0)
            xh1 = min((i+1)*l-offset[0], shape[0])
            yl1 = max(j*l-offset[1], 0)
            yh1 = min((j+1)*l-offset[1], shape[1])
            zl1 = max(k*l-offset[2], 0)
            zh1 = min((k+1)*l-offset[2], shape[2])
            out[xl1:xh1, yl1:yh1, zl1:zh1] = cut
            #tt.check('write', True)

        #tt.summary()
        return out

    def _normalise_idx(self, idx):
        new_idx = []
        shape = np.zeros(3, dtype=np.int)
        offset = np.zeros(3, dtype=np.int)
        for i,sl in enumerate(idx):
            if isinstance(sl, int):
                raise ValueError("Cannot slice single slices, only ranges")
            start, stop, step = sl.indices(self._shape_spatial[i])
            if start < 0 or stop>=self._shape_spatial[i]:
                raise ValueError("Slice %s out of bounce for knossos "
                                 "array of shape %s" %(idx[i], self._shape_spatial[i]))
            if step != 1:
                raise NotImplementedError("Subsampled slicing not implemented:"
                                          "instructins: use mag(stride) cubes "
                                          "and reuse slicing function")
            new_idx.append(slice(start, stop, step))
            shape[i] = (stop - start)//step
            offset[i] = start

        return shape, offset, new_idx

    def __getitem__(self, idx):
        assert len(idx) in [3,4]
        add_channel = False
        if len(idx)==4:
            assert idx[0].start==idx[0].stop==idx[0].step==None
            idx = idx[1:]
            add_channel = True
        shape, offset, new_idx = self._normalise_idx(idx)
        out = self.cut_slice(shape, offset)
        if add_channel:
            out = out[None]
        return out


class KnossosArrayMulti(KnossosArray):
    def __init__(self, path_prefix, feature_paths, max_ram=3000, n_preload=2, fixed_mag=1):
        self._arrays = []
        self._n_f = len(feature_paths)
        self.dtype = np.dtype(np.float32)
        for feature in feature_paths:
            p = os.path.expanduser(os.path.join(path_prefix,feature))
            self._arrays.append(KnossosArray(p, max_ram // self._n_f, n_preload, fixed_mag))


        self._shape = (self._n_f,) + tuple(self._arrays[0].shape)
        self._shape_spatial = self._arrays[0].shape

    def preload(self, position, sync=True):
        if len(position)==4:
            position = position[1:]

        for a in self._arrays:
            a.preload(position, sync=sync)

    def __getitem__(self, idx):
        assert len(idx)==4
        feature_slice = idx[0]
        idx = idx[1:]
        shape, offset, new_idx = self._normalise_idx(idx)
        feature_indices = list(range(*feature_slice.indices(self._n_f)))
        out = np.empty((len(feature_indices),)+tuple(shape), self.dtype)
        for i in feature_indices:
            self._arrays[i].cut_slice(shape, offset, out[i])

        return out

    def cut_slice(self, shape, offset, out=None):
        if out is None:
            out = np.empty((self._n_f,)+tuple(shape), self.dtype)

        for i in range(self._n_f):
            self._arrays[i].cut_slice(shape, offset, out[i])

        return out

    def __repr__(self):
        s = "<KnossosArray> %s\nFirst sub-array:\n"%(self.shape,)
        s += repr(self._arrays[0])
        return s


class KnossosArrayMultiMy(KnossosArrayMulti):
    def __getitem__(self, idx):
        tmp = super(KnossosArrayMultiMy, self).__getitem__(idx)
        sh = list(tmp.shape)
        sh[0] = 2
        out = np.empty(sh, dtype=np.float32)
        out[0] = tmp[0]
        out[1] = np.maximum(np.maximum(tmp[1], tmp[2]) - tmp[3], 0)
        return out

    @property
    def n_f(self):
        return self.n_f

    @property
    def shape(self):
        sh = list(self._shape)
        sh[0] = 2
        return tuple(sh)



if __name__ == "__main__":
    import time, sys
    sys.path.append(os.path.expanduser("~/axon/mkilling/devel/"))
    from knossos_utils import KnossosDataset
    from elektronn2.utils.plotting import scroll_plot, scroll_plot2
    path = os.path.expanduser("~/lustre/sdorkenw/j0126_3d_rrbarrier/")
    arr = KnossosArray(path, max_ram=1200, n_preload=2)

    arr.preload([0,0,0], sync=True)
    sh = np.array([120,240,240])
    off = np.array([0,0,0]) + np.random.randint(0, 500, size=3)

#    a = arr[0:80, 0:220, 0:230]
#    a0 = arr[0:100, 0:400, 0:200]
#    a1 = arr[0:100, 100:400+100, 100:200+100]
#    b = arr[0:640, 0:640, 0:384]
#    c = arr[500:600, 500:800, 500:800]
#
    kds = KnossosDataset()
    kds.initialize_from_knossos_path(path)
    get = utils.timeit(kds.from_raw_cubes_to_matrix)

#    a0 = arr.cut_slice(sh, off)
#    a1 = get(sh[[2,1,0]], off[[2,1,0]])
#    a2 = (a1.T).astype(np.float32)/255
#    print(np.allclose(a0, a2))
#    fig = scroll_plot2([a0, a2], 'ab')

    t = 0
    out =  np.empty(sh, dtype=np.float32)
    N = 500
    for i in range(N):
        t0 = time.time()
        a = arr.cut_slice(sh, off, out=out)
        t += time.time()-t0

        time.sleep(0.2)
        off  += np.random.randint(0, 10, size=3)


    print("%.3f slices/s" %(float(N)/t))

    arr._clear_q(None)
    fig = scroll_plot(out, 'a')










