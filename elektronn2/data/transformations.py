# -*- coding: utf-8 -*-
# ELEKTRONN2 Toolkit
# Copyright (c) 2015 Martin Drawitsch, Marius Killinger
# All rights reserved

from __future__ import absolute_import, division, print_function
from builtins import filter, hex, input, int, map, next, oct, pow, range, super, zip


__all__ = ['warp_slice', 'get_tracing_slice', 'WarpingOOBError',
           'Transform', 'trafo_from_array']


import itertools
import logging
from functools import reduce

import numpy as np
import numba

from .. import utils

logger = logging.getLogger('elektronn2log')
inspection_logger = logging.getLogger('elektronn2log-inspection')


# @utils.my_jit(nopython=True, cache=True)
# def map_coordinates_nearest(src, coords, lo, dest):
#     sh = coords.shape
#     for z in np.arange(sh[0]):
#         for x in np.arange(sh[1]):
#             for y in np.arange(sh[2]):
#                 u = np.int32(np.round(coords[z,x,y,0] - lo[0]))
#                 v = np.int32(np.round(coords[z,x,y,1] - lo[1]))
#                 w = np.int32(np.round(coords[z,x,y,2] - lo[2]))
#                 or
#                 u = coords[z, x, y, 0] - lo[0]
#                 v = coords[z, x, y, 1] - lo[1]
#                 w = coords[z, x, y, 2] - lo[2]
#                 dest[z,x,y] = src[u,v,w]

@numba.guvectorize(['void(float32[:,:,:], float32[:], float32[:], float32[:,],)'],
              '(x,y,z),(i),(i)->()', target='parallel', nopython=True)
def map_coordinates_nearest(src, coords, lo, dest):
    u = np.int32(np.round(coords[0] - lo[0]))
    v = np.int32(np.round(coords[1] - lo[1]))
    w = np.int32(np.round(coords[2] - lo[2]))
    dest[0] = src[u,v,w]

@numba.guvectorize(['void(float32[:,:,:], float32[:], float32[:], float32[:,],)'],
              '(x,y,z),(i),(i)->()', target='parallel', nopython=True)
def map_coordinates_linear(src, coords, lo, dest):
    u = coords[0] - lo[0]
    v = coords[1] - lo[1]
    w = coords[2] - lo[2]
    u0 = np.int32(u)
    u1 = u0 + 1
    du = u - u0
    v0 = np.int32(v)
    v1 = v0 + 1
    dv = v - v0
    w0 = np.int32(w)
    w1 = w0 + 1
    dw = w - w0
    val = src[u0, v0, w0] * (1-du) * (1-dv) * (1-dw) +\
          src[u1, v0, w0] * du * (1-dv) * (1-dw) +\
          src[u0, v1, w0] * (1-du) * dv * (1-dw) +\
          src[u0, v0, w1] * (1-du) * (1-dv) * dw +\
          src[u1, v0, w1] * du * (1-dv) * dw +\
          src[u0, v1, w1] * (1-du) * dv * dw +\
          src[u1, v1, w0] * du * dv * (1-dw) +\
          src[u1, v1, w1] * du * dv * dw
    dest[0] = val

@utils.my_jit(nopython=True, cache=True)
def map_coordinates_max_kernel(src, coords, lo, k, dest):
    k = np.float32(k)
    kz = min(0.5, k/2)
    sh = coords.shape
    sh_src = src.shape
    for z in np.arange(sh[0]):
        for x in np.arange(sh[1]):
            for y in np.arange(sh[2]):
                u = coords[z,x,y,0] - lo[0]
                v = coords[z,x,y,1] - lo[1]
                w = coords[z,x,y,2] - lo[2]

                u0 = np.int32(np.round(max(0, min(sh_src[0], u - kz))))
                u1 = np.int32(np.round(max(0, min(sh_src[0], u + kz))))
                v0 = np.int32(np.round(max(0, min(sh_src[1], v - k))))
                v1 = np.int32(np.round(max(0, min(sh_src[1], v + k))))
                w0 = np.int32(np.round(max(0, min(sh_src[2], w - k))))
                w1 = np.int32(np.round(max(0, min(sh_src[2], w + k))))

                val = src[u0:u1, v0:v1, w0:w1].max()

                dest[z, x, y] = val


def identity():
    return np.eye(4, dtype=np.float32)

def translate(dz, dy, dx):
    return np.array([
        [1.0, 0.0, 0.0,  dz],
        [0.0, 1.0, 0.0,  dy],
        [0.0, 0.0, 1.0,  dx],
        [0.0, 0.0, 0.0, 1.0]
    ], dtype=np.float32)

def rotate_z(a):
    return np.array([
        [1.0, 0.0,    0.0,     0.0],
        [0.0, np.cos(a), -np.sin(a), 0.0],
        [0.0, np.sin(a), np.cos(a),  0.0],
        [0.0, 0.0,    0.0,     1.0]
    ], dtype=np.float32)

def rotate_y(a):
    return np.array([
        [np.cos(a), -np.sin(a), 0.0, 0.0],
        [np.sin(a),  np.cos(a), 0.0, 0.0],
        [0.0,        0.0, 1.0, 0.0],
        [0.0,        0.0, 0.0, 1.0]
    ], dtype=np.float32)

def rotate_x(a):
    return np.array([
        [np.cos(a),  0.0, np.sin(a), 0.0],
        [0.0,     1.0, 0.0,    0.0],
        [-np.sin(a), 0.0, np.cos(a), 0.0],
        [0.0,     0.0, 0.0,    1.0]
    ], dtype=np.float32)

def scale_inv(mz, my, mx):
    return np.array([
        [1/mz,  0.0,    0.0,  0.0],
        [0.0,   1/my,   0.0,  0.0],
        [0.0,   0.0,    1/mx, 0.0],
        [0.0,   0.0,    0.0,  1.0]
    ], dtype=np.float32)

def scale(mz, my, mx):
    return np.array([
        [mz,  0.0, 0.0, 0.0],
        [0.0, my,  0.0, 0.0],
        [0.0, 0.0, mx,  0.0],
        [0.0, 0.0, 0.0, 1.0]
    ], dtype=np.float32)

def chain_matrices(mat_list):
    return reduce(np.dot, mat_list, identity())

def get_euler_angles(direc, gamma):
    """
    tracing_dir (z, x, y) normalised
    angle3 rotation around z in dest frame
    phi is the rotation about the 1-2 axis
    theta is the rotation about the 0'-1' axis
    """
    assert abs(np.linalg.norm(direc) - 1) < 1e-3
    phi = np.arctan2(direc[2], direc[1])
    theta = np.arccos(direc[0])
    return phi, theta, gamma

def get_rotmat_from_direc(direc, gamma=None, rng=None):
    if gamma is None:
        gamma=0.0
    elif gamma=='rand':
        if rng is None:
            gamma = np.random.rand() * 2 * np.pi
        else:
            gamma = rng.rand() * 2 * np.pi

    phi, theta, gamma = get_euler_angles(direc, gamma)
    R1 = rotate_z(-phi)
    R2 = rotate_y(-theta)
    R3 = rotate_z(gamma)
    R = chain_matrices([R3, R2, R1])
    return R

def get_random_rotmat(lock_z=False, amount=1.0, rng=None):
    rng = np.random.RandomState() if rng is None else rng

    gamma = rng.rand() * 2 * np.pi * amount
    if lock_z:
        return rotate_z(gamma)

    phi = rng.rand() * 2 * np.pi * amount
    theta = np.arcsin(rng.rand()) * amount

    R1 = rotate_z(-phi)
    R2 = rotate_y(-theta)
    R3 = rotate_z(gamma)
    R = chain_matrices([R3, R2, R1])
    return R

def get_random_flipmat(no_x_flip=False, rng=None):
    rng = np.random.RandomState() if rng is None else rng
    F = np.eye(4, dtype=np.float32)
    flips = rng.binomial(1, 0.5, 4) * 2 - 1
    flips[3] = 1 # don't flip homogenious dimension
    if no_x_flip:
        flips[2] = 1

    np.fill_diagonal(F, flips)
    return F

def get_random_swapmat(lock_z=False, rng=None):
    rng = np.random.RandomState() if rng is None else rng
    S = np.eye(4, dtype=np.float32)
    if lock_z:
        swaps = [[0, 1, 2, 3],
                 [0, 2, 1, 3]]
    else:
        swaps = [[0, 1, 2, 3],
                 [0, 2, 1, 3],
                 [1, 0, 2, 3],
                 [1, 2, 0, 3],
                 [2, 0, 1, 3],
                 [2, 1, 0, 3]]

    i = rng.randint(0, len(swaps))
    S = S[swaps[i]]
    return S

def get_random_warpmat(lock_z=False, perspective=False, amount=1.0, rng=None):
    W = np.eye(4, dtype=np.float32)
    amount *= 0.1
    perturb = np.random.uniform(-amount, amount, (4, 4))
    perturb[3,3] = 0
    if lock_z:
        perturb[0] = 0
        perturb[:,0] = 0
    if not perspective:
        perturb[3] = 0

    perturb[3,:3] *= 0.05 # perspective parameters need to be very small
    np.clip(perturb[3,:3], -3e-3, 3e-3, out=perturb[3,:3])

    return W + perturb

@utils.cache()
def make_dest_coords(sh):
    """
    Make coordinate list for destiantion array of shape sh
    """
    zz,xx,yy = np.mgrid[0:sh[0], 0:sh[1], 0:sh[2]]
    hh = np.ones(sh, dtype=np.int)
    coords = np.concatenate([zz[...,None], xx[...,None],
                             yy[...,None], hh[...,None]], axis=-1)
    return coords.astype(np.float32)

def make_dest_corners(sh):
    """
    Make coordinate list of the corners of destination array of shape sh
    """
    corners = list(itertools.product(*([0,1],)*3))
    sh = np.subtract(sh, 1) # 0-based indices
    corners = np.multiply(sh, corners)
    corners = np.hstack((corners, np.ones((8,1)))) # homogeneous coords
    return corners


class WarpingOOBError(ValueError):
    def __init__(self, *args, **kwargs):
        super(WarpingOOBError, self).__init__( *args, **kwargs)


class Transform(object):
    def __init__(self, M, position_l=None, aniso_factor=2):
        self.M = M
        self.M_inv = np.linalg.inv(M.astype(np.float64)).astype(np.float32) # stability...
        self.position_l = position_l
        self.aniso_factor = aniso_factor
        self.is_projective = not np.allclose(M[3,:3], 0.0)

    @property
    def M_lin(self):
        if self.is_projective:
            raise ValueError("This transform requires homogenious coordinates")
        else:
            return self.M[:3,:3]

    @property
    def M_lin_inv(self):
        if self.is_projective:
            raise ValueError("This transform requires homogenious coordinates")
        else:
            return self.M_inv[:3, :3]

    def to_array(self):
        return np.hstack([self.M.ravel(), self.position_l, self.aniso_factor])

    def lab_coord2cnn_coord(self, vec_l):
        assert not self.is_projective
        if vec_l.ndim==1:
            vec_c = np.dot(self.M_lin, vec_l)  # rotation
        else:
            # assume vec_l.shape=(n,3)
            assert vec_l.shape[1]==3
            vec_c = np.dot(vec_l, self.M_lin.T)  # rotation
        return vec_c


    def cnn_coord2lab_coord(self, vec_c, add_offset_l=False):
        assert not self.is_projective
        if vec_c.ndim==1:
            vec_l = np.dot(self.M_lin_inv, vec_c)  # rotation
            if add_offset_l:
                vec_l += self.position_l
        else:
            # assume vec_l.shape=(n,3)
            assert vec_c.shape[1]==3
            vec_l = np.dot(vec_c, self.M_lin_inv.T)  # rotation
            if add_offset_l:
                vec_l += self.position_l[None,:]
        return vec_l


    def cnn_pred2lab_position(self, prediction_c):
        assert not self.is_projective
        tracin_direc_l = self.cnn_coord2lab_coord(prediction_c, add_offset_l=False)
        new_position_l = tracin_direc_l + self.position_l
        tracing_direc_il = tracin_direc_l * [self.aniso_factor,1,1]
        assert np.linalg.norm(tracing_direc_il) > 0 # normalise
        tracing_direc_il /= np.linalg.norm(tracing_direc_il)
        return new_position_l, tracing_direc_il

def trafo_from_array(a):
    M = a[:16].reshape((4,4))
    offset_l = a[16:19]
    aniso_factor = a[19]
    return Transform(M, offset_l, aniso_factor)



def warp_slice(img, ps, M, target=None, target_ps=None,
               target_vec_ix=None, target_discrete_ix=None,
               last_ch_max_interp=False, ksize=0.5):
    """
    :param img: (f, z, x, y)
    :param ps: (spatial only) patch_size (z,x,y)
    :param M: forward tansform, must contain translations in source and target array!
    :param target: optional target array to be extracted in the same way
    :param target_ps:
    :param target_vec_ix: list of triples that denote vector value parts in the
     target array e.g. [(0,1,2),(4,5,6)] denotes two vectorfields separated
     by a scalar field in channel 3
    :return:
    """
    #T = utils.Timer(silent_all=True)
    ps = tuple(ps)
    if len(img.shape)==3: # For single knossos_array
        n_f = 1
        sh = img.shape
    elif len(img.shape)==4:
        n_f = img.shape[0]
        sh = img.shape[1:]
    else:
        raise ValueError('img wrong dim/shape')

    M_inv = np.linalg.inv(M.astype(np.float64)).astype(np.float32) # stability...
    dest_corners = make_dest_corners(ps)
    src_corners = np.dot(M_inv, dest_corners.T).T
    if np.any(M[3,:3] != 0): # homogeneous divide
        src_corners /= src_corners[:,3][:,None]

    # check corners
    src_corners = src_corners[:,:3]
    lo = np.min(np.floor(src_corners), 0).astype(np.int)
    hi = np.max(np.ceil(src_corners + 1), 0).astype(np.int) # add 1 because linear interp
    if np.any(lo < 0) or np.any(hi >= sh):
        raise WarpingOOBError("Out of bounds")
    #T.check("corners")
    # compute/transform dense coords
    dest_coords = make_dest_coords(ps)
    src_coords = np.tensordot(dest_coords, M_inv, axes=[[-1],[1]])
    if np.any(M[3,:3] != 0): # homogeneous divide
        src_coords /= src_coords[...,3][...,None]
    #T.check("tensordot")
    # cut patch
    src_coords = src_coords[...,:3]
    img_cut = img[:, lo[0]:hi[0]+1, #add 1 to include this coordinate!
                     lo[1]:hi[1]+1,
                     lo[2]:hi[2]+1,]
    img_cut = np.ascontiguousarray(img_cut, dtype=np.float32)
    #T.check("cut")
    img_new = np.zeros((n_f,)+ps, dtype=np.float32)
    #T.check("alloc")
    lo = lo.astype(np.float32)
    for k in range(n_f):
        if (ksize>0.5) and last_ch_max_interp and (k==n_f-1):
            map_coordinates_max_kernel(img_cut[k], src_coords, lo, ksize, img_new[k])
        else:
            map_coordinates_linear(img_cut[k], src_coords, lo, img_new[k])
    #T.check("map img")

    if np.isnan(img_new).sum():
        print(np.isnan(img_new).sum())
        print("shit")

    if target is not None:
        target_ps = tuple(target_ps)
        n_f_t = target.shape[0]

        off = np.subtract(sh, target.shape[1:])
        if np.any(np.mod(off, 2)):
            raise ValueError("targets must be centered w.r.t to images")
        off //= 2

        off_ps = np.subtract(ps, target_ps)
        if np.any(np.mod(off_ps, 2)):
            raise ValueError("targets must be centered w.r.t to images")
        off_ps //= 2

        src_coords_target = src_coords[off_ps[0]:off_ps[0]+target_ps[0],
                                       off_ps[1]:off_ps[1]+target_ps[1],
                                       off_ps[2]:off_ps[2]+target_ps[2]]
        # shift coords to be w.r.t to origin of target array
        lo_targ = np.floor(src_coords_target.min(2).min(1).min(0) - off).astype(np.int)
        # add 1 because linear interp
        hi_targ = np.ceil(src_coords_target.max(2).max(1).max(0) - off + 1).astype(np.int)
        if np.any(lo_targ < 0) or np.any(hi_targ >= target.shape[1:]):
             raise WarpingOOBError("Out of bounds for target")
        #T.check("target_check")
        target_cut = target[:, lo_targ[0]:hi_targ[0]+1, #add 1 to include this coordinate!
                               lo_targ[1]:hi_targ[1]+1,
                               lo_targ[2]:hi_targ[2]+1]

        target_cut = np.ascontiguousarray(target_cut, dtype=np.float32)
        src_coords_target = np.ascontiguousarray(src_coords_target, dtype=np.float32)
        #T.check("target_cut")
        target_new = np.zeros((n_f_t,)+target_ps, dtype=np.float32)
        #T.check("target_alloc")
        lo_targ = (lo_targ + off).astype(np.float32)
        if target_discrete_ix is None:
            target_discrete_ix = [True for i in range(n_f_t)]
        else:
            target_discrete_ix = [i in target_discrete_ix for i in range(n_f_t)]

        for k, discr in enumerate(target_discrete_ix):
            if discr:
                map_coordinates_nearest(target_cut[k], src_coords_target, lo_targ, target_new[k])
            else:
                map_coordinates_linear(target_cut[k], src_coords_target, lo_targ, target_new[k])

        #T.check("target_map")
        if target_vec_ix is not None: # Vectors must be transformed again
            assert np.allclose(M[3,:3], 0.0) # no projective transform
            M_lin = M[:3,:3]
            for ix in target_vec_ix:
                assert len(ix)==3
                target_new[ix] = np.tensordot(M_lin, target_new[ix], axes=[[1],[0]])
        #T.check("target_vec")
    else:
        target_new =  None
    #T.summary()
    return img_new, target_new

def get_tracing_slice(img, ps, pos, z_shift=0, aniso_factor=2,
                      sample_aniso=True, gamma=0, scale_factor=1.0, direction_iso=None,
                      target=None, target_ps=None, target_vec_ix=None,
                      target_discrete_ix=None, rng=None, last_ch_max_interp=False):

    # positive z_shift --> see more slices in positive z-direction w.r.t pos
    # scale_factor > 1 zooms into image / magifies
    rng = np.random.RandomState() if rng is None else rng
    dest_center = np.array(ps, dtype=np.float)/2
    dest_center[0] -= z_shift
    R = get_rotmat_from_direc(direction_iso, gamma, rng)
    T_src = translate(-pos[0], -pos[1], -pos[2])
    S_src = scale(aniso_factor, 1, 1)
    S_zoom = scale(scale_factor, scale_factor, scale_factor)

    if sample_aniso:
        S_dest = scale_inv(aniso_factor, 1, 1)
    else:
        S_dest = identity()
    T_dest = translate(dest_center[0], dest_center[1], dest_center[2])

    M = chain_matrices([T_dest, S_zoom, S_dest, R, S_src, T_src])
    ksize = min(0.5, 0.5/scale_factor)
    img_new, target_new = warp_slice(img, ps, M,
                     target=target,
                     target_ps=target_ps,
                     target_vec_ix=target_vec_ix,
                     target_discrete_ix=target_discrete_ix,
                     last_ch_max_interp=last_ch_max_interp,
                     ksize=ksize)

    return img_new, target_new, M


def get_warped_slice(img, ps, aniso_factor=2, sample_aniso=True,
                    warp_amount=1.0, lock_z=True, no_x_flip=False, perspective=False,
                    target=None, target_ps=None, target_vec_ix=None,
                    target_discrete_ix=None, rng=None):
    rng = np.random.RandomState() if rng is None else rng


    strip_2d = False
    if len(ps)==2:
        strip_2d = True
        ps = np.array([1]+list(ps))
        if target is not None:
            target_ps = np.array([1]+list(target_ps))


    dest_center = np.array(ps, dtype=np.float) / 2
    src_remainder =  np.array(np.mod(ps, 2), dtype=np.float) / 2

    if target_ps is not None:
        t_center = np.array(target_ps, dtype=np.float) / 2
        off = np.subtract(img.shape[1:], target.shape[1:])
        off //= 2
        lo_pos = np.maximum(dest_center, t_center+off)
        hi_pos = np.minimum(img.shape[1:] - dest_center, target.shape[1:] - t_center + off)
    else:
        lo_pos = dest_center
        hi_pos = img.shape[1:] - dest_center

    z = rng.randint(lo_pos[0], hi_pos[0]) + src_remainder[0]
    y = rng.randint(lo_pos[1], hi_pos[1]) + src_remainder[1]
    x = rng.randint(lo_pos[2], hi_pos[2]) + src_remainder[2]

    F = get_random_flipmat(no_x_flip, rng)
    if no_x_flip:
        S = np.eye(4, dtype=np.float32)
    else:
        S = get_random_swapmat(lock_z, rng)

    if np.isclose(warp_amount, 0):
        R = np.eye(4, dtype=np.float32)
        W = np.eye(4, dtype=np.float32)
    else:
        R = get_random_rotmat(lock_z, warp_amount, rng)
        W = get_random_warpmat(lock_z, perspective, warp_amount, rng)

    T_src = translate(-z, -y, -x)
    S_src = scale(aniso_factor, 1, 1)

    if sample_aniso:
        S_dest = scale(1.0 / aniso_factor, 1, 1)
    else:
        S_dest = identity()
    T_dest = translate(dest_center[0], dest_center[1], dest_center[2])

    M = chain_matrices([T_dest, S_dest, R, W, F, S, S_src, T_src])

#    if target_ps is not None and np.allclose(target_ps, 1):
#        new_target =

    img_new, target_new = warp_slice(img, ps, M,
                                     target=target,
                                     target_ps=target_ps,
                                     target_vec_ix=target_vec_ix,
                                     target_discrete_ix=target_discrete_ix)

    if strip_2d:
        img_new = img_new[:,0]
        if target is not None:
            target_new = target_new[:,0]

    return img_new, target_new





if __name__ =='__main__':
#    pos = [120,175,175]
#    import time
#    from elektronn2 import utils
#    from elektronn2.utils.plotting import scroll_plot
#
#    raw = np.random.rand(250,500,500).astype(np.float32)
#    if True: # insert stripes
#        raw[::10,:,:] = 0
#        raw[:,::20,:] = 0
#        raw[:,:,::20] = 0
#
#    img = raw[None]
#    R = chain_matrices([rotate_y(0.2), rotate_z(0.1)])
#    ps = [46,171,171]
#    z_shift = 8
#    aniso_factor = 2
#    sample_aniso=True
#
#    dest_center = np.array(ps, dtype=np.float32)/2
#    dest_center[0] -= z_shift # sign?
#    T_src = translate(-pos[0], -pos[1], -pos[2])
#    S_src = scale(aniso_factor, 1, 1)
#    if sample_aniso:
#        S_dest = scale(1.0/aniso_factor, 1, 1)
#    else:
#        S_dest = identity()
#    T_dest = translate(dest_center[0], dest_center[1], dest_center[2])
#
#    M = chain_matrices([T_dest, S_dest, R, S_src, T_src])
#
#    warp_time = utils.timeit(warp_slice)
#    for i in range(10):
#        img_new = warp_time(img, ps, M)[0]
#    f = scroll_plot(img_new[0])
    pos = [150,300,300]
    from elektronn2.utils.plotting import scroll_plot
    from elektronn2.data.image import center_cubes
    warp_time = utils.timeit(warp_slice)

    raw = utils.h5load('~/lustre/mkilling/BirdGT/j0126_old/v2_old_0-raw-zyx.h5')
    if True: # insert stripes
        raw[:,::10,:,:] = 0
        raw[:,:,::20,:] = 0
        raw[:,:,:,::20] = 0

    img = raw

    targ = utils.h5load('~/lustre/mkilling/BirdGT/j0126_old/v2_old_0-combo-sparse-zyx.h5')
    targ[:,::10,:,:] = 0
    targ[:,:,::20,:] = 0
    targ[:,:,:,::20] = 0
    targ = targ

    # --------------------------------
    R = chain_matrices([rotate_y(0.05), rotate_z(0.2)])
    #direc = np.array([1,0.15, 0.05])
    #direc /= np.linalg.norm(direc)
    #direc = [1,0,0]
    #R = get_rotmat_from_direc(direc, 0.2)
    #R[3,1] = 0.002
    ps = [31,131,131]

    z_shift = 0
    aniso_factor = 2
    sample_aniso=True

    dest_center = np.array(ps, dtype=np.float32)/2
    dest_center[0] -= z_shift # sign?
    T_src = translate(-pos[0], -pos[1], -pos[2])
    S_src = scale(aniso_factor, 1, 1)
    if sample_aniso:
        S_dest = scale(1.0/aniso_factor, 1, 1)
    else:
        S_dest = identity()
    T_dest = translate(dest_center[0], dest_center[1], dest_center[2])

    M = chain_matrices([T_dest, S_dest, R, S_src, T_src])

    for i in range(5):
        img_new, targ_new = warp_time(img, ps, M, target=targ, target_ps=[11,121,121], target_discrete_ix=[0,1,2])

    #f0 = scroll_plot(img_new[0])
    i, t = center_cubes(img_new, targ_new)
    f1 = scroll_plot([i[0], t[0]])
