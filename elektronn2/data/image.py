# -*- coding: utf-8 -*-
# ELEKTRONN2 Toolkit
# Copyright (c) 2015 Marius Killinger
# All rights reserved
from __future__ import absolute_import, division, print_function
from builtins import filter, hex, input, int, map, next, oct, pow, range, super, zip


__all__ = ['make_affinities', 'downsample_xy', 'ids2barriers', 'smearbarriers',
           'center_cubes', ]


import logging
import multiprocessing
from functools import reduce

import numba
from scipy import ndimage
import scipy.ndimage.filters as filters
from skimage.morphology import watershed
import numpy as np

from .. import malis
from .. import utils

logger = logging.getLogger('elektronn2log')

rand_index = malis.compute_V_rand_N2  # small is better

def make_affinities(labels, nhood=None, size_thresh=1):
    """
    Construct an affinity graph from a segmentation (IDs)

    Segments with ID 0 are regarded as disconnected
    The spatial shape of the affinity graph is the same as of seg_gt.
    This means that some edges are are undefined and therefore treated as disconnected.
    If the offsets in nhood are positive, the edges with largest spatial index are undefined.

    Connected components is run on the affgraph to relabel the IDs locally.

    Parameters
    ----------

    labels: 4d np.ndarray, int (any precision)
        Volumes of segmentation IDs (bs, z, y, x)
    nhood: 2d np.ndarray, int
        Neighbourhood pattern specifying the edges in the affinity graph
        Shape: (#edges, ndim)
        nhood[i] contains the displacement coordinates of edge i
        The number and order of edges is arbitrary
    size_thresh: int
        Size filters for connected components, smaller objects are mapped to BG


    Returns
    -------

    aff: 5d np.ndarray int16
        Affinity graph of shape (bs, #edges, x, y, z)
        1: connected, 0: disconnected
    seg_gt:
        4d np.ndarray int16
        Affinity graph of shape (bs, x, y, z)
        Relabelling of components
    """

    if nhood is None:
        nhood = np.eye(3, dtype=np.int32)

    aff_sh = [labels.shape[0], nhood.shape[0],] + list(labels.shape[1:])
    out_aff = np.zeros(aff_sh,       dtype=np.int16)
    out_seg = np.zeros(labels.shape, dtype=np.int16)
    for i,l in enumerate(labels):
        out_aff[i] = malis.seg_to_affgraph(l, nhood)
        # we throw away the seg sizes
        out_seg[i], _ = malis.affgraph_to_seg(out_aff[i], nhood, size_thresh)
    return out_aff, out_seg


def make_nhood_targets(target, nhood):
    # nhood (edges, displacements) e.g. (5,3)
    if target.ndim==4:
        raise NotImplementedError
    else:
        assert target.ndim==5
        assert target.shape[1]==1
        sh = target.shape
        new_sh = list(sh)
        new_sh[1] = len(nhood)
        new_target = -1 * np.ones(new_sh, dtype=target.dtype)
        sh = sh[2:]
        for i,off in enumerate(nhood):
            tmp = target[:,0,
                  max(0,-off[0]):sh[0]-max(0,off[0]),
                  max(0,-off[1]):sh[1]-max(0,off[1]),
                  max(0,-off[2]):sh[2]-max(0,off[2])]
            if tmp.size:
                new_target[:, i, max(0,off[0]):sh[0]-max(0,-off[0]),
                                 max(0,off[1]):sh[1]-max(0,-off[1]),
                                 max(0,off[2]):sh[2]-max(0,-off[2])] = tmp

    return new_target



def downsample_xy(d, l, factor):
    """
    Downsample by averaging
    :param d: data
    :param l: label
    :param factor:
    :return:
    """
    f     = int(factor)
    l_sh  = l.shape
    cut   = np.mod(l_sh, f)

    d     = d[:, :, :l_sh[-2]-cut[-2], :l_sh[-1]-cut[-1]]
    sh    = d[:, :, ::f, ::f].shape
    new_d = np.zeros(sh, dtype=np.float32)

    l     = l[:, :, l_sh[-2]-cut[-2], :l_sh[-1]-cut[-1]]
    sh    = l[:, :, :f, ::f].shape
    new_l = np.zeros(sh, dtype=l.dtype)

    for i in range(f):
        for j in range(f):
            new_d += d[:, :, i::f, j::f]
            new_l += l[:,    i::f, j::f]

    d = new_d / f**2
    l = new_l / f**2

    return d, l

@utils.timeit
@numba.jit(nopython=True)
def _ids2barriers(ids, barriers, dilute, connectivity):
    """
    Draw a 2 or 4 pix barrier where label IDs are different

    :param ids:  (x,y,z)
    :param barriers:
    :param dilute: e.g. [False, True, True]
    :param connectivity: e.g. [True, True, True]
    :return:
    """
    nx = ids.shape[0]
    ny = ids.shape[1]
    nz = ids.shape[2]

    for x in np.arange(nx-1):
        for y in np.arange(ny-1):
            for z in np.arange(nz-1):
                if connectivity[0]:
                    if ids[x,y,z]!=ids[x+1,y,z]:
                        barriers[x,y,z]   = 1
                        barriers[x+1,y,z] = 1
                        if dilute[0]:
                            if x>0:    barriers[x-1,y,z] = 1
                            if x<nx-2: barriers[x+2,y,z] = 1

                if connectivity[1]:
                    if ids[x,y,z]!=ids[x,y+1,z]:
                        barriers[x,y,z]   = 1
                        barriers[x,y+1,z] = 1
                        if dilute[1]:
                            if y>0:    barriers[x,y-1,z] = 1
                            if y<ny-2: barriers[x,y+2,z] = 1

                if connectivity[2]:
                    if ids[x,y,z]!=ids[x,y,z+1]:
                        barriers[x,y,z]   = 1
                        barriers[x,y,z+1] = 1
                        if dilute[2]:
                            if z>0:    barriers[x,y,z-1] = 1
                            if z<nz-2: barriers[x,y,z+2] = 1




def ids2barriers(ids, dilute=[True,True, True],
                 connectivity=[True, True, True],
                 ecs_as_barr=True,
                 smoothen=False):
    dilute = np.array(dilute)
    connectivity = np.array(connectivity)
    barriers = np.zeros_like(ids, dtype=np.int16)

    _ids2barriers(ids, barriers, dilute, connectivity)
    _ids2barriers(ids[::-1,::-1,::-1],
                  barriers[::-1,::-1,::-1],
                  dilute, connectivity) # apply backwards as lazy hack to fix boundary
                  
    if smoothen:
        kernel = np.array([[[0.1, 0.2, 0.1],
                            [0.2, 0.3, 0.2],
                            [0.1, 0.2, 0.1]],

                           [[0.3, 0.5, 0.3],
                            [0.5, 1.0, 0.5],
                            [0.3, 0.5, 0.3]],

                           [[0.1, 0.2, 0.1],
                            [0.2, 0.3, 0.2],
                            [0.1, 0.2, 0.1]]])
                            
        barriers_s = filters.convolve(barriers.astype(np.float32),
                                      kernel.astype(np.float32))
        barriers = (barriers_s>4).astype(np.int16) # (old - new).mean() ~ 0

    if ecs_as_barr=='new_class':
        ecs  = np.logical_and( (ids==0), (barriers!=1))
        barriers[ecs] = 2
    
    elif ecs_as_barr:
        ecs  = (ids==0).astype(np.int16)
        barriers = np.maximum(ecs, barriers)

    return barriers

def blob(sizes):
    """
    Return Gaussian blob filter
    """
    grids = [np.linspace(-2.2,2.2,size) for size in sizes]
    grids = np.meshgrid(*grids, indexing='ij')
    ret = np.exp(-0.5*(reduce(np.add, list(map(np.square, grids)))))
    ret = ret / np.square(ret).sum()
    return ret

def _smearbarriers(barriers, kernel):
    # Note: this is good but makes holes to small,
    #  besides we must raise/lower all confidences in GT
    barriers = barriers.astype(np.float32)
    if kernel is None:
        kernel = np.array([
            [[ 0.,  0.,  0.,  0.,  0.],
             [ 0.,  0.,  0.1,  0.,  0.],
             [ 0.,  0.1,  0.2,  0.1,  0.],
             [ 0.,  0.,  0.1,  0.,  0.],
             [ 0.,  0.,  0.,  0.,  0.]],

            [[ 0.,  0.,  0.1,  0.,  0.],
             [ 0.,  0.2,  0.4,  0.2,  0.],
             [ 0.1,  0.4,  1.,  0.4,  0.1],
             [ 0.,  0.2,  0.4,  0.2,  0.],
             [ 0.,  0.,  0.1,  0.,  0.]],

            [[ 0.,  0.,  0.,  0.,  0.],
             [ 0.,  0.,  0.1,  0.,  0.],
             [ 0.,  0.1,  0.2,  0.1,  0.],
             [ 0.,  0.,  0.1,  0.,  0.],
             [ 0.,  0.,  0.,  0.,  0.]],
        ]).T
    else:
        sizes = kernel
        kernel = blob(sizes)
        index = np.subtract(sizes, 1)
        index = np.divide(index, 2)
        kernel[tuple(index)] = 1.0 # set center to 1


    barriers = filters.convolve(barriers, kernel)
    barriers = np.minimum(barriers, 1.0)
    return barriers

def smearbarriers(barriers, kernel=None):
    """
    barriers: 3d volume (z,x,y)
    """
    pos = _smearbarriers(barriers, kernel)
    neg = 1.0 - _smearbarriers(1.0 - barriers, kernel)
    barriers = 0.5 * (pos + neg)
    #barriers = np.minimum(barriers, 1.0)
    return barriers


@numba.jit(nopython=True)
def _grow_seg(seg, grow, mask):
    nx = seg.shape[0]
    ny = seg.shape[1]
    nz = seg.shape[2]
    for x in range(1,nx-1):
        for y in range(1,ny-1):
            for z in range(1,nz-1):

                if mask[0] and (seg[x,y,z]!=0) and (seg[x-1,y,z]==0):
                    grow[x-1,y,z]   = seg[x,y,z]
                if mask[0] and (seg[x,y,z]!=0) and (seg[x+1,y,z]==0):
                    grow[x+1,y,z]   = seg[x,y,z]

                if mask[1] and (seg[x,y,z]!=0) and (seg[x,y-1,z]==0):
                    grow[x,y-1,z]   = seg[x,y,z]
                if mask[1] and (seg[x,y,z]!=0) and (seg[x,y+1,z]==0):
                    grow[x,y+1,z]   = seg[x,y,z]

                if mask[2] and (seg[x,y,z]!=0) and (seg[x,y,z-1]==0):
                    grow[x,y,z-1]   = seg[x,y,z]
                if mask[2] and (seg[x,y,z]!=0) and (seg[x,y,z+1]==0):
                    grow[x,y,z+1]   = seg[x,y,z]

def grow_seg(seg, pixel=[1,3,3]):
    """
    Grow segmentation labels into ECS/background by n pixel
    """
    if isinstance(pixel, (list, tuple, np.ndarray)):
        n = np.max(pixel)
    else:
        n = pixel
        pixel = [n,] * 3

    if n==0:
        return seg

    grow = seg.copy()
    for i in range(n):
        mask = np.greater(pixel, 0)
        _grow_seg(seg, grow, mask)
        seg = grow.copy()
        pixel = np.subtract(pixel, 1)

    return seg



def center_cubes(cube1, cube2, crop=True):
    """
    shapes (ch,x,y,z) or (x,y,z)
    """
    is_3d = [False, False]
    if cube1.ndim==3:
        cube1 = cube1[None]
        is_3d[0] = True
    if cube2.ndim==3:
        cube2 = cube2[None]
        is_3d[1] = True

    diffs = np.subtract(cube1.shape, cube2.shape)[1:]
    assert np.all(diffs%2==0)
    diffs //= 2

    slices1 = [slice(None)]
    pad1    = [(0,0)]
    slices2 = [slice(None)]
    pad2    = [(0,0)]
    for d in diffs:
        if d>0: # 1 is larger than 2
            if crop:
                slices1.append(slice(d, -d))
                pad1.append((0,0))

                slices2.append(slice(None))
                pad2.append((0,0))
            else:
                slices1.append(slice(None))
                pad1.append((0,0))

                slices2.append(slice(None))
                pad2.append((d,d))
        elif d<0:
            if crop:
                slices2.append(slice(-d, d))
                pad2.append((0,0))

                slices1.append(slice(None))
                pad1.append((0,0))
            else:
                slices2.append(slice(None))
                pad2.append((0,0))

                slices1.append(slice(None))
                pad1.append((-d, -d))
        else:
            slices2.append(slice(None))
            pad2.append((0,0))

            slices1.append(slice(None))
            pad1.append((0,0))

    cube1 = cube1[slices1]
    cube2 = cube2[slices2]
    cube1 = np.pad(cube1, pad1, 'constant')
    cube2 = np.pad(cube2, pad2, 'constant')

    if is_3d[0]:
        cube1 = cube1[0]
    if is_3d[1]:
        cube2 = cube2[0]

    return cube1, cube2

### Segmentation ##############################################################
def seg_old(pred=None, thresh=10.0, hi_thresh=140, lo_thresh=18,
            grow_it=4, slack_dt=True, scale = [20,9,9]):
    """
    pred: int prob map (z,x,y) !!!!!
    thresh: threshold for
    """

    # This with high threshold -> more holes
    mem_high = np.invert(((pred > hi_thresh) * 255).astype(np.uint8))
    dt_objects_ws     = -ndimage.distance_transform_edt(mem_high, sampling=scale)

    # This is with low threshold -> less holes, closes small stuff
    mem_low = np.invert(((pred > lo_thresh) * 255).astype(np.uint8))
    dt_objects_labels = -ndimage.distance_transform_edt(mem_low, sampling=scale)
    # Both distance transforms have 0 on the "hard membrane" and large negative values inside segments


    if slack_dt:
        smem = ((pred > hi_thresh) * 255).astype(np.uint8)
        dt_slack = -ndimage.distance_transform_edt(smem, sampling=scale)
        dt_comb = dt_slack + dt_objects_ws
        del dt_slack

    else:
        dt_comb = dt_objects_ws

    # create a slack label, this is label 1
    # regions where membrane/background/ecs is thick
    slack_label = ndimage.morphology.binary_erosion(pred > 150, iterations=3) * 1

    seeds, num = ndimage.measurements.label(dt_objects_labels<-thresh) # CC
    seeds[seeds!=0] += 1 # "make space" for the slack label seed with ID 1
    #print "SHIFT ON!"
    seeds[slack_label==1] = 1
    ws = watershed(dt_comb, seeds) -1
    ws = ws.astype(np.int16)
    ws = grow_seg(ws, pixel=grow_it)

    return ws

def seg_proc(kwargs):
    gt = kwargs.pop('gt')
    seg = seg_old(**kwargs)
    kwargs.pop('pred')
    ri, ri_split, ri_merge = rand_index(gt, seg)

    print("RI=%.4f\tthresh=%i\thi_thresh=%i\tlo_thresh=%i\tgrow=%s"\
          %(ri, kwargs['thresh'], kwargs['hi_thresh'], kwargs['lo_thresh'],
            kwargs['grow_it']))

    return ri

def optimise_segmentation(gt, pred, save_name, n_proc=2):
    threshs = [37,40,43,56] # 4
    hi_threshs = [235,240,245] # 4
    lo_threshs = [1,2,3] # 4
    grow_its   = [(0,0,0),(1,3,3),(2,6,6),(3,6,6)] # 3

    args = []
    for thresh in threshs:
        for grow_it in grow_its:
            for hi_thresh in hi_threshs:
                for lo_thresh in lo_threshs:
                    args.append(dict(gt=gt,
                                     pred=pred,
                                     thresh=thresh,
                                     hi_thresh=hi_thresh,
                                     lo_thresh=lo_thresh,
                                     grow_it=grow_it))

    print("Scanning for best SEGMENTATION parameters")
    mp = multiprocessing.Pool(n_proc)
    rand_indices  = list(mp.map(seg_proc, args))
    rand_indices = np.array(rand_indices)

    for i in range(len(args)):
        args[i].pop('gt')

    min_i = np.argmin(rand_indices)
    seg = seg_old(**args[min_i])
    best_ri = rand_indices[min_i]
    for i in range(len(args)):
        args[i].pop('pred')

    report_str = "%s Evaluation\n"%(save_name)
    report_str += "BEST: RI=%.4f "%rand_indices[min_i] + str(args[min_i]) + '\n'


    for re, config in zip(rand_indices, args):
        report_str += "RI=%.4f "%re + str(config) + '\n'

    with open("%s-Seg_Params.txt" %(save_name), 'w') as f:
        f.write(report_str)

    utils.h5save(randomise_colours(seg), '%s_seg.h5' %(save_name,), 'seg')
    print("Best RI=%.4f" %(best_ri))

    return rand_indices, best_ri, seg

def randomise_colours(a, size_filter=1500):
    out = np.zeros_like(a)
    bc = np.bincount(a.ravel())
    colors = (bc>size_filter).nonzero()[0]
    new_cols = np.random.permutation(len(colors)-1)
    new_cols   = np.hstack([0, new_cols])
    for c in colors:
        i = np.argmax(colors==c)
        out[a==c] = new_cols[i]

    return out


def billig_seg(gt, pred, thresh, ecs_thresh):
    seeds, num = ndimage.measurements.label((pred<thresh))
    ws = watershed(pred, seeds)
    ws[pred>ecs_thresh] = 0
    seg = ws
    ri, ri_split, ri_merge = rand_index(gt, seg)
    return ri, seg


def billig_seg_proc(kwargs):
    gt = kwargs['gt']
    pred = kwargs['pred']
    thresh = kwargs['thresh']
    ecs_thresh = kwargs['ecs_thresh']
    ri, seg = billig_seg(gt, pred, thresh, ecs_thresh)
    print("RI=%.4f\tthresh=%i\tecs_thresh=%i"%(ri, thresh, ecs_thresh))
    return ri

def optimise_billig_segmentation(gt, pred, save_name, n_proc=2):
    threshs = np.linspace(100, 240, 6) # 4
    ecs_threshs = [253,] #np.linspace(230, 255,5) # 4

    args = []
    for thresh in threshs:
        for ecs_thresh in ecs_threshs:
            args.append(dict(gt=gt,
                             pred=pred,
                             thresh=thresh,
                             ecs_thresh=ecs_thresh))

    print("Scanning for best SEGMENTATION parameters")
    mp = multiprocessing.Pool(n_proc)
    rand_indices  = list(mp.map(billig_seg_proc, args))
    rand_indices = np.array(rand_indices)

    min_i = np.argmin(rand_indices)
    a = args[min_i]
    ri, seg = billig_seg(a['gt'], a['pred'], a['thresh'], a['ecs_thresh'])

    for i in range(len(args)):
        args[i].pop('pred', None)
        args[i].pop('gt', None)

    report_str = "%s Evaluation\n"%(save_name)
    report_str += "BEST: RI=%.4f "%rand_indices[min_i] + str(args[min_i]) + '\n'

    for re, config in zip(rand_indices, args):
        report_str += "RI=%.4f "%re + str(config) + '\n'

    with open("%s-Seg_Params.txt" %(save_name), 'w') as f:
        f.write(report_str)

    utils.h5save(randomise_colours(seg), '%s_seg.h5' %(save_name,), 'seg')
    print("Best RI=%.4f" %(rand_indices[min_i]))

    return rand_indices[min_i], seg
