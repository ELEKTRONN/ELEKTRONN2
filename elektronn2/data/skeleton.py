# -*- coding: utf-8 -*-
# ELEKTRONN2 Toolkit
# Copyright (c) 2015 Marius Killinger
# All rights reserved
from __future__ import absolute_import, division, print_function
from builtins import filter, hex, input, int, map, next, oct, pow, range, super, zip
# TODO: Python 3 compatibility

__all__ = ['trace_zyx2xyz', 'trace_to_kzip', 'SkeletonMFK',
           'Trace']

import os
import sys
from subprocess import check_call
import logging
import getpass
from collections import OrderedDict

import numba
from scipy import interpolate
from scipy import sparse
from scipy.sparse import csgraph
import numpy as np

from .. import utils

from ..config import config
from . import transformations

logger = logging.getLogger('elektronn2log')
inspection_logger = logging.getLogger('elektronn2log-inspection')

if sys.version_info[:2] != (2, 7):
    raise ImportError(
        '\nSorry, this module only supports Python 2.7.'
        '\nYour current Python version is {}\n'.format(sys.version)
    )

try:
    from knossos_utils import skeleton as knossos_skeleton
except ImportError as e:
    logger.error('\nFor using the tracing_utils module, you will need to'
                 ' install the knossos_utils module'
                 ' (https://github.com/knossos-project/knossos_utils)\n')
    raise e

user_name = getpass.getuser()

with open(os.devnull, 'w') as devnull:
    # mayavi is to dumb to raise an exception and instead crashes whole script....
    try:
        # "xset q" will always succeed to run if an X server is currently running
        check_call(['xset', 'q'], stdout=devnull, stderr=devnull)
        import mayavi.mlab as mlab
        # Don't set backend explicitly, use system default...
    # if "xset q" fails, conclude that X is not running
    except: # (OSError, ImportError, CalledProcessError, ValueError)
        logger.warning("No mayavi imported, cannot plot skeletons")
        mlab = None


###############################################################################

# Constants for scaling of radius
REF_RADIUS = 20.0
BASE = 1.3
BASE_I = BASE ** -1
HYST = 0.75  # 0.5: no memory 1.0 complete non-overlap
assert 0.5 <= HYST <= 1.0
BASE_H = BASE ** HYST
BASE_IH = BASE ** -HYST


@utils.timeit
@numba.jit(nopython=True)
def insert(cube, coords, i, off):
    for k in np.arange(coords.shape[0]):
        cube[coords[k,0]-off[0], coords[k,1]-off[1], coords[k,2]-off[2]] = i


@utils.timeit
@numba.jit(nopython=True)
def insert_vec(cube, coords, vec, off):
    n = len(coords)
    m = len(vec[0])
    double_inserts = 0
    for i in np.arange(n):
        for j in np.arange(m):
            if abs(cube[coords[i,0]-off[0],
                        coords[i,1]-off[1],
                        coords[i,2]-off[2],j])>1e-5:
                double_inserts += 1
                cube[coords[i,0]-off[0], coords[i,1]-off[1], coords[i,2]-off[2],j] = np.nan # in case of doubt, don't train here...
            else:
                cube[coords[i,0]-off[0], coords[i,1]-off[1], coords[i,2]-off[2],j] = vec[i, j]

    return double_inserts


@utils.timeit
@numba.jit(nopython=True)
def ray_cast(max_dists, hull_points, hull_dist, ray_steps, hull_cube, off):
    s = np.float32(0.9) # step length
    sh = hull_cube.shape
    for i in np.arange(len(hull_points)): # take hull point
        # initialise dist and position
        dist = hull_dist[i] + 1e-5
        x = hull_points[i, 0] - off[0]
        y = hull_points[i, 1] - off[1]
        z = hull_points[i, 2] - off[2]
        found = False
        count = 0
        while True:
            count += 1
            x = x + s * ray_steps[i,0]
            y = y + s * ray_steps[i,1]
            z = z + s * ray_steps[i,2]
            if np.int(x)<0.0 or np.int(y)<0.0 or np.int(z)<0.0:
                break
            if np.int(x+0.5)>=sh[0] or np.int(y+0.5)>=sh[1] or np.int(z+0.5)>=sh[2]:
                break
            # search if hull is True in neighbourhood of x,y,z
            found = hull_cube[np.int(x), np.int(y), np.int(z)] or \
                    hull_cube[np.int(x+0.5), np.int(y), np.int(z)] or \
                    hull_cube[np.int(x), np.int(y+0.5), np.int(z)] or \
                    hull_cube[np.int(x), np.int(y), np.int(z+0.5)] or \
                    hull_cube[np.int(x+0.5), np.int(y+0.5), np.int(z)] or \
                    hull_cube[np.int(x+0.5), np.int(y), np.int(z+0.5)] or \
                    hull_cube[np.int(x), np.int(y+0.5), np.int(z+0.5)] or \
                    hull_cube[np.int(x+0.5), np.int(y+0.5), np.int(z+0.5)]
            if not found:
                break
            if count>200:
                break

            dist = dist + s

        max_dists[i] = dist

@numba.jit(nopython=True, cache=True)
def find_peaks_helper(padded_cube, peak_cube):
    sh = padded_cube.shape
    for z in np.arange(1,sh[0]-1):
        for x in np.arange(1,sh[1]-1):
            for y in np.arange(1,sh[2]-1):
                center = padded_cube[z,x,y]
                is_peak = center >= padded_cube[z-1, x-1, y-1] and \
                          center >= padded_cube[z+1, x+1, y+1] and \
                          center >= padded_cube[z+1, x+1, y-1] and \
                          center >= padded_cube[z+1, x-1, y+1] and \
                          center >= padded_cube[z-1, x+1, y+1] and \
                          center >= padded_cube[z+1, x-1, y-1] and \
                          center >= padded_cube[z-1, x-1, y+1] and \
                          center >= padded_cube[z-1, x+1, y-1] and \
                          center >= padded_cube[z, x, y+1] and \
                          center >= padded_cube[z, x, y-1] and \
                          center >= padded_cube[z, x+1, y] and \
                          center >= padded_cube[z, x-1, y] and \
                          center >= padded_cube[z+1, x, y] and \
                          center >= padded_cube[z-1, x, y] and \
                          center >= padded_cube[z, x+1, y+1] and \
                          center >= padded_cube[z, x-1, y-1] and \
                          center >= padded_cube[z+1, x+1, y] and \
                          center >= padded_cube[z-1, x-1, y] and \
                          center >= padded_cube[z+1, x, y+1] and \
                          center >= padded_cube[z-1, x, y-1] and \
                          center >= padded_cube[z, x-1, y+1] and \
                          center >= padded_cube[z, x+1, y-1] and \
                          center >= padded_cube[z+1, x-1, y] and \
                          center >= padded_cube[z+1, x-1, y] and \
                          center >= padded_cube[z-1, x, y+1] and \
                          center >= padded_cube[z-1, x, y+1]
                if is_peak:
                    peak_cube[z-1,x-1,y-1] = center

def find_peaks(cube):
    padded_cube = np.pad(cube, 1, mode='constant')
    peaks = np.zeros_like(cube)
    find_peaks_helper(padded_cube, peaks)
    #peak_label, n = ndimage.label(peaks)
    #coordinates = ndimage.center_of_mass(peaks, peak_label, index=np.arange(1,n+1))
    indices = np.flatnonzero(peaks)
    maxima = cube.ravel()[indices]
    sort_ix = np.argsort(maxima)
    return indices[sort_ix], maxima[sort_ix]



# WARNING / NOTE: skeleton objects are in xyz-order
class SkeletonMFK(object):
    """
    Joints: all branches and end points / node terminations (nodes not of deg 2)
    Branches: Joints of degree >= 3
    """
    @staticmethod
    def find_joints(node_list):
        joints = {}
        branches = {}
        for node in node_list:
            if node.degree() > 2: # branching point
                joints[node.ID] = node
                branches[node.ID] = node
            if node.degree()==1:
                joints[node.ID] = node # end point

        return joints, branches

    def __init__(self, aniso_scale=2, name=None, skel_num=None):
        self.aniso_scale = np.array([[1,1,aniso_scale]], dtype=np.float32)
        self.bones = dict()
        self.edges = list()
        self.branches = dict()
        self.joints = dict()
        self.all_nodes = None
        self.hull_points  = None
        self.hull_skel = dict()
        self.hull_branch = dict()
        self.name = name
        self.skel_num = skel_num

        self.radii = dict()
        self.all_radii = None
        self.joint_radii = None

        self.props = dict()
        self.all_props = None
        self.joint_props = None

        self.joint_id2joint_index = dict()

        # For training
        self.kdt_hull = None
        self.linked_data = None
        self.lost_track = False
        self.position_s = None
        self.position_l = None
        self.direction_il = None
        self.start_new_training = True
        self.prev_batch = None
        self.trafo = None
        self.prev_scale = 1.0
        self.prev_gamma = 0.0
        self.training_traces = []
        self.background_processes = False
        self._hull_point_bg = dict()
        self.cnn_grid = None
        # Old for training
        self.debug_traces = []
        self.debug_traces_current = []
        self.debug_grads = []
        self.debug_grads_current = []


    def init_from_annotation(self, skeleton_annotatation, min_radius=None,
                             interpolation_resolution=0.5, interpolation_order=1):
        # Read annotation data structures and convert to dicts and np.ndarrays
        #print(len(skeleton_annotatation.getNodes()))
        self.joints, self.branches = self.find_joints(skeleton_annotatation.getNodes())
        #print(len(self.joints), len(self.branches))

        visited = {n: False for n in skeleton_annotatation.getNodes()}
        for joint_id, joint in self.joints.items():
            directions = joint.getNeighbors()
            for d in directions:
                if visited[d]: # we have visited this bone already
                    continue
                visited[d] = True # mark as visited
                bone = OrderedDict() # create new bone
                bone[joint] = True # start the bone at the joint
                current_node = d # next go to the node in the selected direction
                while True:
                    bone[current_node] = True
                    if current_node.degree() > 2 or current_node.degree()==1:
                        # At new branch or end point, the bone ends here
                        # add edge between starting joint and this branch
                        if joint_id < current_node.ID:
                            edge = (joint_id, current_node.ID)
                        else:
                            edge = (current_node.ID, joint_id)
                        self.edges.append(edge)
                        break
                    else: # The node has 2 neighbours, one from which we come and
                        # another one to which we go
                        nb = list(current_node.getNeighbors())
                        assert len(nb) == 2
                        # Test which node we visit next
                        if nb[0] in bone:
                            assert nb[1] not in bone
                            current_node = nb[1]
                        if nb[1] in bone:
                            assert nb[0] not in bone
                            current_node = nb[0]

                self.bones[edge] = list(bone.keys())

        # Convert bones to arrays
        for edge, bone in self.bones.items():
            self.bones[edge] = np.array([x.getCoordinate() for x in bone], dtype=np.float32)
            self.radii[edge] = np.array([x.getDataElem('radius') for x in bone], dtype=np.float32)
            try:
                axoness_pred = np.array([x.getDataElem('axoness_pred') for x in bone], dtype=np.int16)
                spiness_pred = np.array([x.getDataElem('spiness_pred') for x in bone], dtype=np.int16)
                props = np.concatenate([axoness_pred[:,None], spiness_pred[:,None]], axis=1)
                self.props[edge] = props
            except KeyError:
                pass




        # convert joints to arrays
        self.joint_radii = np.array([x.getDataElem('radius') for x in self.joints.values()], dtype=np.float32)
        try:
            axoness_pred = np.array([x.getDataElem('axoness_pred') for x in self.joints.values()], dtype=np.int16)
            spiness_pred = np.array([x.getDataElem('spiness_pred') for x in self.joints.values()], dtype=np.int16)
            self.joint_props = np.concatenate([axoness_pred[:,None], spiness_pred[:,None]], axis=1)
        except KeyError:
           pass

        self.joint_id2joint_index = dict(zip(self.joints.keys(), range(len(self.joints))))
        self.joints = np.array([x.getCoordinate() for x in self.joints.values()], dtype=np.float32)

        # convert branches to arrays
        self.branches = np.array([x.getCoordinate() for x in self.branches.values()], dtype=np.float32)

        if interpolation_resolution is not None:
            for edge, bone in self.bones.items():
                if len(bone)<=1:
                    continue
                try:
                    new_bone = self.interpolate_bone(bone,max_k=interpolation_order,
                                resolution=interpolation_resolution)
                    self.radii[edge] = self.interpolate_prop(bone, self.radii[edge],
                             new_bone)
                except:
                    bone, keep_index = utils.unique_rows(bone)

                    new_bone = self.interpolate_bone(bone,max_k=interpolation_order,
                               resolution=interpolation_resolution)
                    self.radii[edge] = self.interpolate_prop(bone, self.radii[edge][keep_index],
                             new_bone)

                try:
                    self.props[edge] = self.interpolate_prop(bone,
                                    self.props[edge],new_bone, discrete=True)
                except:
                    pass

                self.bones[edge] = new_bone

        self.all_nodes = np.vstack([self.joints,] + list(self.bones.values()))
        self.all_radii = np.hstack([self.joint_radii,] + list(self.radii.values()))
        try:
            self.all_props = np.vstack([self.joint_props, ] + list(self.props.values()))
        except:
            pass

        if min_radius:
            self.all_radii = np.maximum(self.all_radii, min_radius)

    def save(self, fname):
        utils.picklesave(self, fname)

    def interpolate_bone(self, bone, max_k=1, resolution=0.5):
        bone_iso = bone * self.aniso_scale
        linear_distances = np.linalg.norm(np.diff(bone_iso, axis=0), axis=1)
        total_dist = linear_distances.sum()
        k = min(max_k, bone_iso.shape[0]-1)
        tck, u = interpolate.splprep(bone_iso.T, k=k)
        n = max(2, int(float(total_dist) / resolution))
        new = interpolate.splev(np.linspace(0,1,n), tck)
        new = np.array(new).T / self.aniso_scale
        return new

    def interpolate_prop(self, old_bone, old_prop, new_bone, discrete=False):
        dtype = np.int16 if discrete else np.float32
        new_prop = np.zeros((len(new_bone),)+old_prop.shape[1:], dtype=dtype)
        old_bone_iso = old_bone * self.aniso_scale
        new_bone_iso = new_bone * self.aniso_scale
        start_i = 0
        stop_i  = 1
        min_dist = np.linalg.norm(new_bone_iso[0] - old_bone_iso[stop_i])

        for i in range(len(new_bone)):
            dist_start = np.linalg.norm(new_bone_iso[i] - old_bone_iso[start_i])
            dist_stop  = np.linalg.norm(new_bone_iso[i] - old_bone_iso[stop_i])
            min_dist = min(min_dist, dist_stop)

            if (min_dist < dist_stop) and stop_i+1<len(old_bone):
                stop_i += 1
                start_i += 1
                dist_start = dist_stop
                dist_stop = np.linalg.norm(new_bone_iso[i] - old_bone_iso[stop_i])
                min_dist = dist_stop

            if discrete:
                if dist_stop > dist_start:
                    new_prop[i] = old_prop[start_i]
                else:
                    new_prop[i] = old_prop[stop_i]
            else:
                d = dist_start + dist_stop
                new_prop[i] = dist_stop/d * old_prop[start_i] + dist_start/d * old_prop[stop_i]

        return new_prop


    @utils.cache()
    def get_kdtree(self, static_points, k=1, jobs=-1):
        kdt = utils.KDT(n_neighbors=k, n_jobs=jobs,
                        algorithm='kd_tree', leaf_size=20)
        kdt.fit(static_points * self.aniso_scale) # change metric)
        #assert np.all(kdt._fit_X / self.aniso_scale == static_points)
        return kdt


    @utils.cache()
    def get_knn(self, kdt, query_points, k=None):
        if k is not None:
            pass
            #assert k==kdt.n_neighbors
        else:
            k = kdt.n_neighbors

        distances, indices =  kdt.kneighbors(query_points * self.aniso_scale, n_neighbors=k) # change metric)
        static_points = kdt._fit_X.astype(np.float32) # Attention those still have the aniso scale in [:,2]
        if k==1:
            indices = indices[:,0]
            distances = distances[:,0].astype(np.float32)
            coordinates = static_points[indices] / self.aniso_scale # change to pixel coordinates
        else:
            distances = distances.astype(np.float32)
            coordinates = static_points[indices] / self.aniso_scale # change to pixel coordinates
            assert coordinates.shape[1] == k

        return distances, indices, coordinates


    def get_closest_node(self, position_s):
        kdt = self.get_kdtree(self.all_nodes, k=1, jobs=1)
        dist, ind, nearest_s = self.get_knn(kdt, position_s)

        if position_s.ndim==1:
            dist = dist[0]
            ind = ind[0]
            nearest_s = nearest_s[0]

        return dist.astype(np.float32), ind, nearest_s


    ### Sampling routines for getting training data ###
    def sample_skel_point(self, rng, joint_ratio=None):
        n = len(self.all_nodes)
        if joint_ratio:
            if rng.rand() < joint_ratio:
                n = len(self.joints)

        i = rng.randint(n)
        node = self.all_nodes[i]
        return  node, i


    def sample_tube_point(self, rng, r_max_scale=0.9, joint_ratio=None):
        """
        This is skeleton node based sampling:
        Go to a random node, sample a random orthogonal direction
        go a random distance into direction (uniform over the
        [0, r_max_scale * local maximal radius])
        """
        # tt = utils.Timer()
        if self.hull_points is None:
            kdt = None
        else:
            if self.kdt_hull is None:
                raise RuntimeError("Hull kdts must be pre initialised")
            kdt = self.kdt_hull

        node, node_i = self.sample_skel_point(rng, joint_ratio)
        direc_iso = self.sample_local_direction_iso(node)
        local_r = self.all_radii[node_i] * r_max_scale

        count = 0
        max_count = 30
        proposal = node
        clipped = False
        while True:
            r = rng.rand() * local_r
            phi = rng.rand() * 2 * np.pi
            cos_theta = rng.rand() * 2 - 1
            sin_theta = np.sqrt(1 - cos_theta ** 2)
            x = np.cos(phi) * sin_theta
            y = np.sin(phi) * sin_theta
            z = cos_theta
            rand_vec = np.array([x, y, z])

            orthogonal_vec_iso = np.cross(direc_iso, rand_vec)
            orthogonal_vec_iso /= np.linalg.norm(orthogonal_vec_iso)
            orthogonal_vec = orthogonal_vec_iso / self.aniso_scale[0]

            proposal = node + orthogonal_vec * r

            if kdt is None:
                return proposal

            dist, ind, coord = self.get_knn(kdt, proposal)
            dist = dist[0]
            if dist < 1.5:  # we are within hull:
                break
            if count >= max_count / 2 and not clipped:
                local_r *= 0.5
                clipped = True
                logger.debug("Sample hull point: clipped r")
            if count >= max_count:
                logger.debug(
                    "Sample hull point: max count %i reached" % max_count)
                proposal = node
                break

            # tt.check("\tdouble_check")
            count += 1

        return proposal


    def sample_local_direction_iso(self, point, n_neighbors=6):
        """
        For a point gives the local skeleton direction/orientation by
        fitting a line through the nearest neighbours, sign is randomly assigned
        """
        kdt = self.get_kdtree(self.all_nodes, k=n_neighbors, jobs=1)
        dist, ind, coord = self.get_knn(kdt, point)
        dist = dist[0]
        ind = ind[0]
        coord = coord[0]
        # maybe use dist as weights for svd?
        neibs_iso = coord * self.aniso_scale # transform to iso space
        uu, dd, vv = np.linalg.svd(neibs_iso - neibs_iso.mean(axis=0))
        direc_iso = vv[0] # take largest eigenvector
        direc_iso /= np.linalg.norm(direc_iso, axis=0) # normalise
        return direc_iso


    def sample_tracing_direction_iso(self, rng, local_direction_iso, c=0.5):
        """
        Sample a direction close to the local direction
        there is a prior so that the normalised (0,1) angle of deviation a
        has this distribution:
        p(a) = 1/N * (1-c*a), where N= 1 - c/2,
        tmp is the inverse cdf of this.
        """
        if rng.rand() > 0.5: # the sign is undefined, choose randomly
            local_direction_iso *= -1

        u = rng.rand()
        tmp = (1 - np.sqrt(1-(2*c - c**2)*u)) /  c # theta scaled between 0 and 1
        # theta scaled between 0 and 90 deg in rad i.e. 0 and pi/2
        theta = tmp * 0.5 * np.pi
        max_count = 1000
        count = 0
        proposal = local_direction_iso
        while True:
            proposal = rng.rand(3) * 2 - 1
            proposal /= np.linalg.norm(proposal, axis=0) # normalise
            cos_alpha = np.dot(proposal, local_direction_iso)
            if cos_alpha < 0: # flip to next best within +/- 90 deg
                cos_alpha *= -1
                proposal *= -1

            alpha = np.arccos(cos_alpha)
            if alpha < theta + 0.01:
                break

            count += 1
            if count>max_count:
                logger.debug("Sample tracing directions: max count reached")
                break

        return proposal


    ### Loss and loss gradient for Theano Graph ###
    def get_loss_and_gradient(self, new_position_s, cutoff_inner=1.0/3,
                              rise_factor=0.1):
        """
        prediction_c (zxy)
        Zoned error surface:
        flat in inner hull (selected at cutoff_inner)
        constant gradient in "outer" hull towards nearest inner hull voxel
        gradient increasing with distance (scaled by rise_factor)
        for predictions outside hull
        """
        inner_hull, indices = self.get_hull_points_inner(cutoff_inner,
                                                         return_indices=True)
        kdt = self.get_kdtree(inner_hull, k=1, jobs=1)
        dist, ind, nearest_s = self.get_knn(kdt, new_position_s)
        dist = dist[0]
        ind = ind[0]
        nearest_s = nearest_s[0]

        if config.inspection:
            inspection_logger.info("nearest_s: %s"% (nearest_s.tolist()))

        if dist<1.5: # we are within inner hull. The maximal distance if
            # within hull is exactly: np.linalg.norm(np.multiply(
            # [0.5, 0.5, 0.6], [1, 1, 2])) = 1.22... --> add some margin
            loss = 0.0
            grad_s = np.zeros((3,), dtype=np.float32)
            self.lost_track = False

        else:
            loss = dist
            # max dist of closest node
            max_dist = self.hull_skel['max_dist'][indices[ind]]
            # pointing from nearest to new position
            unit_grad = (new_position_s - nearest_s)
            unit_grad /= np.linalg.norm(unit_grad * self.aniso_scale[0], axis=0)
            if max_dist > dist: # we are in hull but not in inner tube
                grad_s = unit_grad * 1.0
                self.lost_track = False
            else: # we are outside hull
                self.lost_track = True
                factor = rise_factor * (dist - max_dist)
                grad_s = unit_grad * (1 + factor)

        self.debug_traces_current.append(new_position_s)
        self.debug_grads_current.append(grad_s)
        loss = np.array([loss,], dtype=np.float32)
        return loss, grad_s


    def _new_training_trace(self, **get_batch_kwargs):
        """
        Prepare skeleton for a new training (sample location/direction,
        reset stuff)

        Parameters
        ----------
        get_batch_kwargs

        """
        #tt = utils.Timer()
        if self.current_trace:
            if len(self.training_traces)>20:
                self.training_traces = self.training_traces[-2:]

            self.training_traces.append(self.current_trace)
        self.current_trace = Trace(linked_skel=self)

        r_max_scale = get_batch_kwargs['r_max_scale']
        tracing_dir_prior_c = get_batch_kwargs['tracing_dir_prior_c']
        joint_ratio = get_batch_kwargs.get('joint_ratio', None)

        position_s = self.sample_tube_point(self.linked_data.rng,
                                            r_max_scale=r_max_scale,
                                            joint_ratio=joint_ratio)

        if config.inspection:
            inspection_logger.info("Start new training")
        local_direc_is = self.sample_local_direction_iso(position_s,
                                                         n_neighbors=6)
        tracing_direc_is = self.sample_tracing_direction_iso(self.linked_data.rng,
                                                             local_direc_is,
                                                             c=tracing_dir_prior_c)

        self.position_s = position_s
        self.position_l = position_s[::-1] # from lab2data (xyz)->(zxy)
        self.direction_il = tracing_direc_is[::-1]  # from lab2data (xyz)->(zxy)

        self.current_trace.append(position_s, coord_cnn=[0,]*3,
                                  grad=[0,]*3, features=[0,]*7)

        self.lost_track = False
        self.trafo = None
        #tt.check("final")


    @staticmethod
    def get_scale_factor(radius, old_factor, scale_strenght):
        """
        Parameters
        ----------
        radius: predicted radius (not the true radius)
        old_factor: factor by which the radius prediction and the image was scaled
        scale_strenght: limits the maximal scale factor

        Returns
        -------
        new_factor

        """
        # if old was large (zoom in), radius is smaller
        hi = BASE ** (scale_strenght * 2) + 1e-3  # e.g 1.69 for 1.3**2
        lo = BASE_I ** (scale_strenght * 4) - 1e-3  # e.g. 0.35 for 1/1.3 ** 4

        radius_true = radius / old_factor
        new_factor = REF_RADIUS / radius_true
        new_factor = np.clip(new_factor, lo, hi)
        change = new_factor / old_factor
        if new_factor > 1.0:  # left side
            if change >= BASE_H:  # growing
                new_factor = old_factor * BASE
            elif change < BASE_IH:
                new_factor = old_factor * BASE_I
            else:
                new_factor = old_factor
        elif new_factor < 1.0:  # right side
            if change <= BASE_IH:  # zoom out
                new_factor = old_factor * BASE_I
            elif change > BASE_H:  # zoom in
                new_factor = old_factor * BASE
            else:
                new_factor = old_factor
        else:
            new_factor = old_factor
        if config.inspection:
            inspection_logger.info("SCALE: %.2f -> %.2f, factor0: %.2f, factor: %.2f"
                               % (radius, radius_true, 20.0 / radius_true, new_factor))
        return new_factor

    @staticmethod
    @utils.cache
    def make_grid(t_grid_sh, z_shift):
        """
        Parameters
        ----------
        t_grid_sh: tagged shape (pixel shape + strides)
        z_shift: shift of center (positive means more look ahead)

        Returns
        -------
        points: coordinate list zyx order
        zz,yy,xx: coordinate meshgrid
        """
        sh = np.array(t_grid_sh.spatial_shape)
        st = np.array(t_grid_sh.strides)
        lim = (sh-1) * st + 1
        lim //= 2
        zz,yy,xx = np.mgrid[-lim[0]:lim[0]:1j * sh[0],
                            -lim[1]:lim[1]:1j * sh[1],
                            -lim[2]:lim[2]:1j * sh[2]]

        zz += z_shift
        points = np.hstack([zz.ravel()[:,None],
                            yy.ravel()[:,None],
                            xx.ravel()[:,None]]).astype(np.float32)
        return points, zz,yy,xx


    @staticmethod
    def point_potential(r, margin_scale, size, repulsion=None):
        if repulsion is None:
            repulsion = 1.0
        left = margin_scale * size
        x = (r - left)/(size - left)
        v = 1.0 - (x**3*(x*(x*6 - 15) + 10)) # soft step function
        v = np.minimum(np.maximum(v, 0.0), 1.0)
        return v * repulsion


    def getbatch(self, prediction, scale_strenght, **get_batch_kwargs):
        """
        Parameters
        ----------
        prediction: [[new_position_c, radius, ]]
        scale_strenght: limits the maximal scale factor for zoom
        get_batch_kwargs

        Returns
        -------
        batch: img, target_img, target_grid, target_node
        """
        get_batch_kwargs = dict(get_batch_kwargs) # copy because we destroy it
        if self.start_new_training:
            self._new_training_trace(**get_batch_kwargs)
            self.start_new_training = False
            scale = 1.0
            self.prev_scale = 1.0
            self.prev_gamma = np.random.rand() * 2 * np.pi

        elif np.allclose(prediction, 0):
            scale = self.prev_scale
            if config.inspection:
                inspection_logger.warning("getbatch with no feedback: either "
                                          "training on same skel or error")
        else:
            prediction = prediction[0]
            new_position_c = prediction[:3]
            radius = prediction[3] # this is just the predicted val, not the true
            new_position_l, tracing_direc_il = self.trafo.cnn_pred2lab_position(new_position_c)
            new_position_s = new_position_l[::-1]
            self.position_s = new_position_s
            self.position_l = new_position_s[::-1] # from lab2data (xyz)->(zxy)
            self.direction_il = tracing_direc_il
            scale = self.get_scale_factor(radius, self.prev_scale, scale_strenght)
            self.prev_scale = scale


        grid = get_batch_kwargs.pop('grid', False)
        t_grid_sh = get_batch_kwargs.pop('t_grid_sh', None)
        z_shift   = get_batch_kwargs['z_shift']
        get_batch_kwargs.pop('joint_ratio', None)
        try:
            if config.inspection:
                inspection_logger.info("Getslice from position_l %s in "
                                       "direction_il %s, SCALE %.2f"%(np.array_str(
                                        self.position_l, precision=1,
                                        suppress_small=True),
                                        self.direction_il, scale))
            get_batch_kwargs['gamma'] = self.prev_gamma
            data_batch = self.linked_data.get_newslice(self.position_l,
                                                  self.direction_il,
                                                  scale=scale,
                                                  **get_batch_kwargs)
            img, target_img, trafo = data_batch[:3]

            if grid:
                raise RuntimeError("The creation of the grid target must"
                                   "be testet for spatial coherence again")
                if not self.cnn_grid:
                    self.cnn_grid = self.make_grid(t_grid_sh, z_shift)

                grid_coords_c, zz, yy, xx = self.cnn_grid
                #dir_point_s     = self.position_s + self.direction_il[::-1]/self.aniso_scale[0]
                #dir_momentum_s  = dir_point_s - self.current_trace.coords[-4:].mean(0)
                #dir_momentum_ci = trafo.lab_coord2cnn_coord(dir_momentum_s[::-1])*[2,1,1]
                #directions_ci   = grid_coords_c*[2,1,1]
                #direction_difference = cdist(directions_ci, dir_momentum_ci[None], 'cosine')  # 0..2, 45deg thresh: > 1.7
                #direction_difference = (direction_difference[:,0] - 1.7).astype(np.float32)
                #direction_difference[direction_difference<0.0] = 0.0
                #direction_difference[np.isnan(direction_difference)] = 0.0 # center if even is NULL
                #repulsion = 1.0 - direction_difference * 2 # * strength, without factor ~ -25%
                ### TODO might also make repulsion depending on skel_node instead of grid_position. No WHY?
                ### TODO repulsion is not smooth enough
                repulsion = 1.0

                grid_coords = trafo.cnn_coord2lab_coord(grid_coords_c,add_offset_l=True)
                dist, ind, nearest_s = self.get_closest_node(grid_coords[:,::-1])
                radii = self.all_radii[ind]
                target_grid = self.point_potential(dist, 0.1, radii, repulsion)
                target_grid = target_grid.reshape(zz.shape)[None] # add channel
                if np.allclose(target_grid, 0.0):
                    logger.warning("WTF")

                self.debug_store = [img, target_grid]
                self.debug_store2 = [nearest_s, radii]
            else:
                target_grid = np.ones((1,1,1,1), dtype=np.float32)

            # Get bio labels/classes
            dist, ind, nearest_s = self.get_closest_node(self.position_s)
            classes = self.all_props[ind]
            target_node = np.zeros(7, dtype=np.float32)
            target_node[classes[0]+1] = 1
            target_node[classes[1]+4] = 1
            target_node[0] = self.all_radii[ind] * scale
            if config.inspection:
                inspection_logger.info("target_node %s, (true r: %.1f)"
                                       %(target_node, self.all_radii[ind]))

            batch = (img, target_img, target_grid, target_node)
            self.trafo = trafo
            return batch

        except transformations.WarpingOOBError:
            if config.inspection: inspection_logger.info("OOB in getbatch")
            raise transformations.WarpingOOBError("Batch OOB")


    def step_feedback(self, new_position_s, new_direction_is, pred_c,
                      pred_features, cutoff_inner=1.0/3, rise_factor=0.1):

        inner_hull, indices = self.get_hull_points_inner(cutoff_inner, return_indices=True)
        kdt = self.get_kdtree(inner_hull, k=1, jobs=1)
        dist, ind, nearest_s = self.get_knn(kdt, new_position_s)
        dist = dist[0]
        ind = ind[0]
        nearest_s = nearest_s[0]
        # we are within inner hull. The maximal distance if within hull is 1.2...
        if dist < 1.5:
            loss = 0.0
            grad_s = np.array([0, 0, 0], dtype=np.float32)

        else:
            loss = dist
            max_dist = self.hull_skel['max_dist'][indices[ind]]  # max dist of closest node
            unit_grad = (new_position_s - nearest_s)  # pointing from nearest to new position
            unit_grad /= np.linalg.norm(unit_grad * self.aniso_scale[0],
                                        axis=0)  # normalise grad
            if max_dist > dist:  # we are in hull but not in inner tube
                grad_s = unit_grad * 1.0
            else:  # we are outside hull
                self.lost_track = True
                if config.inspection:
                    inspection_logger.info("Lost track")
                factor = rise_factor * (dist - max_dist)
                grad_s = unit_grad * (1 + factor)

        self.current_trace.append(new_position_s, coord_cnn=pred_c,
                                  grad=grad_s, features=pred_features)
        # Actually the new positions should be set in getbach, but we need to
        # set them here to because sometimes getbatch might be called without
        # "start_new_training" and with only zeros as prediction
        self.position_s = new_position_s
        self.position_l = new_position_s[::-1]  # from lab to data frame (xyz) -> (zxy)
        self.direction_il = new_direction_is[::-1]  # from lab to data frame (xyz) -> (zxy)

        loss = np.array([loss,], dtype=np.float32)
        return loss, grad_s, nearest_s


    def step_grid_update(self, grid, radius, bio):
        pred_features = np.hstack([radius, bio])
        flat_indices, scores = find_peaks(grid[0,0])
        grid_coords_c, zz, yy, xx = self.cnn_grid
        preds_c = grid_coords_c[flat_indices]
        #preds_l = self.trafo.cnn_coord2lab_coord(preds_c,add_offset_l=True)

        if len(scores):
            new_position_c = grid_coords_c[flat_indices[-1]]
            preds_c = new_position_c[None]

        else:
            new_position_c = np.array([2,0,0], dtype=np.float32)

        new_position_l, tracing_direc_il = self.trafo.cnn_pred2lab_position(new_position_c)
        new_position_s = new_position_l[::-1]
        new_direction_is = tracing_direc_il[::-1]

        if config.inspection:
            inspection_logger.info("GridUpdate, node pred %s" % (
            np.array_str(pred_features, precision=2,
                         suppress_small=True),))
            inspection_logger.info(
                "GridUpdate, new_position_c: %s, new_position_l: %s" % (
                new_position_c, np.array_str(new_position_l, precision=1,
                                             suppress_small=True)))
        if config.inspection>1:
            img, grid_t = self.debug_store
            utils.picklesave(
                [img[0,0], grid_t[0], grid[0,0]], '/tmp/{}_debug_skel_{}'.format(user_name, self.skel_num))

        self.current_trace.append(new_position_s, coord_cnn=new_position_c,
                                  features=pred_features)
        # Actually the new positions should be set in getbach, but we need to
        # set them here to because sometimes getbatch might be called without
        # "start_new_training" and with only zeros as prediction
        self.position_s = new_position_s
        self.position_l = new_position_s[::-1]  # from lab to data frame (xyz) -> (zxy)
        self.direction_il = new_direction_is[::-1]  # from lab to data frame (xyz) -> (zxy)

        return new_position_c[None], preds_c, scores


    ### Plotting ###
    def plot_skel(self, fig=None):
        if fig is None:
            fig = mlab.figure(bgcolor=(1.0, 0.8, 0.4), size=(600,400))

        x = self.all_nodes[:,0]
        y = self.all_nodes[:,1]
        z = self.all_nodes[:,2]*self.aniso_scale[0,2]
        mlab.points3d(x,y,z, scale_factor=0.8, color=(1,0,0), figure=fig)
        for bone in self.bones.values():
            x = bone[:,0]
            y = bone[:,1]
            z = bone[:,2]*self.aniso_scale[0,2]
            mlab.plot3d(x,y,z,tube_radius=0.4, color=(0.3,0.3,0.3), figure=fig)

        self._plot_joints(fig=fig)
        return fig

    def plot_debug_traces(self, grads=True, fig=None):
        if fig is None:
            fig = mlab.figure(bgcolor=(1.0, 0.8, 0.4), size=(600,400))

        traces = np.array(self.debug_traces)
        for trace in traces:
            x = trace[:, 0]
            y = trace[:, 1]
            z = trace[:, 2] * self.aniso_scale[0, 2]
            mlab.plot3d(x, y, z, tube_radius=0.2, color=(0.3, 0.3, 0.3), figure=fig)

        if grads:
            grads = np.array(self.debug_grads)
            for grad, trace in zip(grads, traces):
                x = trace[:, 0]
                y = trace[:, 1]
                z = trace[:, 2] * self.aniso_scale[0, 2]
                gx = -grad[:, 0]
                gy = -grad[:, 1]
                gz = -grad[:, 2] * self.aniso_scale[0, 2]
                mlab.quiver3d(x,y,z, gx, gy, gz, figure=fig,
                              color=(0,0.6,0.2), scale_factor=3)

        return fig

    def plot_radii(self, fig=None):
        if fig is None:
            fig = mlab.figure(bgcolor=(1.0, 0.8, 0.4), size=(600,400))

        x = self.all_nodes[:,0]
        y = self.all_nodes[:,1]
        z = self.all_nodes[:,2]*self.aniso_scale[0,2]
        r = self.all_radii
        mlab.points3d(x,y,z,r, scale_mode='scalar', scale_factor=1,
                            color=(0,0.5,0.5), mode='sphere', opacity=0.1, figure=fig)

        return fig

    def _plot_joints(self, fig=None):
        if fig is None:
            fig = mlab.figure(bgcolor=(1.0, 0.8, 0.4), size=(600,400))

        x = self.joints[:,0]
        y = self.joints[:,1]
        z = self.joints[:,2]*self.aniso_scale[0,2]
        mlab.points3d(x,y,z, scale_factor=3, color=(1,1,0), figure=fig)
        return fig


    ### Hull methods ###
    def calc_max_dist_to_skels(self):
        hull = self.hull_points # (n, 3)
        direc = self.hull_skel['direc'] #(n, 3)
        dist = self.hull_skel['dist'] #(n) # true distances
        max_dist = np.zeros(len(hull), dtype=np.float32)

        # This ray has unit magnitude in the true metric
        ray_steps = -direc/(np.linalg.norm(direc * self.aniso_scale, axis=1)[:,None]+1e-5)
        # create dense cube and insert hull
        sh = np.max(hull,0) + 1
        off = np.min(hull,0)
        sh -= off
        hull_cube = np.zeros(sh, dtype=np.bool)
        insert(hull_cube, hull, True, off)
        # cast rays through dense cube
        ray_cast(max_dist, hull, dist, ray_steps, hull_cube, off)

        # in this case the magnitude of the direc vector is 0 anyway
        max_dist[np.any(~np.isfinite(ray_steps), axis=1)] = 1.0

        rel_dist = dist / max_dist
        return max_dist, rel_dist

    def map_hull(self, hull_points):
        """
        Distances take already into account the anisotropy in z
        (i.e. they are true distances)
        But all coordinates for hulls and vectors are still pixel coordinates
        """
        self.hull_points = hull_points.astype(np.int16)
        hull_points = hull_points.astype(np.float32)

        kdt_skel = self.get_kdtree(self.all_nodes)
        dist_skel, ind_skel, coord_skel = self.get_knn(kdt_skel, hull_points)

        self.hull_skel['dist'] = dist_skel
        self.hull_skel['ind'] = ind_skel
        self.hull_skel['direc'] = coord_skel - hull_points ## NNs - Queries

        max_dist, rel_dist = self.calc_max_dist_to_skels()
        self.hull_skel['max_dist'] = max_dist
        self.hull_skel['rel_dist'] = rel_dist

        if len(self.branches):
            kdt_branch = self.get_kdtree(self.branches)
            dist_branch, ind_branch, coord_branch = self.get_knn(kdt_branch, hull_points)

            self.hull_branch['dist'] = dist_branch
            self.hull_branch['ind'] = ind_branch
            self.hull_branch['direc'] = coord_branch - hull_points

        else:
            self.hull_branch['dist'] = np.zeros(len(hull_points), dtype=np.float32)
            self.hull_branch['ind'] = None
            self.hull_branch['direc'] = np.zeros((len(hull_points),3), dtype=np.float32)

        if not np.all(np.isfinite(dist_skel)) or \
           not np.all(np.isfinite(self.hull_branch['dist'])):
             raise ValueError("InfiniteValue")

        self.kdt_hull = self.get_kdtree(self.hull_points, k=1, jobs=1) # store for later use

    @utils.cache()
    def get_hull_points_inner(self, cutoff=1.0/3, return_indices=False):
        mask = self.hull_skel['rel_dist'] < cutoff
        if return_indices:
            return self.hull_points[mask], mask.nonzero()[0]
        else:
            return self.hull_points[mask]

    @utils.cache()
    def get_hull_branch_direc_cutoff(self, cutoff=25, normalise=False):
        mask = self.hull_branch['dist'] < cutoff
        ret = self.hull_branch['direc'] * mask #[mask]
        if normalise:
            ret /= (self.hull_branch['dist'][:,None]+1e-5)
        return ret

    @utils.cache()
    def get_hull_branch_dist_cutoff(self, cutoff=25, normalise=True):
        mask = self.hull_branch['dist'] < cutoff
        ret = self.hull_branch['dist'] * mask #[mask]
        if normalise:
            ret = (ret > 0)
        return ret

    @utils.cache()
    def get_hull_skel_direc_rel(self):
        return self.hull_skel['direc'] / self.hull_skel['max_dist'][:,None]


    def plot_hull(self, fig=None):
        if fig is None:
            fig = mlab.figure(bgcolor=(1.0, 0.8, 0.4), size=(600,400))

        x = self.hull_points[:,0]
        y = self.hull_points[:,1]
        z = self.hull_points[:,2]*self.aniso_scale[0,2]
        mlab.points3d(x,y,z, scale_factor=1, color=(1,1,1), mode='cube',
                            opacity=0.1, figure=fig)
        return fig

    def plot_hull_inner(self, cutoff, fig=None):
        if fig is None:
            fig = mlab.figure(bgcolor=(1.0, 0.8, 0.4), size=(600,400))

        inner_hull = self.get_hull_points_inner(cutoff)
        x = inner_hull[:,0]
        y = inner_hull[:,1]
        z = inner_hull[:,2]*self.aniso_scale[0,2]
        mlab.points3d(x,y,z, scale_factor=1, color=(0.8,0.8,1), mode='cube',
                            opacity=0.1, figure=fig)
        return fig

    def plot_vec(self, substep=15, dict_name='skel', key='direc', vec=None, fig=None):
        if fig is None:
            fig = mlab.figure(bgcolor=(1.0, 0.8, 0.4), size=(600,400))

        x = self.hull_points[:,0]
        y = self.hull_points[:,1]
        z = self.hull_points[:,2]*self.aniso_scale[0,2]
        x, y, z = x[::substep], y[::substep], z[::substep]
        if vec is None:
            dict_ = self.hull_skel if dict_name=='skel' else self.hull_branch
            u = dict_[key][:,0]
            v = dict_[key][:,1]
            w = dict_[key][:,2]*self.aniso_scale[0,2]
        else:
            u = vec[:,0]
            v = vec[:,1]
            w = vec[:,2]*self.aniso_scale[0,2]

        u,v,w = u[::substep], v[::substep], w[::substep]
        mlab.quiver3d(x,y,z, u,v,w, figure=fig)
        return fig


class Trace(object):
    """
    Unless otherwise state all coordinates are in skeleton system (xyz) with
    z-axis anisotrope and all distances are in pixels (conversion to mu: 1/100)
    """
    def __init__(self, linked_skel=None, aniso_scale=2,max_cutoff=200,
                 uturn_detection_k=40, uturn_detection_thresh=0.45,
                 uturn_detection_hold=10, feature_count=7):

        self.aniso_scale = np.array([[1, 1, aniso_scale]], dtype=np.float32)
        self.skel = linked_skel
        self.lost_track = False
        self.uturn_occurred = False

        self.coords     = utils.AccumulationArray(right_shape=3, n_init=500)
        self.seg_length = utils.AccumulationArray(n_init=500)
        self.runlengths = utils.AccumulationArray(n_init=500)
        self.dist_self  = utils.AccumulationArray(right_shape=2, n_init=500)
        self.dist_skel  = utils.AccumulationArray(n_init=500)

        self.uturn_mask = utils.AccumulationArray(n_init=500, dtype=np.bool)
        self.coords_cnn = utils.AccumulationArray(right_shape=3, n_init=500)
        self.grads      = utils.AccumulationArray(right_shape=3, n_init=500)
        self.features   = utils.AccumulationArray(right_shape=feature_count, n_init=500)

        self.max_cutoff = max_cutoff
        self.uturn_detection_k = uturn_detection_k
        self.uturn_detection_thresh = uturn_detection_thresh
        self.uturn_detection_hold = uturn_detection_hold

        self.kdt = utils.DynamicKDT(k=uturn_detection_k, n_jobs=1,
                                    aniso_scale=self.aniso_scale)

        self.root = 0
        self.comment = ""

    def new_reverted_trace(self):
        new_trace = Trace(self.skel, self.aniso_scale[0,2],
                          self.max_cutoff, self.uturn_detection_k, self.uturn_detection_thresh,
                          self.uturn_detection_hold, self.features.data.shape[1:])

        new_trace.coords     = utils.AccumulationArray(data=self.coords[::-1])
        new_trace.seg_length = utils.AccumulationArray(data=self.seg_length[::-1])
        new_trace.runlengths = utils.AccumulationArray(data=self.runlengths[-1]-self.runlengths[::-1])
        new_trace.dist_self  = utils.AccumulationArray(data=self.dist_self[::-1])
        new_trace.dist_skel  = utils.AccumulationArray(data=self.dist_skel[::-1])
        new_trace.uturn_mask = utils.AccumulationArray(data=self.uturn_mask[::-1])
        new_trace.coords_cnn = utils.AccumulationArray(data=self.coords_cnn[::-1])
        new_trace.grads      = utils.AccumulationArray(data=self.grads[::-1])
        new_trace.features   = utils.AccumulationArray(data=self.features[::-1])
        if len(new_trace)<=self.uturn_detection_k:
            kdt = utils.DynamicKDT(k=self.uturn_detection_k, n_jobs=1,
                                   aniso_scale=self.aniso_scale)
            for c in new_trace.coords.data:
                kdt.append(c)

            new_trace.kdt = kdt
        else:
            new_trace.kdt = utils.DynamicKDT(points=new_trace.coords.data,
                                             k=self.uturn_detection_k,
                                             n_jobs=1,
                                             aniso_scale=self.aniso_scale)

        new_trace.root = len(self)-1
        try:
            self.comment
        except AttributeError:
            self.comment = ""

        new_trace.comment = self.comment+ " R"

        return new_trace


    def new_cut_trace(self, start, stop):
        new_trace = Trace(self.skel, self.aniso_scale[0,2],
                          self.max_cutoff, self.uturn_detection_k, self.uturn_detection_thresh,
                          self.uturn_detection_hold, self.features.data.shape[1:])

        new_trace.coords     = utils.AccumulationArray(data=self.coords[start:stop])
        new_trace.seg_length = utils.AccumulationArray(data=self.seg_length[start:stop])
        new_trace.runlengths = utils.AccumulationArray(data=self.runlengths[start:stop]-self.runlengths[start])
        new_trace.dist_self  = utils.AccumulationArray(data=self.dist_self[start:stop])
        new_trace.dist_skel  = utils.AccumulationArray(data=self.dist_skel[start:stop])
        new_trace.uturn_mask = utils.AccumulationArray(data=self.uturn_mask[start:stop])
        new_trace.coords_cnn = utils.AccumulationArray(data=self.coords_cnn[start:stop])
        new_trace.grads      = utils.AccumulationArray(data=self.grads[start:stop])
        new_trace.features   = utils.AccumulationArray(data=self.features[start:stop])
        if len(new_trace)<=self.uturn_detection_k:
            kdt = utils.DynamicKDT(k=self.uturn_detection_k, n_jobs=1,
                                   aniso_scale=self.aniso_scale)
            for c in new_trace.coords.data:
                kdt.append(c)

            new_trace.kdt = kdt
        else:
            new_trace.kdt = utils.DynamicKDT(points=new_trace.coords.data,
                                             k=self.uturn_detection_k,
                                             n_jobs=1,
                                             aniso_scale=self.aniso_scale)
        if (self.root - start) >= 0 and (self.root - start) < len(new_trace):
            new_trace.root = self.root - start
        else:
            new_trace.root = None #np.minimum(len(new_trace)-1, self.root - start)

        try:
            self.comment
        except AttributeError:
            self.comment = ""

        new_trace.comment = self.comment + "C%i-%i"%(start, stop)

        return new_trace


    def __len__(self):
        return len(self.coords)

    def save(self, fname):
        utils.picklesave(self, fname)

    def save_to_kzip(self, fname):
        trace_to_kzip(self, fname)

    def add_offset(self, off):
        off = np.atleast_2d(off)
        self.coords.add_offset(off)
        if len(self)<=self.uturn_detection_k:
            kdt = utils.DynamicKDT(k=self.uturn_detection_k, n_jobs=1,
                                   aniso_scale=self.aniso_scale)
            for c in self.coords.data:
                kdt.append(c)

            self.kdt = kdt
        else:
            self.kdt = utils.DynamicKDT(points=self.coords.data,
                                        k=self.uturn_detection_k, n_jobs=1,
                                        aniso_scale=self.aniso_scale)

    def append(self, coord, coord_cnn=None, grad=None, features=None):
        self.coords.append(coord)
        if len(self)>1:
            diff = np.linalg.norm((coord - self.coords[-2]) * self.aniso_scale[0])
        else:
            diff = 5 # just guess

        self.seg_length.append(diff)
        self.runlengths.append(self.runlength)

        if len(self) > self.uturn_detection_k+1:
            distances, indices, coordinates = self.kdt.get_knn(coord, k=self.uturn_detection_k)
            dist = distances.mean()
        else:
            dist = self.seg_length.ema * float(self.uturn_detection_k + 1) / 2

        normalisation = self.seg_length.ema * float(self.uturn_detection_k + 1) / 2
        self.dist_self.append([dist, dist/normalisation])
        self.kdt.append(coord)

        if self.skel:
            dist, index, node = self.skel.get_closest_node(coord)
            self.dist_skel.append(dist)

        if grad is not None:
            self.grads.append(grad)

        if features is not None:
            self.features.append(features)

        if coord_cnn is not None:
            self.coords_cnn.append(coord_cnn)

        # Check for criteria
        last_dist = self.dist_self[-self.uturn_detection_hold:, 1]
        uturn = np.all(last_dist < self.uturn_detection_thresh)
        self.uturn_mask.append(uturn)
        if not self.uturn_occurred and uturn: # register the first u-turn
            self.uturn_occurred = (len(self), self.runlength)

        if not self.lost_track:
            lost = self.dist_skel.max() > self.max_cutoff
            if lost:
                self.lost_track = (len(self), self.runlength)

    def append_serial(self, *args):
            for arg in zip(*args):
                self.append(*arg)

    @property
    def avg_seg_length(self):
        return self.seg_length.mean()

    @property
    def runlength(self):
        return self.seg_length.sum()

    @property
    def avg_dist_skel(self):
        return self.dist_skel.mean()

    @property
    def max_dist_skel(self):
        return self.dist_skel.max()

    @property
    def avg_dist_self(self):
        return self.dist_self.mean()

    @property
    def min_dist_self(self):
        return self.dist_self.min()[0]

    @property
    def min_normed_dist_self(self):
        return self.dist_self.min()[1]

    def tortuosity(self, start=None, end=None):
        if start is None:
            start = 0
        if end is None:
            end = len(self)

        arc = self.runlengths[end-1] - self.runlengths[start]
        chord = np.linalg.norm((self.coords[end-1] - self.coords[start]) * self.aniso_scale[0])
        t = arc / chord
        return t

    def plot(self, grads=True, skel=True, rand_color=False, fig=None):
        if fig is None:
            fig = mlab.figure(bgcolor=(1.0, 0.8, 0.4), size=(600,400))
        if skel and self.skel:
            fig = self.skel.plot_skel(fig=fig)

        x = self.coords[:, 0]
        y = self.coords[:, 1]
        z = self.coords[:, 2] * self.aniso_scale[0, 2]

        line_c = tuple(np.random.rand(3)) if rand_color else (0, 0, 0.7)
        point_c = line_c if rand_color else (0.6, 0.7, 0.9)

        mlab.plot3d(x, y, z, tube_radius=0.2, color=line_c, figure=fig)
        mlab.points3d(x, y, z, scale_factor=0.8,  color=point_c, figure=fig)

        if grads and self.grads.length:
            x = self.coords[:, 0]
            y = self.coords[:, 1]
            z = self.coords[:, 2] * self.aniso_scale[0, 2]
            gx = -self.grads[:, 0]
            gy = -self.grads[:, 1]
            gz = -self.grads[:, 2] * self.aniso_scale[0, 2]
            mlab.quiver3d(x, y, z, gx, gy, gz, figure=fig,
                          color=(0, 0.6, 0.2), scale_factor=3)

        return fig

    def split_uturns(self, return_accum_pathlength=False, print_stat=False):
        transitions = np.diff(self.uturn_mask, axis=0, n=1)
        transitions = np.nonzero(transitions)[0]
        transitions[0::2] -= self.uturn_detection_hold # if add: end segment closer to uturn
        transitions[1::2] -= self.uturn_detection_hold # if subtract: start new segment closer to uturn
        transitions = np.minimum(np.maximum(0, transitions), len(self))
        transitions = np.hstack((0, transitions, len(self)))

        new_traces = []
        accum_pathlenghts = []
        accum_dist_skel = []
        accum_runlength = 0

        for i in range(0, len(transitions)-1, 2):
            new = self.__class__(self.skel, self.aniso_scale[0, 2], self.max_cutoff,
                 self.uturn_detection_k, self.uturn_detection_thresh,
                                 self.uturn_detection_hold)
            start, stop = transitions[i], transitions[i+1]
            if print_stat:
                print("cutting between %i and %i " % (start, stop))
            if start<stop: # some transitions are too short
                coords = self.coords[start:stop]
                new.append_serial(coords)
                new_traces.append(new)

                # accumulate pathlenghts and dist to skel over splits
                runlengths = self.runlengths[start:stop]
                runlengths = runlengths + accum_runlength - runlengths[0] # shift

                accum_pathlenghts.append(runlengths)
                accum_runlength = runlengths[-1]

                if self.skel:
                    dist_skel = self.dist_skel[start:stop].copy()
                    if i > 0: # this makes the eval stop if the trace deviated from the
                        # skeleton too much during the uturn
                        max_dist_in_uturn = np.max(self.dist_skel[transitions[i-1]:start])
                        dist_skel[0] = np.maximum(dist_skel[0], max_dist_in_uturn)

                    accum_dist_skel.append(dist_skel)
                else:
                    accum_dist_skel.append([])

        if return_accum_pathlength:
            return new_traces, np.hstack(accum_pathlenghts), np.hstack(accum_dist_skel)
        else:
            return new_traces


def normalised_min_dist(tr, point):
    dist, ind, coord = tr.kdt.get_knn(point, k=1)
    radius = tr.features[ind, 0]
    return dist, dist / radius

def simple_stats(a):
    m = np.mean(a, axis=0)
    s = np.std(a, axis=0)
    minv = np.min(a, axis=0)
    maxv = np.max(a, axis=0)
    return np.array([m, s, minv, maxv])

def radius_hist(r):
    bins = np.array([0,8,14,23,35,50,80,200])
    counts, bins = np.histogram(r, bins=bins, density=True)
    return counts



def get_merge_features(main_tr, main_node, sub_tr, sub_node, end_match):
    m_slice_small = slice(max(0,main_node-5), main_node+5)
    m_slice_large = slice(max(0,main_node-25), main_node+25)
    m_points = main_tr.coords[m_slice_small] * main_tr.aniso_scale
    uu, pc_m, pc_dir_m = np.linalg.svd(m_points-m_points.mean(0))
    m_feat_small = simple_stats(main_tr.features[m_slice_small])
    m_feat_large = simple_stats(main_tr.features[m_slice_large])
    m_radius_hist= radius_hist(main_tr.features[m_slice_large, 0])
    m_tortuosity = main_tr.tortuosity()
    main_features= np.hstack([pc_m, pc_dir_m.ravel(), m_feat_small.ravel(),
                              m_feat_large.ravel(), m_radius_hist, m_tortuosity])

    if end_match:
        s_slice_small = slice(max(0,sub_node-10), sub_node)
        s_slice_large = slice(max(0,sub_node-50), sub_node)
    else:
        s_slice_small = slice(sub_node, sub_node+10)
        s_slice_large = slice(sub_node, sub_node+50)

    s_points = sub_tr.coords[s_slice_small] * sub_tr.aniso_scale
    if len(s_points)==0:
        pass
    uu, pc_s, pc_dir_s = np.linalg.svd(s_points-s_points.mean(0))
    s_feat_small = simple_stats(sub_tr.features[s_slice_small])
    s_feat_large = simple_stats(sub_tr.features[s_slice_large])
    s_radius_hist= radius_hist(sub_tr.features[s_slice_large, 0])
    s_tortuosity = sub_tr.tortuosity()
    sub_features = np.hstack([pc_s, pc_dir_s.ravel(), s_feat_small.ravel(),
                              s_feat_large.ravel(), s_radius_hist, s_tortuosity])

    dist = np.linalg.norm((main_tr.coords[main_node]-sub_tr.coords[sub_node])*sub_tr.aniso_scale)
    r_m = main_tr.features[main_node, 0]
    r_s = sub_tr.features[sub_node, 0]
    pc_dir_similarity = np.abs(np.dot(pc_dir_m, pc_dir_s.T))
    joint_features = np.hstack([dist, dist/r_m, dist/r_s,
                                2*dist/(r_m+r_s), pc_dir_similarity.ravel()])

    return main_features, sub_features, joint_features

def split_tree_components(tracetree, cut=False):
    if tracetree.num_components==1:
        return [tracetree,]

    new_trees = [list() for i in range(tracetree.num_components)]
    for tr_i in tracetree.traces:
        c = tracetree.tr_i2comp_i[tr_i]
        tr = tracetree.traces[tr_i]
        cuts = tracetree.trace_cuts.get(tr_i, None)
        if cuts and cut:
            tr = tr.new_cut_trace(*cuts)

        new_trees[c].append(tr)

    for i in range(tracetree.num_components):
        new_tree = TraceTree(new_trees[i],
                           tracetree.spine_thresh,
                           tracetree.endpoint_thresh)

        new_trees[i] = new_tree

    return new_trees



class TraceTree(object):
    def __init__(self, traces, spine_thresh=1.5, endpoint_thresh=0.8):
        """
        :param traces:
        :param spine_thresh:
        """""":param spine_thresh: float
            How large the maximal relative distance needs to be for a loop to be
            retained as a spine branch
        :param endpoint_thresh: float
            Threshold of relative distance between endpoint and other trace tp
            count as a connection
        """
        # Rename trace keys to smaller contiguous numbers
        if not isinstance(traces, dict):
            traces = dict(zip(range(len(traces)), traces))

        self.traces = traces
        self.trace_cuts = dict()
        self.pruned_traces = []
        self.edge_candidates = dict()
        self.edges = []
        self.tr_i2comp_i = None
        self.num_components = 1
        self.aniso = np.array([[1,1,2]])

        self.spine_thresh = spine_thresh
        self.endpoint_thresh = endpoint_thresh
        self.joined_kdt = None
        self.joined_coords = None
        self.joined_radii = None

    def build_joined_features(self):
        self.joined_coords = np.vstack([tr.coords for tr in self.traces.values()])
        self.joined_radii = np.hstack([tr.features[:,0] for tr in self.traces.values()])
        kdt = utils.DynamicKDT(self.joined_coords, n_jobs=-1,
                               aniso_scale=[1, 1, 2], k=1)
        self.joined_kdt = kdt

    def cut_traces_inplace(self):
        for tr_i in self.traces:
            tr = self.traces[tr_i]
            cuts = self.trace_cuts.get(tr_i, None)
            if cuts:
                new_tr = tr.new_cut_trace(*cuts)
                self.traces[tr_i] = new_tr

    def to_kzip(self, fname, save_loops=False, save_edge_candiates=False,
                add_edges=False, save_edges=False):
        fname = os.path.expanduser(fname)
        fpath, comment_name = os.path.split(fname)
        skel_objs       = []
        component_annos = []
        for c in range(self.num_components):
            skel_obj = knossos_skeleton.Skeleton()
            skel_objs.append(skel_obj)
            anno_ = knossos_skeleton.SkeletonAnnotation()
            anno_.scaling = (9.0, 9.0, 20.0)
            anno_.setComment(comment_name+"-c%i"%c)
            skel_obj.add_annotation(anno_)
            component_annos.append(anno_)

        # Save all cut traces to own anno-obj of their component
        node_mappings = dict()
        for tr_i in self.traces:
            if self.tr_i2comp_i is not None:
                c = self.tr_i2comp_i[tr_i]
            else:
                c = 0

            anno = component_annos[c]
            tr = self.traces[tr_i]
            cuts = self.trace_cuts.get(tr_i, None)
            if cuts:
                tr = tr.new_cut_trace(*cuts)

            _, node_mapping = trace_to_anno(tr, fname, anno)
            node_mappings[tr_i] = node_mapping


        # Save all edges (between cut points) to a edge annotation
        if len(self.edges) and save_edges:
            edge_anno = knossos_skeleton.SkeletonAnnotation()
            edge_anno.scaling = (9.0, 9.0, 20.0)
            edge_anno.setComment("Edges-"+comment_name)
            skel_obj_edges = knossos_skeleton.Skeleton()
            skel_obj_edges.add_annotation(edge_anno)

            for e in self.edges:
                try:
                    main, sub = e
                    main_node_i, sub_node_i = self.edge_candidates[tuple(e)][1]
                except KeyError: # MST might turn around edge order
                    main_node_i, sub_node_i = self.edge_candidates[tuple(e)[::-1]][1]
                    main, sub = e[::-1]

                main_node = knossos_skeleton.SkeletonNode()
                x,y,z = np.round(self.traces[main].coords[main_node_i]).astype(np.int16)
                main_node.from_scratch(edge_anno, x,y,z)
                edge_anno.addNode(main_node)

                sub_node = knossos_skeleton.SkeletonNode()
                x,y,z = np.round(self.traces[sub].coords[sub_node_i]).astype(np.int16)
                sub_node.from_scratch(edge_anno, x,y,z)
                edge_anno.addNode(sub_node)
                edge_anno.addEdge(main_node, sub_node)

                if add_edges:
                    main_cut = self.trace_cuts.get(main, [0, None])[0]
                    sub_cut  = self.trace_cuts.get(sub, [0, None])[0]
                    main_i = main_node_i - main_cut
                    sub_i  = sub_node_i  - sub_cut
                    try:
                        n_main = node_mappings[main][main_i]
                        n_sub  = node_mappings[sub][sub_i]
                        n_main.annotation.addEdge(n_main, n_sub)
                    except:
                        pass

            outfile = fpath + "/edges-" + comment_name + '.k.zip'
            skel_obj_edges.to_kzip(outfile)

        # As Node==Edge==Node in one skeleton Tree (for making GT in knossos)
        if save_edge_candiates and self.edge_candidates:
            edge_candiate_anno = knossos_skeleton.SkeletonAnnotation()
            edge_candiate_anno.scaling = (9.0, 9.0, 20.0)
            edge_candiate_anno.setComment(comment_name+"-Edge-Candiates")
            skel_obj_candidates = knossos_skeleton.Skeleton()
            skel_obj_candidates.add_annotation(edge_candiate_anno)
            for e in self.edge_candidates:
                 main, sub = e
                 main_node_i, sub_node_i = self.edge_candidates[e][1]

                 main_node = knossos_skeleton.SkeletonNode()
                 x,y,z = np.round(self.traces[main].coords[main_node_i]).astype(np.int16)
                 main_node.from_scratch(edge_candiate_anno, x,y,z)
                 main_node.setComment(comment_name+"-M%i_%i-S%i_%i-main"
                 %(main, main_node_i, sub, sub_node_i))
                 edge_candiate_anno.addNode(main_node)

                 sub_node = knossos_skeleton.SkeletonNode()
                 x,y,z = np.round(self.traces[sub].coords[sub_node_i]).astype(np.int16)
                 sub_node.from_scratch(edge_candiate_anno, x,y,z)
                 sub_node.setComment(comment_name+"-M%i_%i-S%i_%i-sub"
                 %(main, main_node_i, sub, sub_node_i))
                 edge_candiate_anno.addNode(sub_node)
                 edge_candiate_anno.addEdge(main_node, sub_node)

            outfile = fpath + "/candidates-" + comment_name + '.k.zip'
            skel_obj_candidates.to_kzip(outfile)

        if save_loops:
            skel_obj_loops = knossos_skeleton.Skeleton()
            for i,t in enumerate(self.pruned_traces):
                anno_loop, _ = trace_to_anno(t, comment_name+'-loop%i'%i)
                skel_obj_loops.add_annotation(anno_loop)

            outfile = fname + '-loops.k.zip'
            skel_obj_loops.to_kzip(outfile)

        for i,skel_obj in enumerate(skel_objs):
            outfile = fname + '-c%i.k.zip' %i
            skel_obj.to_kzip(outfile)

    def is_loop(self, trace, traces):
        """
        :param trace:  Trace
            test candidate
        :param traces: list of Traces
        :return: bool
        """
        if not len(traces):
            return False

        # Determine the average distance of the two end points to all other
        # traces. Use the distances normalised by the radius of the other trace
        end_points = trace.coords[[0,-1]]
        relative_distances = np.ones(len(traces)) * np.inf
        for i,tr in enumerate(traces.values()):
            if tr == trace: # don't compare trace to itself, would be loop always
                continue

            dist, ind, coord = tr.kdt.get_knn(end_points, k=1)
            radii = tr.features[ind, 0]
            relative_distances[i] = (dist/radii).mean()

        k = relative_distances.argmin()
        tr_i = traces.keys()[k]
        if relative_distances[k] < self.endpoint_thresh:
            #now check if there exists point that is farther away from main
            dist, rel_dist = normalised_min_dist(traces[tr_i], trace.coords.data)
            max_point = rel_dist.argmax()
            if rel_dist[max_point] >= self.spine_thresh :
                is_loop = False
                cut_a = 0 if rel_dist[0] < rel_dist[-1] else len(rel_dist)
                cut_b = max_point
                cut_0 = min(cut_a, cut_b)
                cut_1 = max(cut_a, cut_b)
                assert cut_0 < cut_1
                cuts = (cut_0, cut_1)

            else:
                is_loop = True
                cuts = None
        else:
            is_loop = False
            cuts = None

        return is_loop, cuts

    def closest_approach(self, tr_a, tr_b):
        """
        :param tr_a: Trace
        :param tr_b: Trace
        :return:
        """
        b0toa = normalised_min_dist(tr_a, tr_b.coords[0])[1]
        b1toa = normalised_min_dist(tr_a, tr_b.coords[-1])[1]
        a0tob = normalised_min_dist(tr_b, tr_a.coords[0])[1]
        a1tob = normalised_min_dist(tr_b, tr_a.coords[-1])[1]

        case = np.argmin([b0toa, b1toa, a0tob, a1tob])
        geometrict_dist = np.min([b0toa, b1toa, a0tob, a1tob])
        if geometrict_dist > self.spine_thresh :
            return None

        end_match = case in [1,3]
        a_is_main = case in [0, 1]
        main_tr = tr_a if case in [0,1] else tr_b
        sub_tr  = tr_b if case in [0,1] else tr_a
        slice_20 = slice(-20, None) if end_match else slice(None, 20)

        sub_coords = sub_tr.coords[slice_20]
        distances, indices, coordinates = main_tr.kdt.get_knn(sub_coords, k=1)
        max_seg_length = sub_tr.seg_length[slice_20].max()
        sub_merge_candidates = (distances < max_seg_length).nonzero()[0]
        if len(sub_merge_candidates):
            sub_node = sub_merge_candidates[0] if end_match else sub_merge_candidates[-1]
        else:
            sub_node = distances.argmin()

        main_node = indices[sub_node] # Take the main node which was found in knn
        if end_match: # For end_match the index needs to be shifted by the trace length to comply with the indices of sub_coords
            sub_node += len(sub_tr) - np.minimum(20, len(sub_tr))

        cut_start = 0 if end_match else sub_node
        cut_end   = sub_node+1 if end_match else len(sub_tr)
        assert cut_end-cut_start>0

        # Check for cases where there is a spine loop
        if (cut_start==sub_node and end_match):
            assert cut_start==0
            cut_end = len(sub_tr)
            end_match = not end_match

        elif (cut_end==sub_node and not end_match):
            assert cut_end==len(sub_tr)
            cut_start = 0
            end_match = not end_match


        cuts = (cut_start, cut_end)
        nodes = (main_node, sub_node)
        try:
            feat = get_merge_features(main_tr,main_node,sub_tr,sub_node, end_match)
        except:
            pass
        # merge_feat_main, merge_feat_sub, merge_feat_joint = feat
        return geometrict_dist, a_is_main, nodes, cuts, feat


    def make_merge_graph(self):
        n = len(self.traces)
        msd_list= []
        keys = self.traces.keys()
        # For all pairwise traces find closest approach, node/cuts indices and
        # features for the edge classifier, collect positive edges in "msd_list"
        for s in range(n):
            for t in range(s+1, n):
                tr_a = keys[s]
                tr_b = keys[t]
                tmp = self.closest_approach(self.traces[tr_a], self.traces[tr_b])
                if tmp is None: # If Components are disconnected still add them
                    msd_list.append([tr_a, tr_a, 0])
                    msd_list.append([tr_b, tr_b, 0])
                    continue

                geometrict_dist, a_is_main, nodes, cuts, feat = tmp
                if np.isclose(geometrict_dist, 0.0):
                    geometrict_dist = 0.1
                    # otherwise connected components will consider this as split

                if a_is_main:
                    main = tr_a
                    sub  = tr_b
                else:
                    main = tr_b
                    sub  = tr_a

                main_coord = self.traces[main].coords[nodes[0]]
                sub_coord  = self.traces[sub].coords[nodes[1]]
                coords = (main_coord, sub_coord)
                # if edge_classifier(feat) > thresh:
                    # don't add edges which are not classified
                self.edge_candidates[(main, sub)] = [geometrict_dist, nodes, cuts, feat, coords]
                msd_list.append([main, sub, geometrict_dist])

        if len(msd_list)==0:
            return

        # Create MST from edge graph (actually MST-Forest)
        main, sub, dist = np.array(msd_list).T
        a = np.hstack([main, sub])
        b = np.hstack([sub, main])
        values   = np.hstack([dist, dist])
        adj_mat  = sparse.csr_matrix(sparse.coo_matrix( (values, (a,b)) ))
        mst = csgraph.minimum_spanning_tree(adj_mat)
        edges_mst = np.array(mst.nonzero()).T
        self.edges = edges_mst

        # For all MST-edges update the cuts (cut as much as possible to cover merge positions)
        for edge_mst in edges_mst:
            try:
                sub = edge_mst[1]
                tmp = self.edge_candidates[tuple(edge_mst)]
            except KeyError: # MST might turn around edge order
                sub = edge_mst[0]
                tmp = self.edge_candidates[tuple(edge_mst)[::-1]]

            old_cuts = self.trace_cuts.get(sub, None)
            if old_cuts is None:
                new_cuts = tmp[2]
            else:
                new_cuts = (np.minimum(old_cuts[0], tmp[2][0]),
                            np.maximum(old_cuts[1], tmp[2][1]))

            self.trace_cuts[sub] = new_cuts

        # If edges candidates were classified negative, the components oft the
        # MST-forest must be split
        # Returns unconnected nodes (empty slots) as component too!
        num, labels = csgraph.connected_components(mst, directed=False)
        # Therefore select only the stuff which is in keys
        comp_names, components = np.unique(labels[keys], return_inverse=True)
        self.tr_i2comp_i = dict(zip(keys, components))
        self.num_components = comp_names.size




    def simplify(self, profile=False):
        if profile:
            tt = utils.Timer()

        keep_traces  = {}
        pruned_traces = {}
        traces = dict(self.traces)

        keys = np.array(traces.keys())
        trace_lengths = np.array([traces[tr_i].runlength for tr_i  in keys])
        keys_sorted = keys[np.argsort(trace_lengths)]

        for tr_i in keys_sorted:
            tr = traces[tr_i]
            is_loop, cuts = self.is_loop(tr, traces)
            if is_loop:
                pruned_traces[tr_i] = traces.pop(tr_i)

            else: # Don't cut traces here, it might mess up connection parts
                #if cuts: # cut traces must be put to stack again because they might be a loop now
                #    cut_tr = tr.new_cut_trace(*cuts)
                #    traces[tr_i] = cut_tr
                #else:
                keep_traces[tr_i] = tr

        self.pruned_traces = pruned_traces
        self.traces = keep_traces
        if profile:
            tt.check(name='prune loops')

        # If all traces are loops with another trace take the largest single trace
        if len(self.traces)==0:
            tr_lengths = np.array([(tr_i, tr.runlength) for tr_i,tr in pruned_traces.items()])
            i = np.argmax(tr_lengths[:,1])
            i = int(tr_lengths[i,0])
            tr0 = pruned_traces.pop(i)
            self.traces[i] = tr0

        self.make_merge_graph()
        if profile:
            tt.check("merge graph")











def make_segment_lenghts(bone):
    segment_lengths = np.linalg.norm(np.diff(bone, n=1, axis=0) * np.array([[1, 1, 2]]), axis=1)
    segment_lengths = np.hstack(([0, ], segment_lengths))
    segment_lengths[0] = segment_lengths[1] * 0.5
    segment_lengths[1:-2] = (segment_lengths[1:-2] + segment_lengths[
                                                     2:-1]) * 0.5
    segment_lengths[-1] = segment_lengths[-1] * 0.5
    return segment_lengths


def runlength_metric(path_lengths, distances, cut_start=10, cut_max=200, num=50):
    cutoffs         = np.linspace(cut_start, cut_max, num=num)
    runlengths      = utils.AccumulationArray()
    correct_lenghts = utils.AccumulationArray()
    for cutoff in cutoffs:
        for i in range(len(distances)):
            larger = np.nonzero(distances[i] >= cutoff)[0]
            if len(larger):
                correct_lenghts.append(path_lengths[i][larger[0]-1])
            else:
                correct_lenghts.append(path_lengths[i][-1])

        mean_correct_lenght = correct_lenghts.mean()
        correct_lenghts.clear()
        runlengths.append(mean_correct_lenght)

    return runlengths.data, cutoffs


def runlength_metric_GT(trace, skel=None, cut_start=10, cut_max=200, num=20):
    if skel is None:
        skel = trace.skel

    cutoffs = np.linspace(cut_start, cut_max, num=num)
    runlengths = np.zeros(num)
    trace_points = trace.coords.data
    trace_kdt = utils.KDT(radius=cut_max, n_jobs=-1)
    trace_kdt.fit(trace_points * np.array([[1,1,2]]))

    for edge, bone  in skel.bones.items():
        segment_lengths = make_segment_lenghts(bone)
        dist, ind = trace_kdt.radius_neighbors(bone * np.array([[1,1,2]]))
        for i, cutoff in enumerate(cutoffs):
            was_traced = np.zeros(len(bone), dtype=np.bool)
            for k in range(len(dist)):
                was_traced[k] = np.any(dist[k]<=cutoff)

            runlengths[i] += segment_lengths[was_traced].sum()

    return cutoffs, runlengths



def trace_to_anno(trace_xyz, name, anno=None, root=None):
    if isinstance(trace_xyz, Trace):
        feature_avail = len(trace_xyz.features)==len(trace_xyz)
    else:
        feature_avail = True
    radius = 1.0
    if anno is None:
        anno = knossos_skeleton.SkeletonAnnotation()
        anno.scaling = (9.0, 9.0 ,20.0)
        anno.setComment(os.path.split(name)[1])

    node_mapping = dict()

    last_node = knossos_skeleton.SkeletonNode()
    trace_coords = np.round(trace_xyz.coords).astype(np.int16) # trace_xyz.coords.data.astype(np.int16)
    if feature_avail:
        radius = trace_xyz.features[0, 0]
    last_node.from_scratch(anno, trace_coords[0,0],
                           trace_coords[0,1], trace_coords[0,2], radius=radius)
    if feature_avail:
        last_node.setDataElem("axoness_proba0", trace_xyz.features[0, 1])
        last_node.setDataElem("axoness_proba1", trace_xyz.features[0, 2])
        last_node.setDataElem("axoness_proba2", trace_xyz.features[0, 3])
        last_node.setDataElem("spiness_proba0", trace_xyz.features[0, 4])
        last_node.setDataElem("spiness_proba1", trace_xyz.features[0, 5])
        last_node.setDataElem("spiness_proba2", trace_xyz.features[0, 6])


    anno.addNode(last_node)
    node_mapping[0] = last_node
    for k in range(1, len(trace_coords)):
        coord = trace_coords[k]
        if feature_avail:
            radius = trace_xyz.features[k,0]

        new_node = knossos_skeleton.SkeletonNode()
        new_node.from_scratch(anno, coord[0], coord[1], coord[2], radius=radius)
        if feature_avail:
            last_node.setDataElem("axoness_proba0", trace_xyz.features[k, 1])
            last_node.setDataElem("axoness_proba1", trace_xyz.features[k, 2])
            last_node.setDataElem("axoness_proba2", trace_xyz.features[k, 3])
            last_node.setDataElem("spiness_proba0", trace_xyz.features[k, 4])
            last_node.setDataElem("spiness_proba1", trace_xyz.features[k, 5])
            last_node.setDataElem("spiness_proba2", trace_xyz.features[k, 6])

        node_mapping[k] = new_node
        anno.addNode(new_node)
        last_node.addChild(new_node)
        last_node = new_node

    if root is None:
        if isinstance(trace_xyz, Trace) and trace_xyz.root is not None:
            root = trace_xyz.root
    if root is not None:
        node_mapping[root].setRoot()

    return anno, node_mapping


def trace_to_kzip(trace_xyz, fname):
    skel_obj = knossos_skeleton.Skeleton()
    anno, node_mapping = trace_to_anno(trace_xyz, fname)
    skel_obj.add_annotation(anno)
    outfile = fname + '.k.zip'
    skel_obj.to_kzip(outfile)


def trace_to_kzip_multi(traces, fname):
    if isinstance(traces, dict):
        traces = traces.values()
    skel_obj = knossos_skeleton.Skeleton()
    for i, trace_xyz in enumerate(traces):
        if not isinstance(trace_xyz, np.ndarray):
            anno, node_mapping = trace_to_anno(trace_xyz, fname+"_%i"%i)
        else:
            trace_xyz = np.round(trace_xyz).astype(np.int16)
            anno = knossos_skeleton.SkeletonAnnotation()
            anno.scaling = (9.0, 9.0 ,20.0)
            anno.setComment(os.path.split(fname)[1]+"_%i"%i)
            last_node = knossos_skeleton.SkeletonNode()
            last_node.from_scratch(anno, trace_xyz[0,0], trace_xyz[0,1], trace_xyz[0,2])
            last_node.setRoot()
            anno.addNode(last_node)
            for coord in trace_xyz[1:]:
                new_node = knossos_skeleton.SkeletonNode()
                new_node.from_scratch(anno, coord[0], coord[1], coord[2])
                anno.addNode(new_node)
                last_node.addChild(new_node)
                last_node = new_node

        skel_obj.add_annotation(anno)

    outfile = fname + '.k.zip'
    skel_obj.to_kzip(os.path.expanduser(outfile))


def bbox_cube_anno(off_xyz, sz_xyz, comment="?", cross_edges=False):
    off_xyz = np.array(off_xyz)
    sz_xyz = np.array(sz_xyz)
    cords = [off_xyz+sz_xyz*[0,0,0],
             off_xyz+sz_xyz*[1,0,0],
             off_xyz+sz_xyz*[1,1,0],#2
             off_xyz+sz_xyz*[0,1,0],
             off_xyz+sz_xyz*[0,0,1],#4
             off_xyz+sz_xyz*[1,0,1],
             off_xyz+sz_xyz*[1,1,1],#6
             off_xyz+sz_xyz*[0,1,1],
            ]

    cords = np.array(cords)

    anno = knossos_skeleton.SkeletonAnnotation()
    anno.scaling = (9.0, 9.0 ,20.0)
    anno.setComment("%s: %s - %s"%(comment, off_xyz,sz_xyz))
    nodes = []
    for x,y,z in cords:
        new_node = knossos_skeleton.SkeletonNode()
        new_node.from_scratch(anno,x,y,z)
        anno.addNode(new_node)
        nodes.append(new_node)

    edges = [(0,1),(0,3),(0,4), (1,2),(1,5), (3,2),(3,7), (4,5),(4,7), (6,7),(6,5),(6,2)]
    for n1, n2 in edges:
        nodes[n1].addChild(nodes[n2])

    if cross_edges:
        for n1 in anno.nodes:
           for n2 in anno.nodes:
               n1.addChild(n2)

    return anno
