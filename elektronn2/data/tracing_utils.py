# -*- coding: utf-8 -*-
# ELEKTRONN2 Toolkit
# Copyright (c) 2015 Marius Killinger
# All rights reserved
from __future__ import absolute_import, division, print_function
# TODO: Python 3 compatibility

__all__ = ['Tracer', 'CubeShape', 'ShotgunRegistry', 'make_shotgun_data_z']

import os
import time
import sys
from subprocess import check_call
import numpy as np
import tqdm
import logging
import scipy.ndimage as ndimage

from elektronn2 import utils
from elektronn2.utils.utils_basic import unique_rows
from elektronn2.data import knossos_array, transformations
from elektronn2.data.skeleton import Trace

logger = logging.getLogger('elektronn2log')

if sys.version_info[:2] != (2, 7):
    raise ImportError(
        '\nSorry, this module only supports Python 2.7.'
        '\nYour current Python version is {}\n'.format(sys.version)
    )

try:
    from knossos_utils import skeleton as knossos_skeleton
    from knossos_utils.knossosdataset import KnossosDataset
except ImportError as e:
    logger.error('\nFor using the tracing_utils module, you will need to'
                 ' install the knossos_utils module'
                 ' (https://github.com/knossos-project/knossos_utils)\n')
    raise e


with open(os.devnull, 'w') as devnull:
    # mayavi is to dumb to raise an exception and instead crashes whole script....
    try:
        # "xset q" will always succeed to run if an X server is currently running
        check_call(['xset', 'q'], stdout=devnull, stderr=devnull)
        import mayavi.mlab as mlab
        from tvtk.util.ctf import ColorTransferFunction, PiecewiseFunction
        # Don't set backend explicitly, use system default...
    # if "xset q" fails, conclude that X is not running
    except: # (OSError, ImportError, CalledProcessError, ValueError)
        logger.warning("No mayavi imported, cannot plot skeletons")
        mlab = None


"""
Modus
m: memory feedback
r: radius scaling
k: maximum kernel interpolation
s: restrict scaling to >0.5

"""

class Tracer(object):
    def __init__(self, model, z_shift=0, data_source=None, bounding_box_zyx=None,
                 trace_kwargs=dict(aniso_scale=2), modus='m', shotgun_registry=None,
                 registry_interval=None, reference_radius=18.0):
        if isinstance(data_source, (knossos_array.KnossosArray, knossos_array.KnossosArrayMulti)):
            assert model.input_node.shape['f']==data_source.shape[0]
        elif isinstance(data_source, np.ndarray):
            assert model.input_node.shape['f']==data_source.shape[0]
        else:
            raise ValueError("Wrong Data source")
        if isinstance(model, str):
            raise ValueError("must pass a model")
            #model = neuromancer.model.modelload(model)

        self.rnn = False
        self.modus = modus
        if 'scan' in model.nodes:
            self.rnn = True
            self.n_steps = model['scan'].n_steps
            self.mem_hid0 = model['mem_hid'].get_value()
            print("Initial State mean=%.3f, std=%.3f" %(self.mem_hid0.mean(), self.mem_hid0.std()))
            self.mem_trace0 = np.zeros((1, 3*self.n_steps), dtype=np.float32)

        self.model = model
        self.z_shift = z_shift
        self.patch_size = np.array(model.input_node.shape.spatial_shape)
        self.data = data_source
        self.trace_kwargs = trace_kwargs
        self.bounding_box_zyx = np.array(bounding_box_zyx)
        self.model.predict_ext() # compile function
        self.shotgun_registry = shotgun_registry
        self.registry_interval = registry_interval
        self.reference_radius = reference_radius

    @staticmethod
    def zeropad(a, length):
        n = len(a)
        if n < length:
            return np.pad(a, [[length-n,0],[0,0]], mode='constant')
        else:
            return a

    def get_scale_factor(self, radius, old_factor, scale_strenght):
        # if old was large (zoom in), radius is smaller
        radius_true = radius * 1.0/old_factor
        # Equilibrium Condition scale_factor=1 <==> radius = 20
        new_factor = self.reference_radius / radius_true
        hi = scale_strenght * 2.0 + (1-scale_strenght) * 1.0 ### Original 1.5
        if 's' in self.modus:
            lo = scale_strenght * 0.5 + (1-scale_strenght) * 1.0 ### Original 0.3
        else:
            lo = scale_strenght * 0.3 + (1 - scale_strenght) * 1.0  ### Original 0.3

        new_factor = np.clip(new_factor, lo, hi)
        change = new_factor / old_factor
        change = np.clip(change, 0.75, 1.25)
        new_factor = old_factor * change

        return new_factor

    def trace(self, position_l, direction_il, count, gamma=0, trace_xyz=None,
              linked_skel=None, check_for_lost_track=True,
              check_for_uturn=False, check_bb=True, profile=False,
              info_str=None, reject_obb_traces=False, initial_scale=None):
        """
        Although psoition_l is in zyx order, the returned trace_obj is in xyz order
        """
        tt = utils.Timer()

        if trace_xyz is None:
            trace_xyz = Trace(linked_skel=linked_skel, **self.trace_kwargs)
            if self.shotgun_registry:
                self.shotgun_registry.new_trace(trace_xyz)

            prediction_c_dummy = np.array([0, 0, 0])
            trace_xyz.append(position_l[::-1], coord_cnn=prediction_c_dummy)

        n_gamma = max(gamma, 1)
        gammas = np.linspace(0, 2*np.pi, n_gamma, endpoint=False)
        i = 0
        break_all = False
        position_samples = utils.AccumulationArray(3)
        prediction_samples = utils.AccumulationArray(3)
        scale = 1.0 if initial_scale is None else initial_scale
        if self.rnn:
            assert len(gammas)==1
            mem_hid = self.mem_hid0
            mem_trace = self.mem_trace0
            self.model.predict_ext()
            memory_track = []
        else:
            self.model.predict() # this compiles the function

        if 'k' in self.modus:
            last_ch_max_interp = True
        else:
            last_ch_max_interp = False

        pbar = tqdm.tqdm(total=count, smoothing=0.15, desc=info_str, leave=False)
        end_message = ""
        continue_tracing = True
        features = np.array([17,0,0,0,0,0,0])
        while i < count:
            for gamma in gammas:
                try:
                    image, _, M = transformations.get_tracing_slice(self.data,
                                  self.patch_size, position_l, z_shift=self.z_shift,
                                  aniso_factor=2, sample_aniso=True, gamma=gamma,
                                  scale_factor=scale, direction_iso=direction_il,
                                  last_ch_max_interp=last_ch_max_interp)

                    trafo = transformations.Transform(M, position_l=position_l,
                                                      aniso_factor=2)
                    tt.check("Slicing", silent=True)
                except (transformations.WarpingOOBError, ValueError):
                    end_message = "Stop: out of BOUNCE after %g pix at %s"\
                    %(trace_xyz.runlength, position_l.astype(np.int))

                    break_all = True
                    break

                if self.rnn:
                    if 'trace_join' in self.model.nodes:
                        pred, gru = self.model.predict_ext(mem_trace, image[None], mem_hid)
                    else:
                        pred, gru = self.model.predict_ext(image[None], mem_hid)
                    prediction_c = pred[0,:3]
                    features = pred[0,3:]
                    if 'r' in self.modus:
                        radius = features[0] # this is only the predicted radius (scaled)
                        features[0] /= scale # make radius true
                        scale = self.get_scale_factor(radius, scale, 1.0)

                else:
                    prediction_c = self.model.predict(image[None])[0]
                    features = None

                tt.check("GPU", silent=True)
                position_l, direction_il = trafo.cnn_pred2lab_position(prediction_c)
                position_samples.append(position_l)
                prediction_samples.append(prediction_c)

            if len(trace_xyz)-len(trace_xyz.features)==1 and self.rnn: # repeat last feature prediction
                trace_xyz.features.append(features) # for first entry

            if break_all:
                break

            position_zyx = position_samples.mean()
            prediction_c = prediction_samples.mean()
            trace_xyz.append(position_zyx[::-1], coord_cnn=prediction_c, features=features)
            prediction_samples.clear()
            position_samples.clear()

            if self.rnn:
                memory_track.append(gru)
                if 'm' in self.modus:
                    mem_hid = gru

            if check_for_lost_track and trace_xyz.lost_track:
                end_message = "Stop: LOST TRACK after %g pix"\
                      %trace_xyz.runlength
                break

            if check_for_uturn and trace_xyz.uturn_occurred:
                end_message = "Stop: U-TURN after %g pix"\
                      % trace_xyz.runlength
                break

            if check_bb:
                if self.bounding_box_zyx is not None:
                    if np.any(position_zyx<self.bounding_box_zyx[0]) or \
                       np.any(position_zyx>self.bounding_box_zyx[1]):
                        end_message = "Stop: out of BBOX after %g pix at %s"\
                        %(trace_xyz.runlength,position_l.astype(np.int))
                        break

            if self.shotgun_registry and (i+2)%self.registry_interval==0 and i > 1:
                continue_tracing = self.shotgun_registry.check(trace_xyz)
                if not continue_tracing:
                    end_message = "Stop: MERGE with (%i) after %g pix at %s"\
                    %(self.shotgun_registry.last_merge_count,
                      trace_xyz.runlength, position_l.astype(np.int))
                    break

            if info_str:
                pbar.set_description(desc=info_str + ' s=%.2f' % scale)
            else:
                pbar.set_description(desc='s=%.2f' % scale)
            pbar.update()
            i += 1
            tt.check("Data struct", silent=True)

        pbar.close()

        if self.shotgun_registry and "MERGE" not in end_message: # check if OOB etc.
            continue_tracing = self.shotgun_registry.check(trace_xyz)
            if not continue_tracing:
                end_message += "\n + MERGE with(%i)"%self.shotgun_registry.last_merge_count
            else:
                end_message += " No MERGE"

        if end_message != "":
            tqdm.tqdm.write("\n"+end_message)

        tt.check("Data struct", silent=True)
        if reject_obb_traces and ("BBOX" in end_message) or \
           reject_obb_traces and ("BOUNCE" in end_message):
            tqdm.tqdm.write("REJECTING Trace because out of BBOX or OOB")
            if profile:
                tt.summary(print_func=tqdm.tqdm.write)
            return None

        if profile:
            tt.summary(print_func=tqdm.tqdm.write)

        if self.rnn:
            assert len(trace_xyz.features)==len(trace_xyz)
            return trace_xyz, memory_track
        else:
            return trace_xyz

    @staticmethod
    def perturb_direciton(direc, azimuth, polar):
        M = transformations.get_rotmat_from_direc(direc)[:3,:3] # dot(M[, b) = [1,0,0]
        M_inv = np.linalg.inv(M)
        direc_new = np.array([np.cos(polar),
                              np.sin(polar)*np.sin(azimuth),
                              np.sin(polar)*np.cos(azimuth)])
        direc_new = np.dot(M_inv, direc_new)
        return direc_new

    @staticmethod
    def plot_vectors(cv, vectors, fig=None):
        if fig is None:
            fig = mlab.figure(bgcolor=(1.0, 0.8, 0.4), size=(600,400))

        vectors = np.array(vectors)
        zero = np.zeros(vectors.shape[0])
        mlab.quiver3d([0],[0],[0], cv[0:1], cv[1:2], cv[2:3], figure=fig,color=(0, 0.0, 0.0), scale_factor=3, line_width=5)
        mlab.quiver3d(zero, zero, zero, *vectors.T, figure=fig,color=(0, 0.6, 0.2), scale_factor=3, line_width=3)
        mlab.quiver3d(zero, zero, zero, vectors[:,0], vectors[:,1], vectors[:,2], figure=fig,color=(0, 0.6, 0.2), scale_factor=3, line_width=3)

        return fig


class BallGenerator(object):
    def __init__(self, aniso):
        self.aniso = aniso
    @staticmethod
    def _make_ball(diameter, aniso):
        r2 = np.linspace(-1, 1, num=diameter) ** 2
        dist2 = r2[:, None, None] + r2[:, None] + r2
        ball = (dist2 <= 1.0 + 1e-3).astype(np.bool)
        if aniso:
            ball = ball[::2, :, :]
        return ball

    @utils.cache()
    def __call__(self, diameter):
        return self._make_ball(diameter, self.aniso)

ball_generator = BallGenerator(True)
ball_generator_iso = BallGenerator(False)

class ShotgunRegistry(object):
    def __init__(self, seeds_zyx, registry_extent, directions=None, debug=False,
                 radius_discout=0.5, check_w=3, occupied_thresh=0.6,
                 candidate_max_rel=0.75, candidate_max_min_margin=1.5):
        self.seeds_zyx = seeds_zyx
        self.registry_extent = np.array(registry_extent)
        self.mask = np.full(registry_extent, -1, dtype=np.int32)
        self.traces = []
        self.edges = []
        self.curr_seed_num = -1 # wrt original seeds
        self.curr_trace_num = -1
        self.directions = directions
        self.last_merge_count = 0

        self.occupied_thresh = occupied_thresh
        self.radius_discount=radius_discout
        self.check_w = check_w
        self.candidate_max_rel = candidate_max_rel
        self.candidate_max_min_margin = candidate_max_min_margin
        self.debug_mask = None

        if debug:
            self.debug_mask = np.full(registry_extent, -1, dtype=np.uint16)

    def new_trace(self, trace):
        self.traces.append(trace)
        self.curr_trace_num += 1
        assert (self.curr_trace_num+1)==len(self.traces)

    def check(self, trace):
        """
        Check if trace goes into masked volume. If so, find out to which trace
        tree this belongs and merge.
        Return False to stop tracing
        Mask seeds and volume mask by current trace's log

        W: window length to do check on
        """
        if len(trace)<(self.check_w):
            return True

        coords_xyz = trace.coords[-self.check_w:]
        positions_zyx = np.round(coords_xyz[:,::-1]).astype(np.int)
        positions_zyx = np.clip(positions_zyx, [0,0,0], self.registry_extent-1)
        z,y,x = positions_zyx[:,0], positions_zyx[:,1], positions_zyx[:,2]
        occupied = (self.mask[z,y,x]>=0).mean() > self.occupied_thresh

        continue_tracing = True
        if occupied:
            continue_tracing = False
            ids = self.mask[z,y,x]
            ids = ids[ids>=0]
            indices, counts = np.unique(ids, return_counts=True)
            self.last_merge_count = len(indices)
            i = indices[counts.argmax()]
            if len(indices)>1:
                pass #check if the different indices are at least connected?

            self.edges.append([self.curr_trace_num, i] + [1,1,1])
            self.edges.append([i, self.curr_trace_num] + [1,1,1])

            # indices, distance_track = self.find_nearest_trace(coords_xyz)
            # self.last_merge_count = len(indices)
            # for i, t in zip(indices, distance_track):
            #     self.edges.append([self.curr_trace_num, i] + list(t))
            #     self.edges.append([i, self.curr_trace_num] + list(t))

        return continue_tracing


    def find_nearest_trace(self, coords_xyz):
        """
        Find all other tracks that are at least as close as 1.5 minimal
        (relative!) distance. (compare to closest point of each track)
        """
        distance_track = np.zeros((len(self.traces[:-1]), 3))
        for i,tr in enumerate(self.traces[:-1]):
            distances, indices, coordinates = tr.kdt.get_knn(coords_xyz, k=1)
            radii = tr.features[:,0][indices]
            k = np.argmin(distances/radii)
            distance_track[i] = [distances[k], radii[k], distances[k]/radii[k]]

        min_dist = (distance_track[:,2]).min()
        mask = (distance_track[:,2] < min_dist*self.candidate_max_min_margin) | (distance_track[:,2] < self.candidate_max_rel)
        distance_track_r = distance_track[mask]
        indices_r = mask.nonzero()[0]
        distance_track_r = np.maximum(distance_track_r, 1e-4) #
        return indices_r, distance_track_r


    def update_mask(self, coords_xyz, radii, index=None):
        radii = radii * self.radius_discount
        diameter = (radii*2).astype(np.int)
        diameter += 1-np.mod(diameter, 4)
        diameter = np.maximum(1, diameter)
        for c_xyz, d  in zip(coords_xyz, diameter):
            x,y,z = np.round(c_xyz).astype(np.int)
            r = d // 2
            r2 = d // 4
            ball = ball_generator(d)
            try:
                self.mask[z-r2:z+r2+1,y-r:y+r+1, x-r:x+r+1][ball] = index
                if index is not None and self.debug_mask is not None:
                    self.debug_mask[z-r2:z+r2+1,
                                    y-r :y+r+1,
                                    x-r :x+r+1] = ball*index
            except (ValueError, IndexError):
                print(z,y,x)
                print(ball.shape)
                print(self.mask[z-r2:z+r2+1,
                              y-r :y+r+1,
                              x-r :x+r+1].shape)
                print("trying to insert ball OBB?")

                raise RuntimeError


    def get_next_seed(self):
       max_count = len(self.seeds_zyx)
       position_l = None
       direc_il   = None
       i = self.curr_seed_num + 1
       while True:
           if i >= max_count:
               logger.info("All Seeds are done")
               break
           if  self.mask[tuple(self.seeds_zyx[i])] < 0:
               position_l = self.seeds_zyx[i]
               self.curr_seed_num = i
               if self.directions is None:
                   direc_il = (self.extent/2-position_l)
                   direc_il /= np.linalg.norm(direc_il)
               else:
                   direc_il = self.directions[i]

               break

           i += 1

       return position_l, direc_il


    def plot_mask_vol(self, figure=None, adjust_tfs=False):
        img = np.repeat(self.mask,2,axis=0).astype(np.float32)
        img = ndimage.filters.gaussian_filter(img, (1.0,1.0,1.0))
        img = img[::4,::4,::4]
        if figure is None:
            figure = mlab.figure()
        vol = mlab.pipeline.scalar_field(img)
        mlab.pipeline.volume(vol, color=(1,1,1), vmin=0.0-1e-3, vmax=1.0+1e-3, figure=figure)
        figure.scene.background = (1.0, 0.8, 0.4)
        figure.scene.parallel_projection = True
        volume = figure.children[0].children[0].children[0]
        volume.volume_mapper.reference_count = 3
        volume.volume_mapper.progress = 1.0
        volume.volume_mapper.interactive_sample_distance = 0.5
        volume.volume_mapper.sample_distance = 0.5
        volume.volume_mapper.maximum_image_sample_distance = 2.0
        volume.volume_property.scalar_opacity_unit_distance = 0.5
        volume.volume_property.specular = 0.3
        volume.volume_property.ambient = 0.2

        figure.scene.render()
        return figure


class CubeShape(object):
    def __init__(self, shape, offset=None, center=None, input_excess=None,
                 bbox_reduction=None):
        shape = np.array(shape)
        assert np.all(shape%2==0)

        self.shape = shape
        if offset is not None:
            assert center is None
            self.offset = np.array(offset)
            self.center = self.offset + self.shape.astype(np.float)/2 - 0.5
        elif center is not None:
            assert offset is None
            self.center =  np.array(center)
            self.offset = (self.center - self.shape//2 + 0.5).astype(np.int)
        else:
            raise ValueError()

        self.input_excess = np.array(input_excess)
        self.bbox_reduction = np.array(bbox_reduction)

    def shrink_off_sh_cent(self, amount):
        amount = np.array(amount)
        off = self.offset - amount
        sh  = self.shape + 2*amount
        cent = off + sh.astype(np.float)/2 - 0.5
        return off, sh, cent

    def input_off_sh_cent(self, input_excess=None):
        if input_excess is None and self.input_excess is not None:
            return self.shrink_off_sh_cent(self.input_excess)
        elif input_excess is not None:
            return self.shrink_off_sh_cent(input_excess)
        else:
            raise ValueError()

    def bbox_off_sh_cent(self, bbox_reduction=None):
        if bbox_reduction is None and self.bbox_reduction is not None:
            return self.shrink_off_sh_cent(-self.bbox_reduction)
        elif bbox_reduction is not None:
            return self.shrink_off_sh_cent(-bbox_reduction)
        else:
            raise ValueError()

    def bbox_wrt_input(self):
        assert self.input_excess is not None and self.bbox_reduction is not None
        lower = self.input_excess + self.bbox_reduction
        upper = self.input_excess + self.shape - 1*self.bbox_reduction
        return np.array([lower, upper])


    def bbox_wrt_self(self):
        assert self.input_excess is not None and self.bbox_reduction is not None
        lower = self.bbox_reduction
        upper = self.shape - 1*self.bbox_reduction
        return np.array([lower, upper])


def sort_seeds(seeds, directions, seed_values, first_dist, border, bbox):
    # first_dist: seeds_values larger than this are sorted before the others
    # border: [dz, dy, dx], thickness of border partition of cube, seeds from
    # the border are sorted prior to other seeds

    # first filter seeds which are not in bounding box
    bbox_mask = np.logical_and(np.all(seeds > bbox[0], axis=1),
                               np.all(seeds < bbox[1],axis=1))

    seeds = seeds[bbox_mask]
    seed_values = seed_values[bbox_mask]

    if len(seeds)==0:
        return None

    border = np.array(border)
    border_mask = np.logical_and(np.all(seeds > bbox[0] + border, axis=1),
                                 np.all(seeds < bbox[1] - border, axis=1))  # inner cube

    sort_mask = np.argsort(seed_values)[::-1] # is ascending!
    try:
        split = (seed_values[sort_mask] < first_dist).nonzero()[0][0]
    except IndexError: # if all values are larger than first dist
        split = 0

    border_mask = border_mask[sort_mask]
    faces0 = sort_mask[:split][~border_mask[:split]]
    inner0 = sort_mask[:split][border_mask[:split]]
    faces1 = sort_mask[split:][~border_mask[split:]]
    inner1 = sort_mask[split:][border_mask[split:]]
    new_index = np.hstack((faces0, inner0, faces1, inner1))

    seeds_new = seeds[new_index]
    seed_vals_new = seed_values[new_index]
    directions_new = directions[new_index]
    return seeds_new, directions_new, seed_vals_new

@utils.timeit
def make_shotgun_data_z(cube_shape, save_name, z_skip=5):
    barr_thresh = 0.7
    z_lookaround = 5
    max_footprint_S = ball_generator(9)
    max_footprint_L = ball_generator(21)
    max_footprint_L = max_footprint_L[3:-3]

    peak_thresh = 3.0
    peak_thresh_L = 9

    kds_barr = KnossosDataset()
    data_prefix = os.path.expanduser("~/lustre/sdorkenw/j0126_")
    kds_barr.initialize_from_knossos_path(data_prefix+'161012_barrier/')
    pred = kds_barr.from_raw_cubes_to_matrix(cube_shape.shape,
                                             cube_shape.offset,
                                             show_progress=False,
                                             zyx_mode=True, datatype=np.float32)
    pred /= 255

    mem_high = np.invert(pred > barr_thresh)
    seeds = []
    seed_values = []
    noise = np.random.rand(*(mem_high[0:0 + z_lookaround].shape)) * 1e-3
    running_sum = 0
    for z in range(0, mem_high.shape[0] - z_lookaround, z_skip):
        try:
            dt = ndimage.distance_transform_edt(mem_high[z:z + z_lookaround], sampling=[2, 1, 1])
            dt = ndimage.filters.gaussian_filter(dt, (1.0,2.0,2.0))
            dt += noise

            z_peaks_S = ndimage.maximum_filter(dt, footprint=max_footprint_S,mode='constant')
            z_peaks_L = ndimage.maximum_filter(dt, footprint=max_footprint_L,mode='constant')

            z_peaks_small = (z_peaks_S == dt) * ((peak_thresh_L > dt) & (dt > peak_thresh))
            z_peaks_large = (z_peaks_L == dt) * ((peak_thresh_L <= dt))
            z_peaks = z_peaks_large + z_peaks_small
            z_peaks *= (pred[z:z + z_lookaround] < 0.5)
            seeds_z = np.array(z_peaks.nonzero()).T
            seeds_z[:, 0] += z
        except KeyboardInterrupt:
            break
        finally:
            seeds.append(seeds_z)
            seed_values.append(dt[z_peaks])
            running_sum += z_peaks.sum()
            print(z, running_sum, z_peaks_small.sum(), z_peaks_large.sum(), z_peaks.sum())

    seeds = np.concatenate(seeds, axis=0)
    seed_values = np.concatenate(seed_values, axis=0)
    seeds, index = unique_rows(seeds)
    seed_values = seed_values[index]


    lar = np.array([4,8,8])
    lari = lar * [2,1,1]
    sz  = lar * 2 + 1
    szi = lari * 2 + 1

    pred2 = kds_barr.from_raw_cubes_to_matrix(cube_shape.shape + 2 * lar,
                                              cube_shape.offset - lar,
                                              show_progress=False,
                                              zyx_mode=True, datatype=np.float32)
    pred2 /= 255

    mem_high2 = np.invert(pred2 > barr_thresh)
    dt = ndimage.distance_transform_edt(mem_high2, sampling=[2, 1, 1])
    local_grid = np.vstack([x-x.mean() for x in np.ones(szi).nonzero()])
    directions = np.zeros((len(seeds), 3))
    perm = np.random.permutation(local_grid.shape[1])[:400]
    for i, (s, v) in enumerate(zip(seeds, seed_values)):
        z, y, x = s  # np.round(s).astype(np.int)
        cavity = dt[z:z + sz[0], y:y + sz[1], x:x + sz[2]]
        cavity = ndimage.zoom(cavity, [float(sz[1])/sz[0],1,1])
        s_val = dt[z + lar[0], y + lar[1], x + lar[2]]
        diff = np.abs(cavity - s_val)
        d_m = diff.mean()
        mask = (diff < d_m)
        d_max = diff[mask].max()
        um = np.zeros_like(diff)
        um[mask] = d_max - diff[mask]
        um = um.ravel()
        uu, dd, vv = np.linalg.svd((um * local_grid).T[perm])
        direc_iso = vv[0]  # take largest eigenvector
        direc_iso /= np.linalg.norm(direc_iso, axis=0)  # normalise
        directions[i] = direc_iso

#    local_grid = np.mgrid[-2:2:5j, -2:2:5j,-2:2:5j]
#    local_grid = np.vstack([g.ravel() for g in local_grid])
#    directions = np.zeros((len(seeds), 3))
#    for i, s in enumerate(seeds):
#        z, y, x = s  # np.round(s).astype(np.int)
#        cavity = pred2[z:z + 3, y:y + 5, x:x + 5]
#        cavity = ndimage.zoom(cavity, [5.0/3.0,1,1] )
#        cavity = (1 - cavity).ravel()
#        uu, dd, vv = np.linalg.svd((cavity * local_grid).T)
#        direc_iso = vv[0]  # take largest eigenvector
#        direc_iso /= np.linalg.norm(direc_iso, axis=0)  # normalise
#        directions[i] = direc_iso

    seeds += cube_shape.offset
    utils.picklesave([seeds, directions, seed_values], save_name)

    # Creater skeleton with seeds and dirs
    skel_obj = knossos_skeleton.Skeleton()
    anno = knossos_skeleton.SkeletonAnnotation()
    anno.scaling = (9.0, 9.0, 20.0)
    skel_obj.add_annotation(anno)
    def add_node(s, r=5):
        z,y,x = s
        new_node = knossos_skeleton.SkeletonNode()
        new_node.from_scratch(anno, x, y, z, radius=r)
        anno.addNode(new_node)
        return new_node

    for s, dir, v in zip(seeds, directions, seed_values):
        n = add_node(s, r=4)
        n.appendComment("%.1f" %v)
        dir = dir.copy()
        dir[0] /= 2
        n1 = add_node((s + 11 * dir), r=1)
        n2 = add_node((s - 11 * dir), r=1)
        n1.addChild(n)
        n2.addChild(n)

    return seeds, directions, seed_values, skel_obj


def rebuild_model(model=None, path=None, save_name=None):
    from elektronn2 import neuromancer
    if model is None:
        model = neuromancer.model.modelload(path)

    model = neuromancer.model.rebuild_rnn(model)
    fail_prob = list(filter(lambda x: 'failing_prob' in x, model.nontrainable_params.keys()))
    if fail_prob:
        assert len(fail_prob)==1
        model.nontrainable_params[fail_prob[0]].set_value(0.0)
        print("NOTE: Setting fail prob to 0")

    if len(model.dropout_rates):
         model.dropout_rates = 0.0
         print("NOTE: Setting dropout_rates to 0")

    if save_name is not None:
        model.save(save_name)
    return model

def get_stitching_edges(traces_a, traces_b, k=2, rel_dist_thresh=1.2):
    edges = []
    #stats = []
    #stats_mean = []
    for i_a,t_a in enumerate(traces_a):
        for i_b,t_b in enumerate(traces_b):
            if len(t_a) < len(t_b):
                t_long = t_b
                t_short = t_a
            else:
                t_long = t_b
                t_short = t_a

            tmp = t_long.kdt.get_knn(t_short.coords.data, k=1)
            distances, indices, coords = tmp
            min_ix = distances.argsort()[:k]
            distances = distances[min_ix]
            indices = indices[min_ix]
            radii = (t_long.features[indices,0] + t_short.features[min_ix,0]) / 2
            rel_distances = distances / radii.mean()
            rel_mean = rel_distances.mean()
            #stats.extend(rel_distances)
            #stats_mean.append(rel_mean)

            if rel_mean < rel_dist_thresh:
                edges.append((i_a, i_b, rel_mean))

    #stats = np.array(stats)
    #stats_mean = np.array(stats_mean)
    return edges

def find_job(args):
    a, t_a, bs, trees_b, rel_dist_thresh, rel_search_radius, max_dist, thresh, k, dir_name = args
    print(a)
    edges = []
    for b, t_b in zip(bs, trees_b):
        if len(t_a.joined_coords) < len(t_b.joined_coords):
            t_long = t_b
            t_short = t_a
        else:
            t_long = t_b
            t_short = t_a

        tmp = t_long.joined_kdt.get_knn(t_short.joined_coords, k=1)
        distances, indices, coords = tmp

        min_ix = distances.argsort()[:k]
        distances = distances[min_ix]
        indices = indices[min_ix]
        radii = (t_long.joined_radii[indices] + t_short.joined_radii[min_ix]) / 2
        rel_distances = distances / radii.mean()
        rel_mean = rel_distances.mean()
        if rel_mean < rel_dist_thresh:
            pass #this should have been added already
        elif rel_mean < rel_search_radius and distances.mean() < max_dist:
            # Warning this works on isotropic coords
            closest_point_long = coords[min_ix[0]].copy()
            closest_point_short = t_short.joined_coords[min_ix[0]].copy()
            try:
                tmp = t_long.joined_kdt.get_knn(closest_point_short, k=20)
            except ValueError:
                k_ = len(t_long.joined_coords)
                tmp = t_long.joined_kdt.get_knn(closest_point_short, k=k_)

            coords_long = tmp[2].copy()
            try:
                tmp = t_short.joined_kdt.get_knn(closest_point_long, k=20)
            except ValueError:
                k_ = len(t_short.joined_coords)
                tmp = t_short.joined_kdt.get_knn(closest_point_long, k=k_)

            coords_short = tmp[2].copy()

            closest_point_long[2] *= 2
            closest_point_short[2] *= 2
            coords_long[:, 2] *= 2
            coords_short[:, 2] *= 2

            center_long = coords_long.mean(0)
            center_short = coords_short.mean(0)

            uu, dd, vv = np.linalg.svd(coords_long - center_long)
            direc_long = vv[0]  # take largest eigenvector

            uu, dd, vv = np.linalg.svd(coords_short - center_short)
            direc_short = vv[0]  # take largest eigenvector

            direc_centers = center_long - center_short
            direc_centers /= np.linalg.norm(direc_centers)

            direc_pairs = closest_point_long - closest_point_short
            direc_pairs /= np.linalg.norm(direc_pairs)

            # align all directions:
            if np.dot(direc_short, direc_centers) < 0:
                direc_short *= -1
            if np.dot(direc_long, direc_centers) < 0:
                direc_long *= -1
            if np.dot(direc_pairs, direc_centers) < 0:
                direc_pairs *= -1

            m1 = np.dot(direc_short, direc_centers)
            m2 = np.dot(direc_long, direc_centers)
            m = (m1 + m2) / 2
            print(m1, m2, m)

            if m>thresh:
                edges.append((a,b))


            ###############################################################
            skel_obj = knossos_skeleton.Skeleton()
            anno = knossos_skeleton.SkeletonAnnotation()
            anno.scaling = (9.0, 9.0, 20.0)
            skel_obj.add_annotation(anno)
            anno.setComment("%.1f %.1f %.1f" %(m, m1, m2,))

            def add_node(x, y, z, r=5):
                new_node = knossos_skeleton.SkeletonNode()
                new_node.from_scratch(anno, x, y, z // 2, radius=r)
                anno.addNode(new_node)
                return new_node

            n0 = add_node(*center_long, r=10)
            n1 = add_node(*center_short, r=10)
            n0.addChild(n1)

            n2 = add_node(*closest_point_long, r=5)
            n3 = add_node(*closest_point_short, r=5)
            n2.addChild(n3)

            n4 = add_node(*(center_long + 20 * direc_long), r=3)
            n5 = add_node(*(center_long - 20 * direc_long), r=3)
            n0.addChild(n4)
            n0.addChild(n5)

            n6 = add_node(*(center_short + 20 * direc_short), r=3)
            n7 = add_node(*(center_short - 20 * direc_short), r=3)
            n1.addChild(n6)
            n1.addChild(n7)

            if not os.path.exists("merge_candidates"):
                os.makedirs("merge_candidates")

            skel_obj.to_kzip("%s/splitfix_candidates/%i.k.zip" %(dir_name, time.time()))
            ###############################################################

    return edges

def find_false_splits(simple_trees, k=2, thresh=0.85, rel_dist_thresh=1.2,
                        rel_search_radius=6, max_dist=40.0, dir_name=''):
    n = len(simple_trees)
    var_args = []
    for a  in range(n):
        t_a = simple_trees[a]
        bs = range(a + 1, n)
        trees_b = simple_trees[slice(a+1, n)]
        var_args.append([a, t_a, bs, trees_b])

    const_args = (rel_dist_thresh, rel_search_radius, max_dist, thresh, k, dir_name)
    edges = utils.parallel_accum(find_job, 1, var_args, const_args, proc=20)
    return np.array(edges)
