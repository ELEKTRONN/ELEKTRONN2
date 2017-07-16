# -*- coding: utf-8 -*-
# ELEKTRONN2 Toolkit
# Copyright (c) 2015 Marius F. Killinger
# All rights reserved

from __future__ import absolute_import, division, print_function
from builtins import filter, hex, input, int, map, next, oct, pow, range, super, zip


import logging
import os
import sys
import time
import getpass
import traceback
from collections import OrderedDict

import numpy as np
import scipy.misc as misc
import matplotlib.pyplot as plt

import theano

from ..neuromancer.model import modelload, rebuild_model
from ..neuromancer.loss import SquaredLoss
from .. import utils
from ..config import config
from ..utils.plotting import plot_trainingtarget, my_quiver
from ..utils.locking import FileLock
from ..data import transformations
from . import trainutils
from .parallelisation import BackgroundProc
from .trainutils import HistoryTracker, Schedule

floatX = theano.config.floatX

logger = logging.getLogger('elektronn2log')
inspection_logger = logging.getLogger('elektronn2log-inspection')
user_name = getpass.getuser()
__all__ = ['Trainer', 'TracingTrainer', 'TracingTrainerRNN']


NLL_TEXT = "The NN diverged to `nan` Loss!!!\n "\
           "You have the chance to inspect the last used examples and the "\
           "internal state of the pipeline in the command line. The last "\
           "presented training input data is `batch[0]` and the "\
           "corresponding target `batch[1]`"\

class Trainer(object):
    def __init__(self, exp_config):
        self.exp_config = exp_config
        self.schedules = OrderedDict()
        self.model      = self._create_model()
        self.data       = self._load_data()
        self.batch_size = self._infer_batch_size()
        self.tracker    = HistoryTracker()
        self.tracker.register_debug_output_names(self.model.debug_output_names)

        self.preview_data = self._load_preview_data()
        self.saved_raw_preview = False

        self.get_batch_kwargs = self.exp_config.data_batch_args
        self.get_batch_kwargs['batch_size'] = self.batch_size
        self.get_batch_kwargs['source'] = 'train'

        if self.exp_config.monitor_batch_size%self.batch_size!=0:
            bs = int(np.ceil(float(self.exp_config.monitor_batch_size)/self.batch_size))
            self.exp_config.monitor_batch_size = bs


        os.chdir(self.exp_config.save_path) # The trainer works directly in the save dir



        #self.debug_store = []


    def _create_model(self):
        if self.exp_config.create_model:
            if self.exp_config.model_load_args:
                mdl =  self.exp_config.create_model(
                    self.exp_config.model_load_args)
            else:
                mdl =  self.exp_config.create_model()
        else:
            mdl =  modelload(self.exp_config.model_load_path,
                             **self.exp_config.model_load_args)

        mdl.set_opt_meta_params(self.exp_config.optimiser,
                                       self.exp_config.optimiser_params)

        for var, params in self.exp_config.schedules.items():
            if params:
                schedule = Schedule(**params)
                try:
                    schedule.bind_variable(obj=mdl, prop_name=var)
                except:
                    logger.debug("%s not found in model, trying config now" %(var))
                    schedule.bind_variable(obj=self.exp_config, prop_name=var)

                self.schedules[var] = schedule
                logger.info(schedule)

        return mdl


    def _load_data(self):
        if isinstance(self.exp_config.data_class, (list, tuple)):
            mod, cls = self.exp_config.data_class
            cls = utils.import_variable_from_file(mod, cls)
        else:
            from .. import data
            cls = getattr(data, self.exp_config.data_class)

        return cls(self.model.input_node, self.model.target_node,
                   **self.exp_config.data_init_kwargs)

    def _load_preview_data(self):
        if self.exp_config.preview_data_path is not None:
            data = utils.h5load(self.exp_config.preview_data_path)
            if not (isinstance(data, list) or  isinstance(data, (tuple, list))):
                data = [data,]
            data = [d.astype(floatX)/d.max() for d in data]
            return data
        else:
            return None

    def _infer_batch_size(self):
        model_bs = self.model.batch_size
        conf_bs  = self.exp_config.batch_size
        if model_bs:
            if conf_bs:
                assert model_bs==conf_bs, "Conflicting batchsizes from " \
                                          "model (%d) and experiment " \
                                          "configuration (%d)" % (model_bs, conf_bs)
            return model_bs
        elif conf_bs:
            return conf_bs


    def run(self):
        exp_config  = self.exp_config
        save_name   = exp_config.save_name
        data        = self.data

        t_passed    = 0
        t_pt        = 2
        t_pi        = 2
        last_save_t = 0
        last_save_t2= 0
        save_time   = config.param_save_h
        save_time2  = config.initial_prev_h

        loss, loss_smooth, train_loss, valid_loss, train_error, valid_error, param_vars = 0, 0, 0, 0, 0, 0, 0
        user_termination = False

        if isinstance(self.model.loss_node.parent[0], SquaredLoss):
            is_regression = True
        else:
            is_regression = False
        pp_err  = 'err' if is_regression else '%'

        # --------------------------------------------------------------------------------------------------------
        if config.background_processes:
            n_proc = max(2, int(config.background_processes))
            bg_worker = BackgroundProc(data.getbatch, n_proc=n_proc, target_kwargs=self.get_batch_kwargs)
        # --------------------------------------------------------------------------------------------------------
        try:
            i = -1
            t0 = time.time()
            while i < exp_config.n_steps:
                print('{:05d}'.format(i), end='\r')
                sys.stdout.flush()
                try:
                    if config.background_processes:
                        batch = bg_worker.get()
                    else:
                        batch = data.getbatch(**self.get_batch_kwargs)

                    if exp_config.class_weights is not None:
                        batch = batch + (exp_config.class_weights,)

                    #self.debug_store.append(batch[1])
                    #-----------------------------------------------------------------------------------------------------
                    loss, t_per_train, debug_outputs = self.model.trainingstep(*batch, optimiser=exp_config.optimiser) # Update step
                    i += 1
                    #-----------------------------------------------------------------------------------------------------
                    t_per_it = time.time() - t0
                    t_passed += t_per_it
                    t0 = time.time()
                    t_pi = 0.8*t_pi + 0.2*t_per_it     # EMA
                    loss_smooth = self.model.loss_smooth

                    # check for divergence
                    if np.any(np.isnan(loss)) or np.any(np.isinf(loss)):
                        logger.warning(NLL_TEXT)
                        raise KeyboardInterrupt
                        #self.self.model.optimisers[exp_config.optimiser].repair()

                    if len(batch) == 1:
                        batch_char = 0
                    else:
                        batch_char = batch[1][:,0].mean() # assuming targets have shape (b,f,...)
                    self.tracker.update_timeline([t_passed, loss, batch_char])
                    if debug_outputs:
                        self.tracker.update_debug_outputs([i, loss,]+debug_outputs)

                    # Save Parameters
                    # if (t_passed-last_save_t)/3600 > config.param_save_h:
                    #     last_save_t = t_passed
                    #     time_string = '-'+str(save_time)+'h'
                    #     self.model.save(os.path.join('Backup', save_name+time_string+'.mdl'))
                    #     save_time += config.param_save_h
                    if i%config.param_save_it==0 and i>0:
                        it_string = '-'+str(i//1000)+'k'
                        self.model.save(os.path.join('Backup', save_name+it_string+'.mdl'))

                    # Create preview prediction images
                    if self.preview_data is not None:
                        if (t_passed-last_save_t2)/3600 > config.prev_save_h \
                            or (t_passed/3600 > config.initial_prev_h and last_save_t2==0): # first time
                            last_save_t2 = t_passed
                            exp_config.preview_kwargs['number'] = save_time2
                            save_time2 += config.prev_save_h
                            try:
                                self.preview_slice(**exp_config.preview_kwargs)
                            except:
                                logger.warning("Preview Predictions failed."
                                               "Are the preview raw data in "
                                               "the correct format?")
                            # reset time because we only count training time
                            # not time spent for previews (making previews
                            # is not a computational payload of the actual
                            # training but just for "fun")
                            t0 = time.time()

                    # Adjust the learning rate and other schedule parameters
                    for schedule in self.schedules.values():
                        if i==schedule.next_update:
                            schedule.update(i)

                    if (i%exp_config.history_freq==0) and exp_config.history_freq!=0:
                        lr = self.model.lr
                        mom = self.model.mom
                        if len(self.model.gradnet_rates):
                            gradnetrate = np.mean(self.model.gradnet_rates)
                        else:
                            gradnetrate = 0

                        ### Training  & Valid Errors ###
                        loss_after = self.model.loss(*batch)
                        loss_gain  = loss_after - loss
                        train_loss, train_error = self.test_model('train')
                        valid_loss, valid_error = self.test_model('valid')
                        if not is_regression:
                            train_error *= 100
                            valid_error *= 100

                        self.tracker.update_history([i, t_passed, train_loss,
                                                     valid_loss, loss_gain,
                                                     train_error, valid_error,
                                                     lr, mom, gradnetrate])

                        ### Plotting / Saving ###
                        self.model.save(save_name+'-LAST.mdl')
                        self.tracker.save(os.path.join('Backup', save_name))
                        if config.plot_on and ((i>=exp_config.history_freq*3) or i>60):
                            self.tracker.plot(save_name)

                        if config.print_status:
                            t = utils.pretty_string_time(t_passed)
                            out = "%05i L_m=%.3f, L=%.2f, tr=%05.2f%s, "%(i,
                                   loss_smooth, loss, train_error, pp_err)
                            out +="vl=%05.2f%s, prev=%04.1f, L_diff=%+.1e, "\
                                   %(valid_error, pp_err, batch_char*100, loss_gain)
                            out +="LR=%.5f, %.2f it/s, %s" %(lr, 1.0/t_pi, t)
                            logger.info(out)

                # User Interface ##############################################
                except (KeyboardInterrupt, ValueError, TypeError) as e:
                    if not isinstance(e, KeyboardInterrupt):
                        traceback.print_exc()
                        print("\nEntering Command line such that Exception can be "
                              "further inspected by user.\n\n")

                    out = "%05i L_m=%.5f, L=%.4f, train=%.5f, valid=%.5f, " %(i,
                           loss_smooth, loss, train_loss, valid_loss)
                    out +="train=%.3f%s, valid=%.3f%s,\n LR=%.6f, MOM=%.6f, "\
                           %(train_error, pp_err, valid_error, pp_err, self.model.lr, self.model.mom)
                    out +="%.1f GPU-it/s, %.1f CPU-it/s, " %(1.0/self.model.time_per_step,
                           1.0/t_pi)

                    t = utils.pretty_string_time(t_passed)
                    logger.info(out+t)

                    # Like a command line, but cannot change singletons
                    var_push = globals()
                    var_push.update(locals())
                    ret = trainutils.user_input(var_push)
                    if ret=='kill':
                        user_termination = True

                    if config.background_processes:
                        bg_worker.reset()

                    plt.close('all')
                    # reset time after user interaction, otherwise time
                    # will appear as pause in plot
                    t0 = time.time()

                    # End UI ##################################################

                # This is in the epoch/UI loop
                if (t_passed > exp_config.max_runtime) or user_termination :
                    logger.info('Timeout or manual Termination')
                    break

            # This is OUTSIDE the training loop i.e.
            # the last block of the function ``run``
            self.model.save(save_name+"-FINAL.mdl")
            if len(self.tracker.timeline) > 10:
                self.tracker.plot(save_name)
            logger.info('End of Training')
            logger.info('#'*60 + '\n' + '#'*60 + '\n')
            # -------------------end of run()----------------------------------

        except:
            sys.excepthook(*sys.exc_info()) # show info on error
        finally:
            if config.background_processes:
                bg_worker.shutdown()

        if self.model.batch_normalisation_active:
            print("Rebuilding model, replacing batch normalisation layers "
                  "with constant values")
            self.model = rebuild_model(self.model, replace_bn='const')


    def test_model(self, data_source):
        """
        Computes Loss and error/accuracy on batch with ``monitor_batch_size``


        Parameters
        ----------

        data_source: string
            'train' or 'valid'

        Returns
        -------
        Loss, error:

        """
        # copy because it is modified in next line!
        kwargs = dict(self.get_batch_kwargs)
        kwargs['source']=data_source
        kwargs['batch_size'] = self.exp_config.monitor_batch_size
        try:
            batch = self.data.getbatch(**kwargs)
        except ValueError:
            logger.warning("Test model, getbatch failed. No validation data?")
            return np.nan, np.nan # 0, 0
        y_aux = []
        if batch[1] is None:
            return 0, 0
        if self.exp_config.class_weights is not None:
            y_aux.append(self.exp_config.class_weights)

        rates = self.model.dropout_rates
        self.model.dropout_rates = ([0.0,]*len(rates))
        batch_axis = self.model.input_node.shape.tag2index('b')
        n = batch[0].shape[batch_axis]
        loss = 0
        error = 0
        for j in range(int(np.ceil(np.float(n)/self.batch_size))):
            slice_obj = [slice(None) for i in range(batch_axis+1)]
            slice_obj[batch_axis] = slice(j*self.batch_size, (j+1)*self.batch_size)
            d = batch[0][slice_obj] # data
            l = batch[1][slice_obj] # target
            if len(batch) > 2:
                aux = []
                for b in batch[2:]:
                    aux.append(b[j*self.batch_size:(j+1)*self.batch_size])

                nl, er, pred =  self.model.predict_ext(d, l, *(aux + y_aux))
            else:
                nl, er, pred =  self.model.predict_ext(d, l, *y_aux)
            nb_samples = d.shape[batch_axis]
            loss += nl*nb_samples
            error += er*nb_samples
        loss /= n
        error /= n
        self.model.dropout_rates = rates # restore old rates
        return loss, error

    def debug_getcnnbatch(self):
        """
        Executes ``getbatch`` but with un-strided labels and always returning
        info. The first batch example is plotted and the whole batch is
        returned for inspection.
        """
        if self.model.ndim>=2:
            kwargs = dict(self.get_batch_kwargs)
            kwargs['force_dense'] = True
            batch = self.data.getbatch(**kwargs)

            data, target = batch[0][0], batch[1][0]
            target[np.isclose(target, -666)] = 0

            if self.model.ndim==2:
                if target.shape[0] >= 3:
                    target = np.transpose(target, (1,2,0))[:,:,:3]
                    target = target[...,[2,1,0]]
                else:
                    target = target[0]
                plot_trainingtarget(data[0], target, 1)
            else:
                t_i = target.shape[1]//2
                if target.shape[0] >= 3:
                    target = np.transpose(target, (1,2,3,0))[0,:,:,:3]
                    target = target[...,[2, 1, 0]]
                else:
                    target = target[0,t_i]
                i = self.data.offsets[0] # z offset
                plot_trainingtarget(data[0,i+t_i], target, 1)

            with FileLock('plotting'):
                plt.savefig('Batch_test_image.png', bbox_inches='tight')
                # Hard to reason about plt here. It's a side effect of plot_traingtarget calls.

            if config.gui_plot:
                plt.ion()
                plt.show()
                plt.pause(0.01)
                plt.pause(2.0)
                plt.close('all')
                plt.pause(0.01)

            return batch
        else:
            logger.warning('debug_getcnnbatch() is only available for "img-img"'
                           ' training mode.\nCheck if your prediction node'
                           ' has the right shape.ndim\n(look if'
                           ' model.prediction_node.shape.ndim is >=2.)')


    def predict_and_write(self, pred_node, raw_img, number=0, export_class='all', block_name='', z_thick=5):
        """
        Predict and and save a slice as preview image

        Parameters
        ----------

        raw_img : np.ndarray
          raw data in the format (ch, x, y, z)
        number: int/float
          consecutive number for the save name (i.e. hours, iterations etc.)
        export_class: str or int
          'all' writes images of all classes, otherwise only the
          class with index ``export_class`` (int) is saved.
        block_name: str
          Name/number to distinguish different raw_imges
        """
        block_name = str(block_name)
        pred = pred_node.predict_dense(raw_img) # returns (k, (z,) y, x)
        z_sh = pred.shape[1]
        pred = pred[:,(z_sh-z_thick)//2:(z_sh-z_thick)//2+z_thick,:,:,]
        save_name = self.exp_config.save_name
        with FileLock('plotting'):  # One lock is enough because everything below calls mpl directly
            for z in range(pred.shape[1]):
                if export_class=='all':
                    for c in range(pred.shape[0]):
                        plt.imsave('%s-pred-%s-z%i-c%i-%shrs.png' \
                                   %(save_name, block_name, z, c, number), pred[c,z,:,:], cmap='gray')
                elif export_class in ['malis', 'affinity']:
                    plt.imsave('%s-pred-%s-aff-z%i-%shrs.png' \
                               %(save_name, block_name, z, number),
                               np.transpose(pred[0:6:2,z,:,:],(1,2,0)), cmap='gray')
                else:
                    if isinstance(export_class, (list, tuple)):
                        for c in export_class:
                            plt.imsave('%s-pred-%s-z%i-c%i-%shrs.png' \
                                       %(save_name, block_name, z, c, number), pred[c,z,:,:], cmap='gray')

                    else:
                        c = int(export_class)
                        plt.imsave('%s-pred-%s-z%i-c%i-%shrs.png' \
                                   %(save_name, block_name, z, c, number), pred[c,z,:,:], cmap='gray')

            if not self.saved_raw_preview: # only do once
                if len(pred_node.shape.offsets)==2:
                    z_off = 0
                else:
                    z_off = int(pred_node.shape.offsets[0])

                for z in range(pred.shape[1]):
                    plt.imsave('%s-raw-%s-z%i.png'%(save_name, block_name, z), raw_img[0,z+z_off,:,:], cmap='gray')


    def preview_slice_from_traindata(self, cube_i=0, off=(0,0,0), sh=(10,400,400), number=0, export_class='all'):
        """
        Predict and and save a selected slice from the training data as preview

        Parameters
        ----------

        cube_i: int
          index of source cube in CNNData
        off: 3-tuple of int
          start index of slice to cut from cube (z,y,x)
        sh: 3-tuple of int
          shape of cube to cut (z,y,x)
        number: int
          consecutive number for the save name (i.e. hours, iterations etc.)
        export_class: str or int
          'all' writes images of all classes, otherwise only the class with
          index ``export_class`` (int) is saved.
        """

        if self.model.prediction_node.shape.ndim >= 2:
            pred_node = self.model.prediction_node
        elif "pred_dense" in self.model.nodes:
            pred_node = self.model['pred_dense']
        else:
            raise RuntimeError("Model have spatial prediction node or"
                               " 'pred_dense' node which is spatial")
        if self.model.ndim==3:
            min_z = self.model.prediction_node.input_nodes[0].shape['z']
            if min_z > sh[0]:
                sh = list(sh)
                sh[0] = min_z
        elif self.model.ndim==2:
            pass
        else:
            raise RuntimeError("Model must be 2/3 dimensional for previews")

        raw_img = self.data.train_d[cube_i]
        raw_img = raw_img[:,
                          off[0]:off[0]+sh[0],
                          off[1]:off[1]+sh[1],
                          off[2]:off[2]+sh[2]]

        self.predict_and_write(pred_node, raw_img, number, export_class)
        self.saved_raw_preview = True


    def preview_slice(self, number=0, export_class='all', max_z_pred=5):
        """
        Predict and and save a data from a separately loaded file as preview

        Parameters
        ----------

        number: int/float
          consecutive number for the save name (i.e. hours, iterations etc.)
        export_class: str or int
          'all' writes images of all classes, otherwise only the class with
          index ``export_class`` (int) is saved.
        max_z_pred: int
          approximate maximal number of z-slices to produce (depends on CNN architecture)
        """

        assert self.preview_data is not None, "You must provide preview data in order to call this function"
        for example_no,raw_img in enumerate(self.preview_data):
            if raw_img.ndim==3:
                if raw_img.shape[0]>raw_img.shape[2]:
                    raw_img = np.transpose(raw_img, (2,0,1))
                    logger.warning("preview_slice: transposing preview image, assuming last dim is z because "
                    " this dim is smaller than the first.")

            z_sh = raw_img.shape[0] if raw_img.ndim==3 else raw_img.shape[1]

            if self.model.prediction_node.shape.ndim>=2:
                pred_node =  self.model.prediction_node
            elif "pred_dense" in self.model.nodes:
                pred_node = self.model['pred_dense']
            else:
                raise RuntimeError("Model have spatial prediction node or"
                                   " 'pred_dense' node which is spatial")
            if pred_node.shape.ndim==3:
                strd_z = pred_node.shape.strides[0]
                out_z  = pred_node.shape.spatial_shape[0] * strd_z
                min_z  = pred_node.input_nodes[0].shape.spatial_shape[0] + strd_z - 1 # input shape
                z_thick = min_z if out_z > max_z_pred else min_z + strd_z*int(np.ceil(float(max_z_pred-out_z)/strd_z))
            elif pred_node.shape.ndim==2:
                z_thick = max_z_pred
            else:
                raise RuntimeError("Model must be 2/3 dimensional for previews")


            if z_thick > z_sh:
                raise ValueError("The preview slices are too small in z-direction for this CNN")

            if raw_img.ndim==3:
                raw_img = raw_img[None, (z_sh-z_thick)//2:(z_sh-z_thick)//2+z_thick, :, :]
            elif raw_img.ndim==4:
                raw_img = raw_img[:, (z_sh-z_thick)//2:(z_sh-z_thick)//2+z_thick, :, :]

            self.predict_and_write(pred_node, raw_img, number, export_class, example_no, max_z_pred)

        self.saved_raw_preview = True

###############################################################################

class TracingTrainer(Trainer):
    @staticmethod
    def save_batch(img, lab, k, lab_img=None):
        img = img[0]
        lab = lab[0]
        off = img.shape[1] - lab.shape[1]
        utils.h5save(img, 'img-%i.h5' % k)
        if lab_img is not None:
            utils.h5save(lab_img, 'lab_img-%i.h5' % k)

        # assert off % 2==0
        # off //= 2
        with FileLock('plotting'):
            for i in range(img.shape[1]):
                plt.imsave('batch-%i-z%i.png' % (k, i), img[0, i], cmap='gray')
            # if 0 <= (i - off) < lab.shape[1]:
            #     lab_small = lab[4, i - off]
            #     lab_up = misc.imresize(lab_small,
            #                            np.multiply(lab_small.shape, 8),
            #                            interp='nearest')
            #     plt.imsave('batch-%i-z%i-l.png' % (k, i), lab_up, cmap='gray')


    # def probmap_preview(self, raw_img, number=0, block_name=''):
    #     """
    #     Predict and and save a slice as preview image
    #
    #     Parameters
    #     ----------
    #
    #     raw_img : np.ndarray
    #       raw data in the format (ch, x, y, z)
    #     number: int/float
    #       consecutive number for the save name (i.e. hours, iterations etc.)
    #     block_name: str
    #       Name/number to distinguish different raw_imges
    #     """
    #     block_name = str(block_name)
    #     pred = self.model['pred_dense'].predict_dense(raw_img)  # returns (k, z, y, x)
    #
    #     save_name = self.exp_config.save_name
    #     #names = ['vz', 'vx', 'vy', 'br', 'barr', 'bg', 'syn', 'ves', 'mito']
    #     for z in range(pred.shape[1]):
    #         for c in range(pred.shape[0]):
    #             plt.imsave('%s-pred-%s-z%i-c%i-%shrs.png' \
    #                        % (save_name, block_name, z, c, number),
    #                        pred[c, z, :, :], cmap='gray')
    #
    #     z_off = int(self.model['vec'].shape.offsets[0])
    #     for z in range(pred.shape[1]):
    #         plt.imsave('%s-raw-%s-z%i.png' % (save_name, block_name, z),
    #                    raw_img[0, z + z_off, :, :], cmap='gray')


    def debug_getcnnbatch(self, extended=False):
        """
        Executes ``getbatch`` but with un-strided labels and always returning
        info. The first batch example is plotted and the whole batch is
        returned for inspection.
        """
        kwargs = dict(self.get_batch_kwargs)
        kwargs['force_dense'] = True
        batch = self.data.getbatch(**kwargs)

        data, target = batch[0][0], batch[1][0]
        target[np.isclose(target, -666)] = 0
        if self.model.ndim==2:
            if target.shape[0] >= 3:
                target = np.transpose(target, (1, 2, 0))[:, :, :3]
                target = target[...,[2, 0, 1]]
            else:
                target = target[0]
            plot_trainingtarget(data[0], target, 1)
        else:
            t_i = target.shape[1] // 2
            if target.shape[0] >= 3:
                target = np.transpose(target, (1, 2, 3, 0))[0, :, :, :3]
                target = target[...,[2, 0, 2]]
            else:
                target = target[0, t_i]

            i = self.data.offsets[0]  # z offset
            plot_trainingtarget(data[0, i+t_i], target, 1)


        if self.model.ndim==3 and extended:
            dest = '/tmp/%s_' % user_name
            data, target = batch[0], batch[1]
            target[np.isclose(target, -666)] = 0
            i = self.data.offsets[0]  # z offset
            with FileLock('plotting'):
                for j in range(data.shape[2]):
                    plt.imsave('/tmp/%s_img-%i.png' % (user_name, j), data[0, 0, j], cmap='gray')
                    if j - i >= 0 and j - i < target.shape[2]:
                        plt.imsave(dest+'img-%i-br.png'%j, target[0, 4, j - i],cmap='gray')
                        plt.imsave(dest+'img-%i-z.png'%j, target[0,0,j-i], cmap='gray')
                        plt.imsave(dest+'img-%i-y.png'%j, target[0,1,j-i], cmap='gray')
                        plt.imsave(dest+'img-%i-x.png'%j, target[0,2,j-i], cmap='gray')
                        plt.imsave(dest+'img-%i-barr.png'%j, target[0,3,j-i], cmap='gray')

                        plt.imsave(dest+'img-%i-syn.png'%j, target[0,6,j-i], cmap='gray')
                        plt.imsave(dest+'img-%i-ves.png'%j, target[0,7,j-i], cmap='gray')
                        plt.imsave(dest+'img-%i-mito.png'%j, target[0,8,j-i], cmap='gray')

                        quiver = my_quiver(target[0,2,j-i], target[0,1,j-i],
                                  img=target[0, 4, j - i], c=target[0,0,j-i])

                        quiver.savefig(dest+'vec-%i.png'%j, bbox_inches='tight')

        with FileLock('plotting'):
            plt.savefig('Batch_test_image.png', bbox_inches='tight')

        if config.gui_plot:
            plt.ion()
            plt.show()
            plt.pause(0.01)
            plt.pause(2.0)
            plt.close('all')
            plt.pause(0.01)

        return batch

    def run(self):
        exp_config = self.exp_config
        save_name = exp_config.save_name
        data = self.data
        self.tracker.register_debug_output_names(self.model.debug_output_names[1:]) #remove first because it is prediction

        t_passed = 0
        t_pt = 2
        t_pi = 2
        last_save_t = 0
        last_save_t2 = 0
        save_time = config.param_save_h
        save_time2 = config.initial_prev_h

        loss, loss_smooth, train_loss, valid_loss, train_error, valid_error, param_vars = 0, 0, 0, 0, 0, 0, 0
        user_termination = False

        if isinstance(self.model.loss_node.parent[0], SquaredLoss):
            is_regression = True
        else:
            is_regression = False
        pp_err = 'err' if is_regression else '%'

        # --------------------------------------------------------------------------------------------------------
        if config.background_processes:
            n_proc = max(2, int(config.background_processes))
            bg_worker = BackgroundProc(data.getbatch, n_proc=n_proc,
                                       target_kwargs=self.get_batch_kwargs)
        # --------------------------------------------------------------------------------------------------------
        try:
            lost_track = True
            tracing_length = 1

            i = -1
            t0 = time.time()
            while i < exp_config.n_steps:
                # update max every loop to make it modifiable during training
                max_tracing = exp_config.sequence_training if exp_config.sequence_training>1 else 50
                try:
                    if config.inspection:
                        inspection_logger.info("#BATCH %i" % (i + 1))

                    # check if we lost the skeleton track
                    if exp_config.sequence_training:
                        try:
                            lost_track= batch[2].lost_track
                        except UnboundLocalError:
                            pass
                        if tracing_length >= max_tracing:
                            lost_track = True

                    # if we are still on track get a new slice from this skeleton
                    if not lost_track and exp_config.sequence_training:
                        tracing_length  += 1
                        position_l, direction_il = batch[3].cnn_pred2lab_position(
                            prediction_c)
                        try:
                            tmp = data.get_newslice(position_l, direction_il,
                                                           **self.get_batch_kwargs)
                            img, vec, trafo = tmp[:3]
                            if len(tmp)==4:
                                batch = (img, vec, batch[2], trafo, tmp[3])
                            else:
                                batch = (img, vec, batch[2], trafo)
                        # if get_newslice fails, do same as when track is lost
                        except transformations.WarpingOOBError:
                            lost_track = True

                    if lost_track and exp_config.sequence_training:
                        print("Traced for %i iterations" %(tracing_length,))
                        tracing_length = 0
                        try:
                            skel = batch[2]
                            skel.debug_traces.append(np.array(skel.debug_traces_current))
                            skel.debug_traces_current = []
                            skel.debug_grads.append(np.array(skel.debug_grads_current))
                            skel.debug_grads_current = []
                        except UnboundLocalError:
                            pass

                    # non-sequence training or getting a new skeleton
                    if lost_track or (not exp_config.sequence_training):
                        if config.background_processes:
                            batch = bg_worker.get()
                        else:
                            batch = data.getbatch(**self.get_batch_kwargs)

                        if exp_config.sequence_training:
                            #print("Next skel: %i" % (batch[2],))
                            pass

                        batch = list(batch)
                        batch[2] = data.train_s[batch[2]]
                        batch[3] = transformations.trafo_from_array(batch[3][0])

                    if config.inspection:
                        lab_img = batch[4]
                        batch = batch[:4]
                        if (i+1) % 50==0:
                            self.save_batch(batch[0], batch[1], (i + 1), lab_img)

                    # -----------------------------------------------------------------------------------------------------
                    loss, t_per_train, debug_outputs = self.model.trainingstep(*batch, optimiser=exp_config.optimiser)  # Update step
                    prediction_c = debug_outputs[0][0]
                    debug_outputs = debug_outputs[1:]
                    i += 1
                    # -----------------------------------------------------------------------------------------------------
                    t_per_it = time.time() - t0
                    t_passed += t_per_it
                    t0 = time.time()
                    t_pi = 0.8 * t_pi + 0.2 * t_per_it  # EMA
                    loss_smooth = self.model.loss_smooth


                    # check for divergence
                    if np.any(np.isnan(loss)) or np.any(np.isinf(loss)):
                        logger.warning(NLL_TEXT)
                        raise KeyboardInterrupt
                        # self.model.optimisers[exp_config.optimiser].repair()

                    self.tracker.update_timeline([t_passed, loss, debug_outputs[0]/10])
                    if debug_outputs:
                        self.tracker.update_debug_outputs([i, loss, ] + debug_outputs)

                    # Save Parameters
                    # if (t_passed-last_save_t)/3600 > config.param_save_h:
                    #     last_save_t = t_passed
                    #     time_string = '-'+str(save_time)+'h'
                    #     self.model.save(os.path.join('Backup', save_name+time_string+'.mdl'))
                    #     save_time += config.param_save_h
                    if i%config.param_save_it==0 and i>0:
                        it_string = '-'+str(i//1000)+'k'
                        self.model.save(os.path.join('Backup', save_name+it_string+'.mdl'))

                    # Create preview prediction images
                    if self.preview_data is not None:
                        if (t_passed - last_save_t2) / 3600 > config.prev_save_h \
                           or (t_passed / 3600 > config.initial_prev_h
                           and last_save_t2==0):  # first time
                            last_save_t2 = t_passed
                            exp_config.preview_kwargs['number'] = save_time2
                            save_time2 += config.prev_save_h
                            try:
                                self.preview_slice(**exp_config.preview_kwargs)
                            except:
                                logger.warning("Preview Predictions failed."
                                               "Are the preview raw data in "
                                               "the correct format?")
                            # reset time because we only count training time
                            # not time spent for previews (making previews
                            # is not a computational payload of the actual
                            # training but just for "fun")
                            t0 = time.time()

                    # Adjust the learning rate and other schedule parameters
                    for schedule in self.schedules.values():
                        if i==schedule.next_update:
                            schedule.update(i)

                    if (i % exp_config.history_freq==0) \
                        and exp_config.history_freq!=0:
                        lr = self.model.lr
                        mom = self.model.mom
                        if len(self.model.gradnet_rates):
                            gradnetrate = np.mean(self.model.gradnet_rates)
                        else:
                            gradnetrate = 0

                        ### Training  & Valid Errors ###
                        loss_after = self.model.loss(*batch)
                        loss_gain = loss_after - loss
                        train_loss, train_error = self.test_model('train')
                        valid_loss, valid_error = self.test_model('valid')
                        self.tracker.update_history([i, t_passed, train_loss,
                                                     valid_loss, loss_gain,
                                                     train_error, valid_error,
                                                     lr, mom, gradnetrate])

                        ### Plotting / Saving ###
                        self.model.save(save_name + '-LAST.mdl')
                        self.tracker.save(os.path.join('Backup', save_name))
                        if config.plot_on and ((i>=exp_config.history_freq*3) or i>60):
                            self.tracker.plot(save_name)

                        if config.print_status:
                            t = utils.pretty_string_time(t_passed)
                            out = "%05i L_m=%.3f, L=%.2f, tr=%05.2f%s, " % (i,
                                                                            loss_smooth,
                                                                            loss,
                                                                            train_error,
                                                                            pp_err)
                            out += "vl=%05.2f%s, prev=%04.1f, L_diff=%+.1e, " \
                                   % (valid_error, pp_err, debug_outputs[0] * 100,
                                      loss_gain)
                            out += "LR=%.5f, %.2f it/s, %s" % (
                            lr, 1.0 / t_pi, t)
                            logger.info(out)

                # User Interface ##############################################
                except (KeyboardInterrupt, ValueError, TypeError) as e:
                    if not isinstance(e, KeyboardInterrupt):
                        traceback.print_exc()
                        print(
                            "\nEntering Command line such that Exception can be "
                            "further inspected by user.\n\n")

                    out = "%05i L_m=%.5f, L=%.4f, train=%.5f, valid=%.5f, " % (
                    i,
                    loss_smooth, loss, train_loss, valid_loss)
                    out += "train=%.3f%s, valid=%.3f%s,\n LR=%.6f, MOM=%.6f, " \
                           % (
                           train_error, pp_err, valid_error, pp_err, self.model.lr,
                           self.model.mom)
                    out += "%.1f GPU-it/s, %.1f CPU-it/s, " % (
                    1.0 / self.model.time_per_step,
                    1.0 / t_pi)

                    t = utils.pretty_string_time(t_passed)
                    logger.info(out + t)

                    # Like a command line, but cannot change singletons
                    var_push = globals()
                    var_push.update(locals())
                    ret = trainutils.user_input(var_push)
                    if ret=='kill':
                        user_termination = True

                    if config.background_processes:
                        bg_worker.reset()

                    plt.close('all')
                    # reset time after user interaction, otherwise time
                    # will appear as pause in plot
                    t0 = time.time()

                    # End UI ##################################################

                # This is in the epoch/UI loop
                if (t_passed > exp_config.max_runtime) or user_termination:
                    logger.info('Timeout or manual Termination')
                    break

            # This is OUTSIDE the training loop i.e.
            # the last block of the function ``run``
            self.model.save(save_name + "-FINAL.mdl")
            if len(self.tracker.timeline) > 10:
                self.tracker.plot(save_name)
            logger.info('End of Training')
            logger.info('#' * 60 + '\n' + '#' * 60 + '\n')
            # -------------------end of run()----------------------------------

        except:
            sys.excepthook(*sys.exc_info())  # show info on error
        finally:
            if config.background_processes:
                bg_worker.shutdown()

        if self.model.batch_normalisation_active:
            self.model = rebuild_model(self.model, replace_bn='const')

    def test_model(self, data_source):
        """
        Computes Loss and error/accuracy on batch with ``monitor_batch_size``


        Parameters
        ----------

        data_source: string
            'train' or 'valid'

        Returns
        -------
        Loss, error:

        """
        assert self.batch_size==1
        # copy because it is modified in next line!
        kwargs = dict(self.get_batch_kwargs)
        kwargs['source'] = data_source
        kwargs['batch_size'] = self.exp_config.monitor_batch_size
        try:
            batch = self.data.getbatch(**kwargs)
        except ValueError:
            logger.warning("Test model, getbatch failed. No validation data?")
            return np.nan, np.nan  # 0, 0

        batch = list(batch)
        batch[2] = list(batch[2])
        batch[3] = list(batch[3])
        for i in range(self.exp_config.monitor_batch_size):
            if data_source=='train':
                batch[2][i] = self.data.train_s[batch[2][i]]
            elif data_source=='valid':
                batch[2][i] = self.data.valid_s[batch[2][i]]

            batch[3][i] = transformations.trafo_from_array(batch[3][i])

        if config.inspection:
            batch = batch[:4]

        rates = self.model.dropout_rates
        self.model.dropout_rates = ([0.0, ] * len(rates))
        n = len(batch[0])
        loss = 0
        error = 0
        for j in range(n):
            d = batch[0][j:j+1]  # data
            l = batch[1][j:j+1]  # target
            if len(batch) > 2:
                aux = []
                for b in batch[2:]:
                    aux.append(b[j])

                nl, er, pred = self.model.predict_ext(d, l, *aux)
                skel = batch[2][j] # predict_ext calls get_loss_and_gradient
                # which adds current position and grad, but for testing model
                # we do not want these tracked --> remove again
                skel.debug_traces_current.pop()
                skel.debug_grads_current.pop()
            else:
                nl, er, pred = self.model.predict_ext(d, l)

            loss += nl * len(d)
            error += er * len(d)

        loss /= n
        error /= n
        self.model.dropout_rates = rates  # restore old rates
        return loss, error


###############################################################################

class TracingTrainerRNN(TracingTrainer):
    def run(self):
        exp_config = self.exp_config
        save_name = exp_config.save_name
        data = self.data
        if 'scan_out_radius_t'==self.model.debug_output_names[-1]:
            assert 'scan_out_radius'==self.model.debug_output_names[-2]
            print("Found regression targets for scan")
            self.tracker.register_debug_output_names(self.model.debug_output_names[:-2])

        mem_hid = np.zeros(self.model['mem_hid'].shape, dtype=floatX)

        t_passed = 0
        t_pt = 2
        t_pi = 2
        last_save_t = 0
        last_save_t2 = 0
        save_time = config.param_save_h
        save_time2 = config.initial_prev_h

        loss, loss_smooth, train_loss, valid_loss, train_error, valid_error, param_vars = 0, 0, 0, 0, 0, 0, 0
        user_termination = False

        is_regression = True
        pp_err = 'err'

        # --------------------------------------------------------------------------------------------------------
        try:
            lost_track = True
            tracing_length = 0

            i = -1
            t0 = time.time()
            skel_example = None
            while i < exp_config.n_steps:
                # update max every loop to make it modifiable during training
                max_tracing = exp_config.sequence_training
                try:
                    if skel_example is None:
                        skel_example, skel_index = data.getskel('train')
                        if len(self.model.loss_node.input_nodes)==2:
                            batch = (skel_example, mem_hid)
                        else:
                            batch = (skel_example,)
                        skel_example.start_new_training = True
                        tracing_length = 0
                        if config.inspection:
                            inspection_logger.info("NEW SKEL %i" %skel_index)

                    # -----------------------------------------------------------------------------------------------------
                    try:
                        loss, t_per_train, debug_outputs = self.model.trainingstep(*batch, optimiser=exp_config.optimiser)  # Update step
                        tracing_length += 1
                        i += 1
                        if config.inspection:
                            inspection_logger.info("-"*20)
                    except transformations.WarpingOOBError:
                        if config.inspection:
                            inspection_logger.info("OOB, Traced for %i iterations" % (tracing_length,))
                        skel_example = None
                        continue
                    # -----------------------------------------------------------------------------------------------------
                    t_per_it = time.time() - t0
                    t_passed += t_per_it
                    t0 = time.time()
                    t_pi = 0.8 * t_pi + 0.2 * t_per_it  # EMA
                    loss_smooth = self.model.loss_smooth

                    if skel_example.lost_track or tracing_length >= max_tracing:
                        if config.inspection:
                            inspection_logger.info("Traced for %i iterations" % (tracing_length,))
                        skel_example = None

                    # check for divergence
                    if np.any(np.isnan(loss)) or np.any(np.isinf(loss)):
                        logger.warning(NLL_TEXT)
                        raise KeyboardInterrupt
                        # self.model.optimisers[exp_config.optimiser].repair()

                    self.tracker.update_timeline([t_passed, loss, 0])
                    if debug_outputs:
                        if 'scan_out_radius_t'==self.model.debug_output_names[-1]:
                            r = debug_outputs[-2]
                            r_t = debug_outputs[-1]
                            debug_outputs = debug_outputs[:-2]
                            self.tracker.update_regression(r.ravel(), r_t.ravel())

                        debug_outputs_ = [np.mean(x) for x in debug_outputs]
                        self.tracker.update_debug_outputs([i, loss, ] + debug_outputs_)

                    # Save Parameters
                    # if (t_passed-last_save_t)/3600 > config.param_save_h:
                    #     last_save_t = t_passed
                    #     time_string = '-'+str(save_time)+'h'
                    #     self.model.save(os.path.join('Backup', save_name+time_string+'.mdl'))
                    #     save_time += config.param_save_h
                    if i%config.param_save_it==0 and i>0:
                        it_string = '-'+str(i//1000)+'k'
                        self.model.save(os.path.join('Backup', save_name+it_string+'.mdl'))

                    # Create preview prediction images
                    if self.preview_data is not None:
                        if (t_passed - last_save_t2) / 3600 > config.prev_save_h \
                           or (t_passed / 3600 > config.initial_prev_h
                           and last_save_t2==0):  # first time
                            last_save_t2 = t_passed
                            exp_config.preview_kwargs['number'] = save_time2
                            save_time2 += config.prev_save_h
                            try:
                                self.preview_slice(**exp_config.preview_kwargs)
                            except:
                                logger.warning("Preview Predictions failed."
                                               "Are the preview raw data in "
                                               "the correct format?")
                            # reset time because we only count training time
                            # not time spent for previews (making previews
                            # is not a computational payload of the actual
                            # training but just for "fun")
                            t0 = time.time()

                    # Adjust the learning rate and other schedule parameters
                    for schedule in self.schedules.values():
                        if i==schedule.next_update:
                            schedule.update(i)

                    if (i % exp_config.history_freq==0) \
                        and exp_config.history_freq!=0:
                        lr = self.model.lr
                        mom = self.model.mom
                        if len(self.model.gradnet_rates):
                            gradnetrate = np.mean(self.model.gradnet_rates)
                        else:
                            gradnetrate = 0

                        ### Training  & Valid Errors ###
                        loss_after = loss # too expensive to really compute for RNN
                        loss_gain = loss_after - loss
                        train_loss, train_error = self.test_model('train')
                        valid_loss, valid_error = self.test_model('valid')
                        self.tracker.update_history([i, t_passed, train_loss,
                                                     valid_loss, loss_gain,
                                                     train_error, valid_error,
                                                     lr, mom, gradnetrate])

                        ### Plotting / Saving ###
                        self.model.save(save_name + '-LAST.mdl')
                        self.tracker.save(os.path.join('Backup', save_name))
                        if config.plot_on and ((i>=exp_config.history_freq*3) or i>60):
                            self.tracker.plot(save_name)

                        if config.print_status:
                            t = utils.pretty_string_time(t_passed)
                            out = "%05i L_m=%.3f, L=%.2f, tr=%05.2f%s, "% (i,
                            loss_smooth, loss, train_error, pp_err)
                            out += "vl=%05.2f%s, prev=%04.1f, L_diff=%+.1e, " \
                                   % (valid_error, pp_err, 0 * 100,
                                      loss_gain)
                            out += "LR=%.5f, %.2f it/s, %s" % (
                            lr, 1.0 / t_pi, t)
                            logger.info(out)

                # User Interface ##############################################
                except (KeyboardInterrupt, ValueError, TypeError) as e:
                    if not isinstance(e, KeyboardInterrupt):
                        traceback.print_exc()
                        print(
                            "\nEntering Command line such that Exception can be "
                            "further inspected by user.\n\n")

                    out = "%05i L_m=%.5f, L=%.4f, train=%.5f, valid=%.5f, " % (
                    i,
                    loss_smooth, loss, train_loss, valid_loss)
                    out += "train=%.3f%s, valid=%.3f%s,\n LR=%.6f, MOM=%.6f, " \
                           % (
                           train_error, pp_err, valid_error, pp_err, self.model.lr,
                           self.model.mom)
                    out += "%.1f GPU-it/s, %.1f CPU-it/s, " % (
                    1.0 / self.model.time_per_step,
                    1.0 / t_pi)

                    t = utils.pretty_string_time(t_passed)
                    logger.info(out + t)

                    # Like a command line, but cannot change singletons
                    var_push = globals()
                    var_push.update(locals())
                    ret = trainutils.user_input(var_push)
                    if ret=='kill':
                        user_termination = True

                    plt.close('all')
                    # reset time after user interaction, otherwise time
                    # will appear as pause in plot
                    t0 = time.time()

                    # End UI ##################################################

                # This is in the epoch/UI loop
                if (t_passed > exp_config.max_runtime) or user_termination:
                    logger.info('Timeout or manual Termination')
                    break

            # This is OUTSIDE the training loop i.e.
            # the last block of the function ``run``
            self.model.save(save_name + "-FINAL.mdl")
            if len(self.tracker.timeline) > 10:
                self.tracker.plot(save_name)
            logger.info('End of Training')
            logger.info('#' * 60 + '\n' + '#' * 60 + '\n')
            # -------------------end of run()----------------------------------

        except:
            sys.excepthook(*sys.exc_info())  # show info on error
        finally:
            pass

        # if self.model.batch_normalisation_active:
        #     self.model = rebuild_model(self.model, replace_bn='const')



    def test_model(self, data_source):
        #return 1.0, 0.5
        """
        Computes Loss and error/accuracy on batch with ``monitor_batch_size``


        Parameters
        ----------

        data_source: string
            'train' or 'valid'

        Returns
        -------
        Loss, error:

        """
        try:
            skel_example, skel_index = self.data.getskel(data_source)
        except ValueError:
            logger.warning("Test model, getbatch failed. No validation data?")
            return np.nan, np.nan  # 0, 0

        rates = self.model.dropout_rates
        self.model.dropout_rates = ([0.0, ] * len(rates))

        n = self.exp_config.monitor_batch_size
        loss = 0
        j = 0
        while j < n:
            #print(j, end="")
            try:
                skel_example, skel_index = self.data.getskel(data_source)
                skel_example.start_new_training = True
                nl = self.model.loss(skel_example)
            except transformations.WarpingOOBError:
                continue
            j += 1
            loss += nl

        loss /= n
        self.model.dropout_rates = rates  # restore old rates
        return loss, loss
