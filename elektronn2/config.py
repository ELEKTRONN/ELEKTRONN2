# -*- coding: utf-8 -*-
# ELEKTRONN2 Toolkit
# Copyright (c) 2015 Marius F. Killinger
# All rights reserved
from __future__ import absolute_import, division, print_function
from builtins import filter, hex, input, int, map, next, oct, pow, range, \
    super, zip

import datetime
import os
import socket
import logging
import numpy as np

from .utils import gpu


logger = logging.getLogger('elektronn2log')

host_name = socket.gethostname()
if host_name in ['synapse03', 'synapse04', 'synapse05', 'synapse06',
                 'synapse07', 'synapse08']:
    cuda_root = "/usr/local/centos-cuda/cuda-6.5"
    logger.info("On Host %s: setting cuda root to %s and disabling DNN!" % (
        host_name, cuda_root))

    import theano

    theano.config.cuda.root = cuda_root
    theano.config.dnn.enabled = "False"

__all__ = ['config', 'change_logging_file']

np.set_printoptions(precision=3, linewidth=90)


def change_logging_file(logger_, save_path, file_name='elektronn2.log'):
    old_lfile_handler = \
        [x for x in logger_.handlers if isinstance(x, logging.FileHandler)][0]

    logger_.removeHandler(old_lfile_handler)

    if not os.path.exists(save_path):
        os.mkdir(save_path)

    lfile_formatter = logging.Formatter(
        '[%(asctime)s] [%(levelname)s]\t%(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')
    lfile_path = os.path.join(save_path, file_name)
    lfile_level = logging.DEBUG
    lfile_handler = logging.FileHandler(lfile_path)
    lfile_handler.setLevel(lfile_level)
    lfile_handler.setFormatter(lfile_formatter)
    logger_.addHandler(lfile_handler)


class DefaultConfig(object):
    def __init__(self):
        """
        This class hard-codes the distribution-wide default values
        """
        ### Toolkit Setup ###------------------------------------------------------
        # (*) <String>: where to create the CNN directory.
        # In this directory a new folder is created with the name of the model
        self.save_path = os.path.abspath(os.path.expanduser('~/elektronn2_training/'))
        self.plot_on = True  # <Bool>: whether to create plots of the errors etc.
        self.print_status = True  # <Bool>: whether to print Training status to std.out
        self.device = None  # None/int/'auto' (use .theanorc value) or int (use gpu<i>)
        self.param_save_h = 1.0  # hours: frequency to save a permanent parameter snapshot
        self.param_save_it = 10000
        self.initial_prev_h = 1.0  # hours: time after which first preview is made
        self.prev_save_h = 3.0  # hours: frequency to create previews
        self.overwrite = True  # <Bool>: whether to delete/overwrite existing directory
        # <Bool>/<Int>: whether to "pre-fetch" batches in separate background
        # process, <Bool> or number of processes (True-->2)
        self.background_processes = False
        self.time_per_step_smoothing_length = 50
        self.loss_smoothing_length = 200  #
        self.use_manual_cudnn_conv = True
        self.use_manual_cudnn_conv_not_w1 = True
        self.use_manual_cudnn_pool = True
        self.allow_floatX_downcast = True
        self.show_axis_order_warning = True
        self.use_ortho_init = False
        # Flag used to conditionally do something different for inspection/debugging
        self.inspection = False
        # Whether to create a backup of the current source code in the save directory
        self.backupsrc = True
        # Whether to display some initial plots in a pop-up GUI.
        # TODO: This needs to be cleaned up. Plotting outputs (files, GUI) should be more distinguished globally.
        #       (Currently, several plotting functions try to both write pngs and open GUIs all over the place).
        self.gui_plot = False
        self.__doc__ = ""  # Just a hack

        self.read_user_config()

        # Init GPU stuff
        if self.device or self.device==0:
            if self.device=='auto':
                gpu_num = gpu.get_free_gpu(wait=2.0)
                if gpu_num < 0:
                    raise RuntimeError("Could not find free GPU.")

            else:
                assert isinstance(self.device, int)
                gpu_num = self.device

            gpu.initgpu(gpu_num)

        change_logging_file(logger, self.save_path, file_name='elektronn2.log')

    def read_user_config(self):
        config_dict = dict()
        user_path = os.path.abspath(os.path.expanduser('~/.elektronn2rc'))
        if not os.path.exists(user_path):
            logger.debug("No user config file at %s, using default values" % (
                user_path,))
        else:
            try:
                logger.debug("Reading user DefaultConfig from ~/.elektronn2rc")
                exec (compile(open(user_path).read(), user_path, 'exec'), {},
                      config_dict)
            except Exception as e:
                raise RuntimeError("The user config file %s does exist, "
                                   "but an error happened during reading, it might contain "
                                   "invalid code. Error: \n  %s" % (
                                       user_path, e))

            for key in config_dict:
                if key in ['save_path']:
                    config_dict[key] = os.path.abspath(
                        os.path.expanduser(config_dict[key]))
                if key=='param_save_h':
                    raise DeprecationWarning(
                        "Don't use param_save_h, use param_save_it")
                setattr(self, key, config_dict[key])


config = DefaultConfig()
