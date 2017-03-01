# -*- coding: utf-8 -*-
# ELEKTRONN2 Toolkit
# Copyright (c) 2015 Marius Killinger
# All rights reserved
# This Code is adapted from Sven Dorkenwald

from __future__ import absolute_import, division, print_function
from builtins import filter, hex, input, int, map, next, oct, pow, range, \
    super, zip

import sys
import subprocess
import time


def initgpu(gpu):
    if gpu is None:
        gpu = 'none'
    no_gpu = ['none', 'None']
    import theano.sandbox.cuda

    if theano.sandbox.cuda.cuda_available:
        if isinstance(gpu, str) and gpu.lower() == 'auto':
            gpu = int(get_free_gpu())
            print("Automatically assigned free GPU %i" % (gpu,))

        if gpu in no_gpu and gpu != 0:
            pass
        else:
            try:
                theano.sandbox.cuda.use("gpu" + str(gpu))
                print("Initialising GPU to %s" % gpu)
            except:
                sys.excepthook(*sys.exc_info())
                print("Failed to init GPU, argument not understood.")

    else:
        if gpu in no_gpu and gpu != 0:
            pass
        else:
            print("'--gpu' argument is not 'none' but CUDA is not available. "
                  "Falling back to CPU.")


def _check_if_gpu_is_free(nb_gpu):
    process_output = subprocess.Popen('nvidia-smi -i %d -q -d PIDS' % nb_gpu,
                                      stdout=subprocess.PIPE,
                                      shell=True).communicate()[0]
    if b"Process ID" in process_output and b"Used GPU Memory" in process_output:
        return 0
    else:
        return 1


def _get_number_gpus():
    process_output = subprocess.Popen('nvidia-smi -L', stdout=subprocess.PIPE,
                                      shell=True).communicate()[0].decode()
    nb_gpus = 0
    while True:
        if "GPU %d" % nb_gpus in process_output:
            nb_gpus += 1
        else:
            break
    return nb_gpus


def get_free_gpu(wait=0, nb_gpus=-1):
    import theano.sandbox.cuda

    if not theano.sandbox.cuda.cuda_available:
        print("Cannot get free gpu. Cuda not available on this "
              "host")
        return -1
    if nb_gpus==-1:
        nb_gpus = _get_number_gpus()
    while True:
        for nb_gpu in range(nb_gpus):
            if _check_if_gpu_is_free(nb_gpu)==1:
                return nb_gpu
        if wait > 0:
            time.sleep(2)
        else:
            return -1
