#!/usr/bin/python
# -*- coding: utf-8 -*-
# ELEKTRONN2 Toolkit
# Copyright (c) 2015 Marius F. Killinger
# All rights reserved
from __future__ import absolute_import, division, print_function
from builtins import filter, hex, input, int, map, next, oct, pow, range, super, zip

import sys
import os
import traceback
import time
import argparse
import ast
from subprocess import check_call, CalledProcessError, call

import numpy as np
import matplotlib

sys.setrecursionlimit(10000)

with open('/tmp/fucker.txt', 'a') as f:
    f.write("I managed to start. Fuck this.")

def parseargs():
    def convert(s):
        if s=='auto':
            return s
        elif s.lower()=='false':
            return False
        elif s.lower()=='none':
            return None
        else:
            return int(s)

    def convert_sh(sh):
        return ast.literal_eval(sh)

    parser = argparse.ArgumentParser(
    usage="MemProfile </path/model_file> [--gpu={Auto|False|<int>}]")
    parser.add_argument("model_path", type=str)
    parser.add_argument("--in_sh", default=None, type=convert_sh)
    parsed = parser.parse_args()
    return parsed.model_path, parsed.in_sh

model_path, in_sh = parseargs()

# prevent setting of mpl qt-backend on machines without X-server before other
# modules import mpl #  Redirect to /dev/null because xset output is unimportant
with open(os.devnull, 'w') as devnull:
    try:
        # "xset q" will always succeed to run if an X server is currently running
        check_call(['xset', 'q'], stdout=devnull, stderr=devnull)
        print('X available')
        # Don't set backend explicitly, use system default...
    # if "xset q" fails, conclude that X is not running
    except (OSError, CalledProcessError):
        print('X unavailable')
        matplotlib.use('AGG')


from elektronn2.config import config
config.use_manual_cudnn_conv = True
config.use_manual_cudnn_conv_not_w1 = False
config.uuse_manual_cudnn_pool = True
if config.device is None:
    from elektronn2.utils.gpu import initgpu
    initgpu('auto')

from elektronn2.neuromancer.model import modelload, kernel_lists_from_node_descr

if model_path in [None, 'None']:
    model_path  = "~/axon/mkilling/investigation/MA-TEX/CNN-Timings/old_cnn_rec.mdl"

no_gc = False
mfp = False



model_path = os.path.expanduser(model_path)
model_dir, model_name = os.path.split(model_path)
os.chdir(model_dir)
f_name = model_name[:-4]+'-Speed.csv'


# Benchmark Model Compiled with static Shape #
if in_sh:
    model_static = modelload(model_path, override_mfp_to_active=mfp,
                             imposed_patch_size=in_sh[2:],
                             imposed_batch_size=1, make_weights_constant=True)
    try:
        val = np.random.rand(*in_sh).astype(np.float32)
        model_static.predict(val)
        t0 = time.time()
        for i in range(3):
            y = model_static.predict(val)

        t1 = time.time()
        n = np.prod(y.shape[2:])
        speed = float(n) / (t1 - t0) / 1e6 * 3
        n = np.prod(y.shape[2:])
        if len(in_sh)==5:
            z = in_sh[2]
            x = in_sh[3]
            s = '%i\t%i\t%i\t%f\t- static\n' % (z, x, n, speed)
        else:
            x = in_sh[2]
            s = '%i\t%i\t%f\t - static\n' % (x, n, speed)

        print(s)
        with open(f_name, 'a') as f:
            f.write(s)
    except:
        traceback.print_exc(file=sys.stdout)

    exit()

else:
    model_flexi = modelload(model_path, override_mfp_to_active=mfp,
                            imposed_batch_size='flexi', make_weights_constant=True)

    fov = model_flexi.prediction_node.shape.fov
    in_sh0 = model_flexi.input_node.shape.spatial_shape
    z_ratio = float(fov[-1])/fov[0]
    ndim = len(fov)
    fp = "~/devel/ELEKTRONN2/scripts/MemProfile.py"
    with open(f_name, 'a') as f:
        if ndim==3:
            f.write("z-shape\txy-shape\tpixel\tSpeed [MPix/s]\n")
        else:
            f.write("xy-shape\tpixel\tSpeed [MPix/s]\n")

    call_args = []
    try:
        if ndim==2:
            for x in range(in_sh0[-1],3000):
                in_sh = list(model_flexi.input_node.shape)
                in_sh[0] = 1
                in_sh[2] = x
                in_sh[3] = x
                print(in_sh)
                try:
                    val = np.random.rand(*in_sh).astype(np.float32)
                    y = model_flexi.predict(val)
                    t0 = time.time()
                    for i in range(10):
                        y = model_flexi.predict(val)

                    t1 = time.time()
                    n = np.prod(y.shape[2:])
                    speed = float(n) / (t1 - t0) / 1e6 * 10
                    s = '%i\t%i\t%f\n' % (x, n, speed)
                    print(s)
                    with open(f_name, 'a') as f:
                        f.write(s)

                    in_sh_str = repr(in_sh)
                    in_sh_str = in_sh_str.replace(' ', '')
                    call_args.append(in_sh_str)
                except (AssertionError, ValueError, RuntimeError) as e:
                    traceback.print_exc(file=sys.stdout)

        elif ndim==3:
            x = in_sh0[-1]+50
            x_incr = 1
            possible_z = np.arange(500)
            z_incr = 1
            while x < 2000:
                z_base = x/z_ratio
                z = possible_z[np.abs(z_base - possible_z).argmin()]
                for z_mod in range(-16,16,z_incr):
                    in_sh = list(model_flexi.input_node.shape)
                    in_sh[0] = 1
                    in_sh[2] = z + z_mod
                    in_sh[3] = x
                    in_sh[4] = x
                    print(in_sh)
                    try:
                        val = np.random.rand(*in_sh).astype(np.float32)
                        y = model_flexi.predict(val)
                        t0 = time.time()
                        for i in range(3):
                            y = model_flexi.predict(val)
                        t1 = time.time()
                        n = np.prod(y.shape[2:])
                        speed = float(n) / (t1 - t0) / 1e6 * 3
                        # z, x, static, speed
                        s = '%i\t%i\t%i\t%f\n' % (in_sh[2], x, n, speed)
                        print(s)
                        x_incr = 16
                        z_incr = 16
                        possible_z = np.arange(z,500,16)
                        with open(f_name, 'a') as f:
                            f.write(s)

                        in_sh_str = repr(in_sh)
                        in_sh_str = in_sh_str.replace(' ', '')
                        call_args.append(in_sh_str)
                    except (AssertionError, ValueError, RuntimeError) as e:
                        traceback.print_exc(file=sys.stdout)

                x += x_incr

    except MemoryError:
        print("Flexi has reached memory error")

    for in_sh_str in call_args[::-1]:
        if no_gc:
            call("python %s %s --in_sh=%s"%(fp, model_path, in_sh_str),
                 shell=True, executable="/bin/bash")
        else:
            call("THEANO_FLAGS='linker=cvm,allo_gc=True' python %s %s --in_sh=%s"%(fp, model_path, in_sh_str),
                 shell=True, executable="/bin/bash")
