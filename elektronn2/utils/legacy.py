# -*- coding: utf-8 -*-
# ELEKTRONN2 Toolkit
# Copyright (c) 2015 Marius F. Killinger
# All rights reserved
from __future__ import absolute_import, division, print_function
from builtins import filter, hex, input, int, map, next, oct, pow, range, \
    super, zip

import numpy as np

from .. import neuromancer
from .utils_basic import pickleload


try:
    from elektronn.training.config import Config
except:
    Config = None
    print('Warning: elektronn not installed. create_cnn will fail!')


def create_cnn(config_file, n_ch, param_file=None, mfp=False,
               axis_order='theano', constant_weights=False,
               imposed_input_size=None):
    raise RuntimeError("Don't use this, rebuild the graph and import the "
                       "weights using load_params_into_model")

    config = Config(config_file, None, None, use_existing_dir=True,
                    override_MFP_to_active=mfp,
                    imposed_input_size=imposed_input_size)

    if config.mode!='img-img':
        raise NotImplementedError()

    if axis_order=='theano':
        ps = config.patch_size
        ndim = len(ps)

        input_size = [None, ] * (2 + ndim)
        input_size[0] = config.batch_size
        if ndim==3:
            tags = 'b,z,f,y,x'
            input_size[1] = config.patch_size[0]
            input_size[2] = n_ch
            input_size[3] = config.patch_size[1]
            input_size[4] = config.patch_size[2]
        elif ndim==2:
            tags = 'b,f,x,y'
            input_size[1] = n_ch
            input_size[2] = config.patch_size[0]
            input_size[3] = config.patch_size[1]

        if param_file is None:
            param_file = config.paramfile
        params = pickleload(param_file)
        pool = params[-1]
        f_shapes = params[0]
        params = params[1:-1]  # come in order W0, b0, W1, b1,...

        neuromancer.node_basic.model_manager.newmodel('legacy')
        inp = neuromancer.Input(input_size, tags)
        conv = list(
            zip(config.nof_filters,  # doesn't have to be a list, does it?
                config.filters, config.pool, config.activation_func,
                config.pooling_mode, params[::2], params[1::2]))
        for i, (n, f, p, act, p_m, W, b) in enumerate(conv):
            W = [W, 'const'] if constant_weights else W
            b = [b, 'const'] if constant_weights else b
            inp = neuromancer.Conv(inp, n, f, p, mfp=mfp, activation_func=act,
                                   w=W, b=b)

        # last Layer
        W = [params[-2], 'const'] if constant_weights else params[-2]
        b = [params[-1], 'const'] if constant_weights else params[-1]
        out = neuromancer.Conv(inp, config.n_lab, (1,) * ndim, (1,) * ndim,
                               activation_func='lin', w=W, b=b)
        if mfp:
            out = neuromancer.FragmentsToDense(out)

        if config.target in ['affinity', 'malis']:
            probs = neuromancer.Softmax(out, n_class=2, n_indep=3,
                                        name='class_probabilities')
        else:
            probs = neuromancer.Softmax(out, n_class=config.n_lab,
                                        name='class_probabilities')


    elif axis_order=='dnn':
        raise NotImplementedError()

    model = neuromancer.model_manager.getmodel('legacy')
    model.designate_nodes(input_node=inp, prediction_node=probs)

    return model


def load_params_into_model(param_file, model):
    """
    Loads parameters directly from save file into a graph manager (this requires
    that the graph is identical to the cnn from the param file
    :param param_file:
    :param gm:
    :return:
    """
    params = pickleload(param_file)[1:-1]

    i = 0
    for node in model.nodes.values():
        if hasattr(node, 'w'):
            try:
                w = np.transpose(params[i], (0, 2, 1, 3, 4))
                node.w.set_value(w)
            except:
                node.w.set_value(params[i])

            node.b.set_value(params[i + 1])
            i += 2
