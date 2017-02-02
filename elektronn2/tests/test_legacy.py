# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 18:19:00 2015

@author: Marius Felix Killinger
"""
from __future__ import absolute_import, division, print_function
from builtins import filter, hex, input, int, map, next, oct, pow, range, \
    super, zip

import os

import numpy as np
import matplotlib.pyplot as plt

from ..utils import legacy
from ..utils import pickleload, picklesave, h5load


config_file = os.path.expanduser(
    '~/CNN_Training/3D/BIRD_MEMBRANE_v51_1099/Backup/config.py')
param_file = os.path.expanduser(
    '~/CNN_Training/3D/BIRD_MEMBRANE_v51_1099/BIRD_MEMBRANE_v51_1099-LAST.param')

probs, gm = legacy.create_cnn(config_file, 1, param_file=param_file,
                              constant_weights=False)


# activations = []
# for node in graph_manager.nodes.values():
#    if not node.is_source:
#        activations.append(node)
#
# inp = graphutils.getinput_for_multioutput(activations)
# activations = [a.output for a in activations]
# activation_func = graphutils.make_func(inp, activations,
#                                                        name='Activations')

### ELEKTRONN
class Data(object):
    pass


def test_legacy():
    from elektronn.training.config import Config  # the global user-set config

    config = Config(config_file, None, None, use_existing_dir=True)
    from elektronn.training import trainer  # contains import of theano

    T = trainer.Trainer(config)
    T.data = Data()
    T.data.n_lab = 2
    T.data.n_ch = 1
    T.createNet()
    T.cnn.loadParameters(param_file)

    ### Create test data
    in_sh = list(T.cnn.input_shape)
    if in_sh[0] is None:
        in_sh[0] = 1

    img = h5load(os.path.expanduser(
        '~/devel/data/MPI/barrier_gt_4/cube0_barrier_raw.h5'))
    x, y, z = in_sh[1], in_sh[3], in_sh[4]
    img = img[:x, :y, :z]
    img = img.astype(np.float32) / 255
    img = img[None, :, None]
    x = img
    # x = np.random.rand(*in_sh).astype(np.float32)


    ### Test ELEKTRONN2
    # y = probs(x)
    # t0 = time.time()
    # y1 = probs(x)
    # t1 = time.time()
    # n = np.prod(y.shape[2:])
    # speed1 = float(n)/(t1-t0)/1e6
    #
    #### Test ELEKTRONN
    # t0 = time.time()
    # y2 = T.cnn.class_probabilities(x)
    # t1 = time.time()
    # n = np.prod(y.shape[2:])
    # speed2 = float(n)/(t1-t0)/1e6
    # y2 = np.swapaxes(y2, 1,2)
    #
    #
    #### Compare
    # plt.figure()
    # plt.gray()
    # plt.imshow(y1[0,1,0], interpolation='none')
    # plt.title('Neuro')
    # plt.figure()
    # plt.imshow(y2[0,1,0], interpolation='none')
    # plt.title('Elektronn')
    # assert np.allclose(y1, y2)
    # W2 = T.cnn.layers[0].W.get_value()
    # W1 = graph_manager.nodes[graph_manager.nodes.keys()[1]].W.get_value()

    ### Save elektronn2 graph
    print("=" * 50)
    print("=" * 50)
    picklesave(gm.serialise(), '/tmp/model.pkl')

    gm.reset()
    records = pickleload('/tmp/model.pkl')
    gm.restore(records, make_weights_constant=False)
    print("=" * 50)
    print("=" * 50)
    out = gm.sinks
    print(out)
    print(gm.nodes.keys())
    probs = gm.nodes['class_probabilities']
    y = probs(x)
    plt.figure()
    plt.gray()
    plt.imshow(y[0, 1, 0], interpolation='none')
    plt.title('Neuro')
