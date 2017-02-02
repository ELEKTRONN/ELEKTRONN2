# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function
from builtins import filter, hex, input, int, map, next, oct, pow, range, \
    super, zip


def test_schedule_config():
    prev_save_h = 2.0

    save_name = 'Test'
    # save_path = None
    # model_load_path = None # alternative to above
    # model_load_args = None # to override mfp, inputsize etc
    # batch_size = None # try to infer from model itself unless it is flexible
    # preview_data_path = None
    # preview_kwargs = None
    data_class = 'CNNData'  # <String>: Name of Data Class in TrainData or <tuple>: (path_to_file, class_name)

    data_init_kwargs = dict(d_path='~/lustre/mkilling/BirdGT/j0126_new/',
                            l_path='~/lustre/mkilling/BirdGT/j0126_new',
                            d_files=[('raw_%i.h5' % i, 'raw') for i in
                                     range(1)],
                            l_files=[('barrier_int16_%i.h5' % i, 'lab') for i
                                     in range(1)], valid_cubes=[], n_target=2,
                            anisotropic_data=True, data_order='xyz',
                            affinity=None)

    # del i

    data_batch_args = dict(flip=True, grey_augment_channels=[0],
                           ret_ll_mask=False, warp_on=False, ignore_thresh=0.5)

    n_steps = 30000
    max_runtime = 1 * 3600  # in seconds
    history_freq = 500
    monitor_batch_size = 1

    optimiser = 'SGD'
    optimiser_params = dict(lr=0.001, mom=0.8, wd=0.1)
    lr_schedule = 0.995
    dropout_schedule = None
    gradnet_schedule = None
    class_weights = None
    lazy_labels = None

    batch_size = 1
    lr_schedule = dict(dec=0.95)
    wd_schedule = dict(lindec=[4000, 0.1])
    mom_schedule = dict(
        updates=[(500, 0.8), (1000, 0.7), (1500, 0.9), (2000, 0.2)])
    dropout_schedule = dict(updates=[(1000, [0.2, 0.2])])
    gradnet_schedule = None


def create_model():
    import numpy as np
    from elektronn2 import neuromancer
    from elektronn2.neuromancer import graph_manager
    from elektronn2.model import Model

    graph_manager.reset()

    inp = neuromancer.Input((1, 1, 1, 59, 59), 'b,f,x,y,z', name='raw')
    out = neuromancer.Conv(inp, 2, (1, 6, 6), (1, 2, 2), dropout_rate=0.8)
    out = neuromancer.Conv(out, 3, (1, 6, 6), (1, 2, 2), dropout_rate=0.5)
    out = neuromancer.Conv(out, 4, (1, 4, 4), (1, 1, 1))
    out = neuromancer.Conv(out, 5, (1, 1, 1), (1, 1, 1))
    out = neuromancer.Conv(out, 2, (1, 1, 1), (1, 1, 1), activation_func='lin')

    out = neuromancer.Softmax(out, n_indep=1, name='probs')

    l_sh = out.shape.copy()
    l_sh.updateshape('f', 1)
    l = neuromancer.Input_like(l_sh, dtype='int16', name='labels')
    loss_pix = neuromancer.MultinoulliNLL(out, l, target_is_sparse=True)
    loss = neuromancer.AggregateLoss(loss_pix, name='nll')

    graph_manager.designate_nodes(input=inp, target=l, loss=loss,
                                  prediction=out,
                                  prediction_ext=[loss, loss, out])
    m = Model(graph_manager)
    return m
