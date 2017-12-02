# -*- coding: utf-8 -*-
# TODO: Rewrite this to use publicly available data (e.g. neuro_data).
save_path = '~/CNN_Training/2D/'
preview_data_path = '~/lustre/mkilling/BirdGT/test_cubes_zyx.h5'
preview_kwargs    = dict(export_class=[1,2,3,7,8], max_z_pred=3)
initial_prev_h   = 0.3                  # hours: time after which first preview is made
prev_save_h      = 3.0
data_class = 'BatchCreatorImage' # <String>: Name of Data Class in TrainData or <tuple>: (path_to_file, class_name)
background_processes = 8
data_init_kwargs = {
    'd_path': '~/lustre/mkilling/BirdGT/',
    'l_path': '~/lustre/mkilling/BirdGT/',
    'd_files':
        [('j0126_old/v2_old_%i-raw-zyx.h5' % ii, 'raw') for ii in range(6)] + \
        [('j0126_new/v2_new_%i-raw-zyx.h5' % ii, 'raw') for ii in range(22)] + \
        [('neg_ex/v2_neg_%i-raw-zyx.h5' % ii, 'raw') for ii in range(5)] + \
        [('myelin/v2_myelin_%i-raw-zyx.h5' % ii, 'raw') for ii in range(19)] + \
        [('objects/v2_center_cube-raw-zyx.h5', 'raw'), ],
    'l_files':
        [('j0126_old/v2_old_%i-combo-sparse-zyx.h5' % ii, 'combo') for ii in range(6)] + \
        [('j0126_new/v2_new_%i-combo-sparse-zyx.h5' % ii, 'combo') for ii in range(22)] + \
        [('neg_ex/v2_neg_%i-combo-sparse.h5' % ii, 'combo') for ii in range(5)] + \
        [('myelin/v2_myelin_%i-combo-propagate-sparse.h5' % ii, 'combo') for ii in range(19)] + \
        [('objects/v2_center_cube-combo-sparse-zyx.h5', 'combo'), ],
    'cube_prios':
        [3, ] * (6 - 1 + 22 - 2) + [1, ] * 5 + [0.3, ] * 19 + [10, ],
    'aniso_factor': 2.0,
    'valid_cubes': [0, 6, 7],
    'target_discrete_ix': [0, 1, 2],
    'h5stream': True
}
data_batch_args = {
    'grey_augment_channels': [0],
    'ret_ll_mask': False,
    'warp': 0.4,
    'warp_args': {
        'sample_aniso': True,
        'lock_z': False,
        'no_x_flip': False,
        'warp_amount': 0.4,
        'perspective': True
    },
    'ignore_thresh': False
}
n_steps = 800000
max_runtime = 4 * 24 * 3600 # in seconds
history_freq = 150
monitor_batch_size = 10
optimiser = 'Adam'
optimiser_params = {
    'lr': 0.7e-5,
    'mom': 0.95,
    'wd': 0.5e-3,
    'beta2': 0.995
}
batch_size = 1

def create_model():
    from elektronn2 import neuromancer
    import theano.tensor as T
    import numpy as np

    in_sh = (None,1,572-32*9,572-32*9)
    img = neuromancer.Input(in_sh, 'b,f,x,y', name='raw')

    out0  = neuromancer.Conv(img,  64,  (3,3), (1,1))
    out1  = neuromancer.Conv(out0, 64,  (3,3), (1,1))
    out2  = neuromancer.Pool(out1, (2,2))

    out3  = neuromancer.Conv(out2, 128,  (3,3), (1,1))
    out4  = neuromancer.Conv(out3, 128,  (3,3), (1,1))
    out5  = neuromancer.Pool(out4, (2,2))

    out6  = neuromancer.Conv(out5, 256,  (3,3), (1,1))
    out7  = neuromancer.Conv(out6, 256,  (3,3), (1,1))
    out8  = neuromancer.Pool(out7, (2,2))

    out9  = neuromancer.Conv(out8, 512,  (3,3), (1,1))
    out10 = neuromancer.Conv(out9, 512,  (3,3), (1,1))
    out11 = neuromancer.Pool(out10, (2,2))

    out12 = neuromancer.Conv(out11, 1024,  (3,3), (1,1))
    out13 = neuromancer.Conv(out12, 1024,  (3,3), (1,1))
    out14 = neuromancer.Pool(out13, (2,2))

    ####

    up0 = neuromancer.UpConvMerge(out10, out14, 1024)
    up1 = neuromancer.Conv(up0, 512,  (3,3), (1,1))
    up2 = neuromancer.Conv(up1, 512,  (3,3), (1,1))

    up3 = neuromancer.UpConvMerge(out7, up2, 512)
    up4 = neuromancer.Conv(up3, 256,  (3,3), (1,1))
    up5 = neuromancer.Conv(up4, 256,  (3,3), (1,1))

    up6 = neuromancer.UpConvMerge(out4, up5, 256)
    up7 = neuromancer.Conv(up6, 128,  (3,3), (1,1))
    up8 = neuromancer.Conv(up7, 128,  (3,3), (1,1))

    up9 = neuromancer.UpConvMerge(out1, up8, 128)
    up10 = neuromancer.Conv(up9, 64,  (3,3), (1,1))
    top_feat = neuromancer.Conv(up10, 64,  (3,3), (1,1))


    # Target outputs
    barr_out = neuromancer.Conv(top_feat,  3, (1,1), (1,1), activation_func='lin', name='barr')
    obj_out  = neuromancer.Conv(top_feat,  4, (1,1), (1,1), activation_func='lin', name='obj')
    my_out   = neuromancer.Conv(top_feat,  3, (1,1), (1,1), activation_func='lin', name='my')
    barr_out = neuromancer.Softmax(barr_out)
    obj_out  = neuromancer.Softmax(obj_out)
    my_out   = neuromancer.Softmax(my_out)

    target   = neuromancer.Input_like(top_feat, dtype='int16', override_f=3, name='target')
    barr, obj, my = neuromancer.split(target, 'f', n_out=3, name=['barr_t', 'obj_t', 'my_t'])

    # Target loss
    barr_loss_pix = neuromancer.MultinoulliNLL(barr_out, barr, target_is_sparse=True,name='nll_barr')
    obj_loss_pix  = neuromancer.MultinoulliNLL(obj_out, obj, target_is_sparse=True, name='nll_obj')
    my_loss_pix   = neuromancer.MultinoulliNLL(my_out, my, target_is_sparse=True, name='nll_my')
    pred          = neuromancer.Concat([barr_out, obj_out, my_out], axis='f')
    pred.feature_names = ['barrier_bg', 'barr_mem', 'barr_ecs', 'obj_bg',
                          'obj_mito', 'obj_ves', 'obj_syn', 'my_bg', 'my_out', 'my_in']

    # Objective
    weights = np.array([2.154, 0.42, 0.42])
    weights *= len(weights) / weights.sum()
    loss = neuromancer.AggregateLoss([barr_loss_pix,
                                      obj_loss_pix,
                                      my_loss_pix],
                                      mixing_weights=weights)
    # Monitoring  / Debug outputs
    nll_barr   = neuromancer.ApplyFunc(barr_loss_pix, T.mean, name='mnll_barr')
    nll_obj    = neuromancer.ApplyFunc(obj_loss_pix, T.mean, name='mnll_obj')
    nll_my   = neuromancer.ApplyFunc(my_loss_pix, T.mean, name='mnll_my')
    errors = neuromancer.Errors(barr_out, barr, target_is_sparse=True)

    model = neuromancer.model_manager.getmodel()
    model.designate_nodes(input_node=img, target_node=target, loss_node=loss,
                                  prediction_node=pred,
                                  prediction_ext=[loss, errors, pred],
                                  debug_outputs =[nll_barr, errors, nll_obj, nll_my])

    return model


if __name__ == "__main__":
    print('Testing and visualising model...\n(If you want to train with this '
          'config file instead, run '
          '"$ elektronn2-train {}".)\n'.format(__file__))
    import traceback

    model = create_model()

    try:
        model.test_run_prediction()
    except Exception as e:
        traceback.print_exc()
        print('Test run failed.\nIn case your GPU ran out of memory, the '
              'principal setup might still be working')

    try:
        from elektronn2.utils.d3viz import visualise_model
        import getpass

        user_name = getpass.getuser()
        filename_noext = __file__.split('.')[-2]
        vispath = '/tmp/{}_{}_model-graph'.format(user_name, filename_noext)
        visualise_model(model, vispath)
        print('Visualisation files are saved at {}'.format(
            vispath + '.{png,html}'))
        # import webbrowser
        # webbrowser.open(vispath + '.png')
        # webbrowser.open(vispath + '.html')
    except Exception as e:
        traceback.print_exc()
        print('Could not visualise model graph.\n'
              'Are pydotplus and graphviz properly installed?')
