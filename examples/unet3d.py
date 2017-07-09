# -*- coding: utf-8 -*-

# Implements the "3D U-Net" by Özgün Çiçek et al.
# (https://arxiv.org/abs/1606.06650). Notable differences:
# - Optimiser changes (Adam instead of SGD, changed lr etc.)
# - Enhanced image augmentation pipeline
# - No batch normalization
# - 2 output channels instead of 3 (because of neuro_data_zxy)
#
# Note that this network has high memory requirements and is
# very slow to train. A similar, but more light-weight model
# can be found in examples/unet_3d_lite.py.

save_path = '~/elektronn2_training/'
preview_data_path = '~/neuro_data_zxy/preview_cubes.h5'
preview_kwargs    = {
    'export_class': [1],
    'max_z_pred': 3
}
initial_prev_h = 1.0  # hours: time after which the first preview is made
prev_save_h = 1.0  # hours: time interval between planned previews.
data_class = 'BatchCreatorImage'
background_processes = 2
data_init_kwargs = {
    'd_path' : '~/neuro_data_zxy/',
    'l_path': '~/neuro_data_zxy/',
    'd_files': [('raw_%i.h5' %i, 'raw') for i in range(3)],
    'l_files': [('barrier_int16_%i.h5' %i, 'lab') for i in range(3)],
    'aniso_factor': 2,
    'valid_cubes': [2],
}
data_batch_args = {
    'grey_augment_channels': [0],
    'warp': 0.5,
    'warp_args': {
        'sample_aniso': True,
        'perspective': True
    }
}
n_steps = 150000
max_runtime = 24 * 3600 # in seconds
history_freq = 200
monitor_batch_size = 30
optimiser = 'Adam'
optimiser_params = {
    'lr': 0.0005,
    'mom': 0.9,
    'beta2': 0.999,
    'wd': 0.5e-4
}
schedules = {
    'lr': {'dec': 0.995}, # decay (multiply) lr by this factor every 1000 steps
}
batch_size = 1


def create_model():
    from elektronn2 import neuromancer

    in_sh = (None,1,116,132,132)
    inp = neuromancer.Input(in_sh, 'b,f,z,x,y', name='raw')

    # Convolution, downsampling of intermediate features
    conv0  = neuromancer.Conv(inp,  32,  (3,3,3), (1,1,1))
    conv1  = neuromancer.Conv(conv0, 64,  (3,3,3), (1,1,1))
    down0  = neuromancer.Pool(conv1, (2,2,2), mode='max')  # mid res
    conv2  = neuromancer.Conv(down0, 64,  (3,3,3), (1,1,1))
    conv3  = neuromancer.Conv(conv2, 128,  (3,3,3), (1,1,1))
    down1  = neuromancer.Pool(conv3, (2,2,2), mode='max')  # low res
    conv4  = neuromancer.Conv(down1, 128,  (3,3,3), (1,1,1))
    conv5  = neuromancer.Conv(conv4, 256,  (3,3,3), (1,1,1))
    down2  = neuromancer.Pool(conv5, (2,2,2), mode='max')  # very low res
    conv6  = neuromancer.Conv(down2, 256,  (3,3,3), (1,1,1))
    conv7  = neuromancer.Conv(conv6, 512,  (3,3,3), (1,1,1))

    # Merging very low-res features with low-res features
    mrg0   = neuromancer.UpConvMerge(conv5, conv7, 512)
    mconv0 = neuromancer.Conv(mrg0, 256,  (3,3,3), (1,1,1))
    mconv1 = neuromancer.Conv(mconv0, 256,  (3,3,3), (1,1,1))

    # Merging low-res with mid-res features
    mrg1   = neuromancer.UpConvMerge(conv3, mconv1, 256)
    mconv2 = neuromancer.Conv(mrg1, 128,  (3,3,3), (1,1,1))
    mconv3 = neuromancer.Conv(mconv2, 128,  (3,3,3), (1,1,1))

    # Merging mid-res with high-res features
    mrg2   = neuromancer.UpConvMerge(conv1, mconv3, 128)
    mconv4 = neuromancer.Conv(mrg2, 64,  (3,3,3), (1,1,1))
    mconv5 = neuromancer.Conv(mconv4, 64,  (3,3,3), (1,1,1))

    barr = neuromancer.Conv(mconv5,  2, (1,1,1), (1,1,1), activation_func='lin', name='barr')
    probs = neuromancer.Softmax(barr)

    target = neuromancer.Input_like(mconv5, override_f=1, name='target')

    loss_pix = neuromancer.MultinoulliNLL(probs, target, target_is_sparse=True, name='nll_barr')

    loss = neuromancer.AggregateLoss(loss_pix , name='loss')
    errors = neuromancer.Errors(probs, target, target_is_sparse=True)

    model = neuromancer.model_manager.getmodel()
    model.designate_nodes(
        input_node=inp,
        target_node=target,
        loss_node=loss,
        prediction_node=probs,
        prediction_ext=[loss, errors, probs]
    )
    return model


if __name__ == '__main__':
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
        vispath = '/tmp/' + __file__.split('.')[-2] + '_model-graph'
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
