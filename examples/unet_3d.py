# -*- coding: utf-8 -*-

save_path = '~/elektronn2_examples/'
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
    import theano.tensor as T
    import numpy as np

    in_sh = (None,1,24,188,188)
    # For quickly trying out input shapes via CLI args, uncomment:
    #import sys
    #a = int(sys.argv[1])
    #b = int(sys.argv[2])
    #in_sh = (None,1,a,b,b)
    inp = neuromancer.Input(in_sh, 'b,f,z,x,y', name='raw')

    out0  = neuromancer.Conv(inp,  64,  (1,3,3), (1,1,1))
    out1  = neuromancer.Conv(out0, 64,  (1,3,3), (1,1,1))
    out2  = neuromancer.Pool(out1, (1,2,2))

    out3  = neuromancer.Conv(out2, 128,  (1,3,3), (1,1,1))
    out4  = neuromancer.Conv(out3, 128,  (1,3,3), (1,1,1))
    out5  = neuromancer.Pool(out4, (1,2,2))

    out6  = neuromancer.Conv(out5, 256,  (1,3,3), (1,1,1))
    out7  = neuromancer.Conv(out6, 256,  (1,3,3), (1,1,1))
    out8  = neuromancer.Pool(out7, (1,2,2))

    out9  = neuromancer.Conv(out8, 512,  (3,3,3), (1,1,1))
    out10 = neuromancer.Conv(out9, 512,  (3,3,3), (1,1,1))
    out11 = neuromancer.Pool(out10, (2,2,2))

    out12 = neuromancer.Conv(out11, 1024,  (1,3,3), (1,1,1))
    out13 = neuromancer.Conv(out12, 1024,  (1,3,3), (1,1,1))
    out14 = neuromancer.Pool(out13, (1,2,2))

    up0 = neuromancer.UpConvMerge(out10, out14, 1024)
    up1 = neuromancer.Conv(up0, 512,  (1,3,3), (1,1,1))
    up2 = neuromancer.Conv(up1, 512,  (1,3,3), (1,1,1))

    up3 = neuromancer.UpConvMerge(out7, up2, 512)
    up4 = neuromancer.Conv(up3, 256,  (1,3,3), (1,1,1))
    up5 = neuromancer.Conv(up4, 256,  (1,3,3), (1,1,1))

    up6 = neuromancer.UpConvMerge(out4, up5, 256)
    up7 = neuromancer.Conv(up6, 128,  (3,3,3), (1,1,1))
    up8 = neuromancer.Conv(up7, 128,  (3,3,3), (1,1,1))

    up9 = neuromancer.UpConvMerge(out1, up8, 128)
    up10 = neuromancer.Conv(up9, 64,  (3,3,3), (1,1,1))
    up11 = neuromancer.Conv(up10, 64,  (3,3,3), (1,1,1))

    barr = neuromancer.Conv(up11,  3, (1,1,1), (1,1,1), activation_func='lin', name='barr')
    probs = neuromancer.Softmax(barr)

    target = neuromancer.Input_like(up11, override_f=1, name='target')

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
