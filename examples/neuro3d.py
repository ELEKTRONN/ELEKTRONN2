# -*- coding: utf-8 -*-

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
    in_sh = (None,1,23,185,185)
    inp = neuromancer.Input(in_sh, 'b,f,z,x,y', name='raw')

    out   = neuromancer.Conv(inp, 20,  (1,6,6), (1,2,2))
    out   = neuromancer.Conv(out, 30,  (1,5,5), (1,2,2))
    out   = neuromancer.Conv(out, 40,  (1,5,5))
    out   = neuromancer.Conv(out, 80,  (4,4,4), (2,1,1))

    out   = neuromancer.Conv(out, 100, (3,4,4))
    out   = neuromancer.Conv(out, 100, (3,4,4))
    out   = neuromancer.Conv(out, 150, (2,4,4))
    out   = neuromancer.Conv(out, 200, (1,4,4))
    out   = neuromancer.Conv(out, 200, (1,4,4))

    out   = neuromancer.Conv(out, 200, (1,1,1))
    out   = neuromancer.Conv(out,   2, (1,1,1), activation_func='lin')
    probs = neuromancer.Softmax(out)

    target = neuromancer.Input_like(probs, override_f=1, name='target')
    loss_pix  = neuromancer.MultinoulliNLL(probs, target, target_is_sparse=True)

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
