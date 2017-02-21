# -*- coding: utf-8 -*-

save_path = '~/elektronn2_examples/'
save_name = '3d_cnn'
preview_data_path = '~/neuro_data/preview_cubes.h5'
preview_kwargs    = {
    'export_class': 'all',
    'max_z_pred': 5
}
initial_prev_h = 0.5  # hours: time after which the first preview is made
prev_save_h = 1.0  # hours: time interval between planned previews.
data_class = 'BatchCreatorImage' # <String>: Name of the data class in
                                 # ``elektronn2.data.traindata`` (as used here) or
                                 # <tuple>: (path_to_file, class_name)
background_processes = 2
data_init_kwargs = {
    'd_path' : '~/neuro_data/',
    'l_path': '~/neuro_data/',
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
max_runtime = 4 * 24 * 3600 # in seconds
history_freq = 600
monitor_batch_size = 30
optimiser = 'Adam'
optimiser_params = {
    'lr': 0.0005,
    'mom': 0.9,
    'beta2': 0.999,
    'wd': 0.5e-4
}
lr_schedule = None  # dict(updates=[])
mom_schedule = None  # dict(updates=[])
batch_size = 1


def create_model():
    from elektronn2 import neuromancer
    in_sh = (None,1,23,185,185)
    inp = neuromancer.Input(in_sh, 'b,f,z,x,y', name='raw')

    out   = neuromancer.Conv(inp, 20,  (1,6,6), (1,2,2))
    out   = neuromancer.Conv(out, 30,  (1,5,5), (1,2,2))
    out   = neuromancer.Conv(out, 40,  (1,5,5), (1,1,1))
    out   = neuromancer.Conv(out, 80,  (4,4,4), (2,1,1))

    out   = neuromancer.Conv(out, 100, (3,4,4), (1,1,1))
    out   = neuromancer.Conv(out, 100, (3,4,4), (1,1,1))
    out   = neuromancer.Conv(out, 150, (2,4,4), (1,1,1))
    out   = neuromancer.Conv(out, 200, (1,4,4), (1,1,1))
    out   = neuromancer.Conv(out, 200, (1,4,4), (1,1,1))

    out   = neuromancer.Conv(out, 200, (1,1,1), (1,1,1))
    out   = neuromancer.Conv(out,   2, (1,1,1), (1,1,1), activation_func='lin')
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


if __name__ == "__main__":
    print('Testing and visualising model...\n(If you want to train with this '
          'config file instead, run '
          '"$ elektronn2-train {}".)\n\n'.format(__file__))
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
        visualise_model(model, 'model-graph')
        import webbrowser
        webbrowser.open('model-graph.png')
        webbrowser.open('model-graph.html')
    except Exception as e:
        traceback.print_exc()
        print('Could not print model model graph.\n'
              'Are pydot and graphviz properly installed?')
