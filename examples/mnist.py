# -*- coding: utf-8 -*-
# ELEKTRONN2 Toolkit
# Copyright (c) 2016 Philipp J. Schubert
# All rights reserved
save_path = '~/elektronn2_examples/'
save_name = 'MNIST'
data_class = 'MNISTData'  # <String>: Name of the data class in
                          # ``elektronn2.data.traindata`` (as used here) or
                          # <tuple>: (path_to_file, class_name)
background_processes = 2
n_steps = 30000
max_runtime = 10 * 60 # in seconds
history_freq = 500
monitor_batch_size = 20
optimiser = 'Adam'
data_batch_args = {}
data_init_kwargs = {}
optimiser_params = {
    'lr': 2e-4,
    'mom': 0.9,
    'beta2': 0.99,
    'wd': 0.5e-3
}
batch_size = 20

def create_model():
    from elektronn2 import neuromancer

    act = 'relu'
    in_sh = (None,1,26,26)
    inp = neuromancer.Input(in_sh, 'b,f,y,x', name='raw')

    out = neuromancer.Conv(inp, 12, (3,3), (2,2), activation_func=act, batch_normalisation='train')
    out = neuromancer.Conv(out, 36, (3,3), (2,2), activation_func=act, batch_normalisation='train')
    out = neuromancer.Conv(out, 64, (3,3), (1,1), activation_func=act, batch_normalisation='train')
    out = neuromancer.Perceptron(out, 200, flatten=True)
    out = neuromancer.Perceptron(out, 10, activation_func='lin')
    out = neuromancer.Softmax(out)
    target = neuromancer.Input_like(out, override_f=1, name='target')
    loss  = neuromancer.MultinoulliNLL(out, target, name='nll_', target_is_sparse=True)
    # Objective
    loss = neuromancer.AggregateLoss(loss)
    # Monitoring / Debug outputs
    errors = neuromancer.Errors(out, target, target_is_sparse=True)

    model = neuromancer.model_manager.getmodel()
    model.designate_nodes(
        input_node=inp,
        target_node=target,
        loss_node=loss,
        prediction_node=out,
        prediction_ext=[loss, errors, out]
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
        vispath = '/tmp/' + save_name + '_model-graph'
        visualise_model(model, vispath)
        print('Visualisation files are saved at {}'.format(
            vispath + '.{png,html}'))
        import webbrowser
        webbrowser.open(vispath + '.png')
        webbrowser.open(vispath + '.html')
    except Exception as e:
        traceback.print_exc()
        print('Could not visualise model graph.\n'
              'Are pydotplus and graphviz properly installed?')
