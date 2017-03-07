# -*- coding: utf-8 -*-
# ELEKTRONN2 Toolkit
# Copyright (c) 2016 Philipp J. Schubert
# All rights reserved

save_path = '~/elektronn2_examples/'
data_class = 'PianoData_perc'
background_processes = 2
n_steps=10000
max_runtime = 4 * 24 * 3600 # in seconds
history_freq = 200
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
    in_sh = (20, 20 * 58)
    inp = neuromancer.Input(in_sh, 'b,f', name='raw')
    out = neuromancer.Perceptron(inp, 700, 'lin')
    out = neuromancer.Perceptron(inp, 500, 'lin')
    out = neuromancer.Perceptron(inp, 300, 'lin')
    out = neuromancer.Perceptron(out, 2*58, 'lin')
    out = neuromancer.Softmax(out, n_indep=58)
    target = neuromancer.Input_like(out, override_f=1, name='target')
    weights = neuromancer.ValueNode((116, ), 'f', value=[0.2, 1.8]*58)
    loss = neuromancer.MultinoulliNLL(
        out, target, target_is_sparse=True, class_weights=weights, name='nll'
    )
    # Objective
    loss = neuromancer.AggregateLoss(loss)
    # Monitoring  / Debug outputs
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
