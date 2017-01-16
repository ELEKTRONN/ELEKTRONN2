# -*- coding: utf-8 -*-
# ELEKTRONN2 Toolkit
# Copyright (c) 2016 Philipp J. Schubert
# All rights reserved
save_path = '~/numa_examples/'
save_name = "mnist_test"

preview_data_path = None
preview_kwargs    = dict(export_class='all', max_z_pred=5)
initial_prev_h   = 0.5                  # hours: time after which first preview is made
prev_save_h      = 1.0
data_class = 'MNISTData' # <String>: Name of Data Class in TrainData or <tuple>: (path_to_file, class_name)
background_processes = 2

n_steps = 300000
max_runtime = 4 * 24 * 3600 # in seconds
history_freq = 2000
monitor_batch_size = 20
optimiser = 'Adam'
data_batch_args = {}
data_init_kwargs = {}#'shift_augment': False}
optimiser_params = dict(lr=2e-4, mom=0.9, beta2=0.99, wd=0.5e-3)
batch_size = 20

def create_model():
    from elektronn2 import neuromancer

    act = 'relu'
    in_sh = (20,1,26,26)
    inp = neuromancer.Input(in_sh, 'b,f,y,x', name='raw')

    out = neuromancer.Conv(inp, 12,  (3,3), (2,2), activation_func=act, batch_normalisation = 'train')
    out = neuromancer.Conv(out, 36,  (3,3), (2,2), activation_func=act, batch_normalisation = 'train')
    out = neuromancer.Conv(out, 64,  (3,3), (1,1), activation_func=act, batch_normalisation = 'train')
    out = neuromancer.Perceptron(out, 200, flatten=True)
    out = neuromancer.Perceptron(out, 10, activation_func='lin')
    out = neuromancer.Softmax(out)
    target   = neuromancer.Input_like(out, override_f=1, name='target')
    loss  = neuromancer.MultinoulliNLL(out, target, name='nll_', target_is_sparse=True)
    # Objective
    loss = neuromancer.AggregateLoss(loss)
    # Monitoring  / Debug outputs
    errors = neuromancer.Errors(out, target, target_is_sparse=True)

    model = neuromancer.model_manager.getmodel()
    model.designate_nodes(input_node=inp, target_node=target, loss_node=loss,
                                  prediction_node=out,
                                  prediction_ext=[loss, errors, out])
    return model

if __name__ == "__main__":
    import traceback
    model = create_model()


    try:
        model.test_run_prediction()
    except Exception as e:
        traceback.print_exc()
        print("Test run failed. In case your GPU ran out of memory the \
               principal setup might still be working")

    try:
        from elektronn2.utils.d3viz import visualise_model
        visualise_model(model, 'model-graph')
        import webbrowser
        webbrowser.open('model-graph.png')
        webbrowser.open('model-graph.html')
    except Exception as e:
        traceback.print_exc()
        print("Could not print model model graph. Is pydot/graphviz properly installed?")