# -*- coding: utf-8 -*-
import numpy as np
import os
from syconnfs.handler.basics import get_filepaths_from_dir
save_path = '~/elektronn2_training/decorr/'
preview_data_path = '~/neuro_data_zxy/preview_cubes.h5'
preview_kwargs    = {
    'export_class': [1],
    'max_z_pred': 3
}
initial_prev_h = 1.0  # hours: time after which the first preview is made
prev_save_h = 1.0  # hours: time interval between planned previews.
data_class = 'BatchCreatorImage'
background_processes = 4
h5_fnames = get_filepaths_from_dir('/wholebrain/scratch/j0126/barrier_gt_phil/', ending="rawbarr-zyx.h5")
data_init_kwargs = {
    'zxy': True,
    'd_path' : '/wholebrain/scratch/j0126/barrier_gt_phil/',
    'l_path': '/wholebrain/scratch/j0126/barrier_gt_phil/',
    'd_files': [(os.path.split(fname)[1], 'raW') for fname in h5_fnames],
    'l_files': [(os.path.split(fname)[1], 'labels') for fname in h5_fnames],
    'aniso_factor': 2,
    'valid_cubes': [2],
}

data_batch_args = {
    'grey_augment_channels': [0], 'warp': 0.0,
    'warp_args': {'sample_aniso': True, 'perspective': True
    }
}
n_steps = 1e6
max_runtime = 2 * 24 * 3600  # in seconds
history_freq = 1200
monitor_batch_size = 5
optimiser = 'Adam'
optimiser_params = {'lr': 0.001, 'mom': 0.9, 'beta2': 0.999,
                    'wd': 0.5e-3}
schedules = {'lr': {'dec': 0.99}, }
batch_size = 1
dr = 0.01

def create_model():
    from elektronn2 import neuromancer
    # ------------------------------------------- dummy compilation
    #  to get FOV and with this the actual input shape
    # TODO: Find a way to compute FOV before compilation...
    part_inp_sh = (1, 1, 12, 52, 52)
    inp = neuromancer.ValueNode(part_inp_sh, 'b,f,z,x,y', name='raw',
                                dtype="float32")

    out0  = neuromancer.Conv(inp,  20,  (1,3,3), (1,1,1))
    out1  = neuromancer.Conv(out0, 20,  (1,3,3), (1,1,1))
    out2 = neuromancer.Pool(out1, (1, 2, 2))

    out3  = neuromancer.Conv(out2, 30,  (2,3,3), (1,1,1))
    out4  = neuromancer.Conv(out3, 30,  (2,3,3), (1,1,1))
    out5 = neuromancer.Pool(out4, (1, 2, 2))

    out6  = neuromancer.Conv(out5, 40,  (2,3,3), (1,1,1))
    out7  = neuromancer.Conv(out6, 40,  (2,3,3), (1,1,1))

    up3 = neuromancer.UpConvMerge(out4, out7, 60)
    up4 = neuromancer.Conv(up3, 50,  (2,3,3), (1,1,1))
    up5 = neuromancer.Conv(up4, 50,  (2,3,3), (1,1,1))

    up6 = neuromancer.UpConvMerge(out1, up5, 40)
    up7 = neuromancer.Conv(up6, 40,  (2,3,3), (1,1,1))
    up8 = neuromancer.Conv(up7, 40,  (2,3,3), (1,1,1))
    dummy_out = neuromancer.Conv(up8,  2, (1,1,1), (1,1,1), activation_func='lin',
                              name='barr_dummy')
    # ------------------------------------------- dummy end

    # get the actual input
    part_inp_sh_spatial = np.array(inp.shape.spatial_shape, dtype=np.int32)
    part_target_sh = np.array(dummy_out.shape.spatial_shape, dtype=np.int32)
    inp_sh_spatial = 2 * part_inp_sh_spatial - part_target_sh
    inp_sh = (None, 1, inp_sh_spatial[0], inp_sh_spatial[1], inp_sh_spatial[2])
    inp = neuromancer.Input(inp_sh, 'b,f,z,x,y', name='raw')
    target = neuromancer.Input_like(dummy_out, override_f=1, name='target')

    # split input into cubes with little overlap (overlap = target size / 2)
    inputs_split = neuromancer.decorr_split(inp, part_inp_sh_spatial)
    barrs = []
    part_losses = []

    # define network with shared weights
    def partial_unet(inp_sp):
        out0_b  = neuromancer.Conv(inp_sp,  20,  (1,3,3), (1,1,1), w=out0.w, b=out0.b)
        out1_b  = neuromancer.Conv(out0_b, 20,  (1,3,3), (1,1,1), w=out1.w, b=out1.b)
        out2_b = neuromancer.Pool(out1_b, (1, 2, 2))
        out3_b  = neuromancer.Conv(out2_b, 30,  (2,3,3), (1,1,1), w=out3.w, b=out3.b)
        out4_b  = neuromancer.Conv(out3_b, 30,  (2,3,3), (1,1,1), w=out4.w, b=out4.b)
        out5_b = neuromancer.Pool(out4_b, (1, 2, 2))
        out6_b  = neuromancer.Conv(out5_b, 40,  (2,3,3), (1,1,1), w=out6.w, b=out6.b)
        out7_b  = neuromancer.Conv(out6_b, 40,  (2,3,3), (1,1,1), w=out7.w, b=out7.b)
        up3_b = neuromancer.UpConvMerge(out4_b, out7_b, 60)
        up4_b = neuromancer.Conv(up3_b, 50,  (2,3,3), (1,1,1), w=up4.w, b=up4.b)
        up5_b = neuromancer.Conv(up4_b, 50,  (2,3,3), (1,1,1), w=up5.w, b=up5.b)
        up6_b = neuromancer.UpConvMerge(out1_b, up5_b, 40)
        up7_b = neuromancer.Conv(up6_b, 40,  (2,3,3), (1,1,1), w=up7.w, b=up7.b)
        up8_b = neuromancer.Conv(up7_b, 40,  (2,3,3), (1,1,1), w=up8.w, b=up8.b)
        barr_sp = neuromancer.Conv(up8_b,  2, (1,1,1), (1,1,1), activation_func='lin', name='barr', w=dummy_out.w, b=dummy_out.b)
        return barr_sp

    # cubes 1-8
    for i in range(0, 8):
        partial_res = partial_unet(inputs_split[i])
        part_probs = neuromancer.Softmax(partial_res, name="softmax_part")
        losses_pix = neuromancer.MultinoulliNLL(part_probs, target,
                                                target_is_sparse=True,
                                                name="nll_part")
        part_loss = neuromancer.AggregateLoss(losses_pix, name='loss_part')
        part_losses.append(part_loss)
        barrs.append(part_probs)
    barrs_conc = neuromancer.Concat(barrs, axis='f')
    ensemble_res = neuromancer.Conv(barrs_conc, 30, (1,1,1), (1,1,1))
    ensemble_res = neuromancer.Conv(ensemble_res, 20, (1,1,1), (1,1,1))
    ensemble_res = neuromancer.Conv(ensemble_res, 2, (1,1,1), (1,1,1),
                                    activation_func="lin")
    probs = neuromancer.Softmax(ensemble_res, name="softmax_ensemble")
    losses_pix = neuromancer.MultinoulliNLL(probs, target, target_is_sparse=True,
                                             name="nll_ensemble")
    loss = neuromancer.AggregateLoss(losses_pix, name='loss_ensemble')
    loss = neuromancer.AggregateLoss([loss,] + part_losses, name='loss_total', mixing_weights=[0.5, ] + [0.0625,]*8)
    # hack, error will only be evaluated on the first cube
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
        from elektronn2.utils.d3viz import visualise_model
        vispath = __file__.split('.')[-2] + '_model-graph'
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

    try:
        model.test_run_prediction()
    except Exception as e:
        traceback.print_exc()
        print('Test run failed.\nIn case your GPU ran out of memory, the '
              'principal setup might still be working')
