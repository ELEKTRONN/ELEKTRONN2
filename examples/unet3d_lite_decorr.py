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
    'grey_augment_channels': [0], 'warp': 0.0,
    'warp_args': {'sample_aniso': True, 'perspective': True
    }
}
n_steps = 1e6
max_runtime = 2 * 24 * 3600  # in seconds
history_freq = 1200
monitor_batch_size = 5
optimiser = 'SGD'
optimiser_params = {'lr': 10e-3, 'mom': 0.9,# 'beta2': 0.99,
                    'wd': 0.5e-3}
schedules = {'lr': {'dec': 0.99}, }
batch_size = 1
dr = 0.01


def create_model():
    from elektronn2 import neuromancer
    import numpy as np

    in_sh = (None,1,22,140,140)
    # For quickly trying out input shapes via CLI args, uncomment:
    #import sys; a = int(sys.argv[1]); b = int(sys.argv[2]); in_sh = (None,1,a,b,b)
    inp = neuromancer.Input(in_sh, 'b,f,z,x,y', name='raw')
    inputs_split = neuromancer.multi_dim_split(inp, axes=["z", "y", "x"], n_outs=[2, 2, 2])

    # x1
    out0  = neuromancer.Conv(inputs_split[0],  20,  (1,3,3), (1,1,1))
    out1  = neuromancer.Conv(out0, 20,  (1,3,3), (1,2,2))
    out2 = neuromancer.Pool(out1, (1, 2, 2))

    out3  = neuromancer.Conv(out2, 30,  (1,3,3), (1,1,1))
    out4  = neuromancer.Conv(out3, 30,  (1,3,3), (1,1,1))
    out5 = neuromancer.Pool(out4, (1, 2, 2))

    out6  = neuromancer.Conv(out5, 40,  (1,3,3), (1,1,1))
    out7  = neuromancer.Conv(out6, 40,  (1,3,3), (1,1,1))

    up3 = neuromancer.UpConvMerge(out4, out7, 60)
    up4 = neuromancer.Conv(up3, 50,  (1,3,3), (1,1,1))
    up5 = neuromancer.Conv(up4, 50,  (1,3,3), (1,1,1))

    up6 = neuromancer.UpConvMerge(out1, up5, 40)
    up7 = neuromancer.Conv(up6, 40,  (2,3,3), (1,1,1))
    up8 = neuromancer.Conv(up7, 40,  (2,3,3), (1,1,1))
    barr_x1 = neuromancer.Conv(up8,  2, (1,1,1), (1,1,1), activation_func='lin', name='barr')

    # x2-x8
    def partial_unet(inp_sp):
        out0_b  = neuromancer.Conv(inp_sp,  20,  (1,3,3), (1,1,1), w=out0.w, b=out0.b)
        out1_b  = neuromancer.Conv(out0_b, 20,  (1,3,3), (1,1,1), w=out1.w, b=out1.b)
        out2_b = neuromancer.Pool(out1_b, (1, 2, 2))
        out3_b  = neuromancer.Conv(out2_b, 30,  (1,3,3), (1,1,1), w=out3.w, b=out3.b)
        out4_b  = neuromancer.Conv(out3_b, 30,  (1,3,3), (1,1,1), w=out4.w, b=out4.b)
        out5_b = neuromancer.Pool(out4_b, (1, 2, 2))
        out6_b  = neuromancer.Conv(out5_b, 40,  (1,3,3), (1,1,1), w=out6.w, b=out6.b)
        out7_b  = neuromancer.Conv(out6_b, 40,  (1,3,3), (1,1,1), w=out7.w, b=out7.b)
        up3_b = neuromancer.UpConvMerge(out4_b, out7_b, 60)
        up4_b = neuromancer.Conv(up3_b, 50,  (1,3,3), (1,1,1), w=up4.w, b=up4.b)
        up5_b = neuromancer.Conv(up4_b, 50,  (1,3,3), (1,1,1), w=up5.w, b=up5.b)
        up6_b = neuromancer.UpConvMerge(out1_b, up5_b, 40)
        up7_b = neuromancer.Conv(up6_b, 40,  (2,3,3), (1,1,1), w=up7.w, b=up7.b)
        up8_b = neuromancer.Conv(up7_b, 40,  (2,3,3), (1,1,1), w=up8.w, b=up8.b)
        barr_sp = neuromancer.Conv(up8_b,  2, (1,1,1), (1,1,1), activation_func='lin', name='barr', w=barr_x1.w, b=barr_x1.b)
        return barr_sp

    barr_x2 = partial_unet(inputs_split[1])
    barr_x3 = partial_unet(inputs_split[2])
    barr_x4 = partial_unet(inputs_split[3])
    barr_x5 = partial_unet(inputs_split[4])
    barr_x6 = partial_unet(inputs_split[5])
    barr_x7 = partial_unet(inputs_split[6])
    barr_x8 = partial_unet(inputs_split[7])

    # merge again
    barr_x_conc1 = neuromancer.Concat([barr_x1, barr_x2], axis='x')
    barr_x_conc2 = neuromancer.Concat([barr_x3, barr_x4], axis='x')
    barr_x_conc3 = neuromancer.Concat([barr_x5, barr_x6], axis='x')
    barr_x_conc4 = neuromancer.Concat([barr_x7, barr_x8], axis='x')
    barr_y_conc1 = neuromancer.Concat([barr_x_conc1, barr_x_conc2], axis='y')
    barr_y_conc2 = neuromancer.Concat([barr_x_conc3, barr_x_conc4], axis='y')
    barr = neuromancer.Concat([barr_y_conc1, barr_y_conc2], axis='z')
    target = neuromancer.Input_like(barr, override_f=1, name='target')

    probs = neuromancer.Softmax(barr)
    loss_pix = neuromancer.MultinoulliNLL(probs, target, target_is_sparse=True,
                                          name='nll_barr')

    loss = neuromancer.AggregateLoss(loss_pix, name='loss')
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

