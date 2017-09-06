# ELEKTRONN2
# Copyright (c) 2016 Philipp J. Schubert
# All rights reserved
import os
import numpy as np
from syconnfs.handler.basics import get_filepaths_from_dir
save_path = '~/elektronn2_training/'
preview_data_path = '~/neuro_data_zxy/preview_cubes.h5'
preview_kwargs    = {
    'export_class': [1],
    'max_z_pred': 3
}
trainee_path = os.path.expanduser("~/devel/ELEKTRONN2/examples/unet3d_litelite.py")
trainee_dict = dict()
exec (compile(open(trainee_path).read(), trainee_path, 'exec'), {},
      trainee_dict)
# training is alternated between training trainee and adversarial network
# TODO: define #steps for each network individually
p = [.5, .5]  # probability to present ground truth to adversarial network; first entry while training trainee; second entry while training adversarial
adv_class_w = [[0, 1], [1, 1]]  # class weights for adversarial loss term, 0: trainee output 1: ground truth; first entry while training trainee network, i.e. here onyl the class 1 (ground truth)
mixing_w = [[0.5, 1], [1, 0]]  # mixing weights for adversarial and trainee loss; instead of training -lambda l_bce(a(x, s(x)), 0), we use +lambda l_bce(a(x, s(x)), 1) -> stronger gradient (see Goodfellow et al. 2014)
# permut_steps: number of steps after which training is switched
# (0-9: segmentor, 10-19: adversarial, 20:29: segmentor, ...)
# mixing weights
network_arch = {"adversarial": {"permut_steps": 10, 'mixing_weights': mixing_w,
    'class_weights': adv_class_w, "target_p": p, "lr_fact": 0.1}}
initial_prev_h = 1.0  # hours: time after which the first preview is made
prev_save_h = 1.0  # hours: time interval between planned previews.
data_class = trainee_dict["data_class"]
background_processes = 2
h5_fnames = get_filepaths_from_dir('/wholebrain/scratch/j0126/barrier_gt_phil/', ending="rawbarr-zyx.h5")
data_init_kwargs = {
    'zxy': True,
    'make_l_unique' : False,
    'd_path' : '/wholebrain/scratch/j0126/barrier_gt_phil/',
    'l_path': '/wholebrain/scratch/j0126/barrier_gt_phil/',
    'd_files': [(os.path.split(fname)[1], 'raW') for fname in h5_fnames],
    'l_files': [(os.path.split(fname)[1], 'labels') for fname in h5_fnames],
    'aniso_factor': 2,
    'valid_cubes': [2],
}
data_batch_args = {
    'grey_augment_channels': [0],
    'warp': 0.1,
    'warp_args': {
        'sample_aniso': True,
        'perspective': True
    }
}
n_steps = 1500000
max_runtime = 4 * 24 * 3600  # in seconds
history_freq = 400
monitor_batch_size = 5
optimiser = 'Adam'
dr = 0.01  # dropout
act = 'relu'
optimiser_params = {
    'lr': 0.0001,
    'mom': 0.9,
    'wd': 0.5e-4,
    'beta2': 0.999
}
schedules = {
    'lr': {'dec': 0.995},  # decay (multiply) lr by this factor every 1000 steps
}
batch_size = trainee_dict["batch_size"]

def create_model():
    from elektronn2 import neuromancer as nm
    trainee = trainee_dict["create_model"]()
    inp = trainee.input_node
    trainee_gt = trainee.target_node
    trainee_out = trainee.prediction_node
    trainee_loss = trainee.loss_node
    adv_target = nm.AdvTarget([trainee_out, trainee_gt])
    adv_input = nm.AdvMerge([trainee_out, trainee_gt, adv_target])

    # raw data
    diff = np.array(inp.shape.spatial_shape, dtype=np.int32)-np.array(trainee_out.shape.spatial_shape, dtype=np.int32)
    assert not np.any(diff % 2)
    raw_inp = nm.Crop(inp, list((diff // 2)))
    conv0  = nm.Conv(raw_inp,  15,  (2,3,3), activation_func=act, dropout_rate=dr, name="conv_adv")
    conv1  = nm.Conv(conv0, 20,  (2,3,3), activation_func=act, dropout_rate=dr, name="conv_adv")
    down0  = nm.Pool(conv1, (1,2,2), mode='max')  # mid res
    conv2  = nm.Conv(down0, 20,  (2,3,3), activation_func=act, dropout_rate=dr, name="conv_adv")
    conv3  = nm.Conv(conv2, 25,  (2,3,3), activation_func=act, dropout_rate=dr, name="conv_adv")
    # Merging low-res with mid-res features
    mrg1   = nm.UpConvMerge(conv1, conv3, 30, name="upconv_adv")
    mconv2 = nm.Conv(mrg1, 25,  (2,3,3), activation_func=act, dropout_rate=dr, name="conv_adv")
    raw_out = nm.Conv(mconv2, 25,  (2,3,3), activation_func=act, dropout_rate=dr, name="conv_adv")

    # segmentation
    conv0  = nm.Conv(adv_input, 15, (2,3,3), activation_func=act, dropout_rate=dr, name="conv_adv")
    conv1  = nm.Conv(conv0, 20, (2,3,3), activation_func=act, dropout_rate=dr, name="conv_adv")
    down0  = nm.Pool(conv1, (1,2,2), mode='max')  # mid res
    conv2  = nm.Conv(down0, 20, (2,3,3), activation_func=act, dropout_rate=dr, name="conv_adv")
    conv3  = nm.Conv(conv2, 25, (2,3,3), activation_func=act, dropout_rate=dr, name="conv_adv")
    # Merging low-res with mid-res features
    mrg1   = nm.UpConvMerge(conv1, conv3, 30, name="upconv_adv")
    mconv2 = nm.Conv(mrg1, 25,  (2,3,3), activation_func=act, dropout_rate=dr, name="conv_adv")
    seg_out = nm.Conv(mconv2, 25,  (2,3,3), activation_func=act, dropout_rate=dr, name="conv_adv")

    out = nm.Concat([raw_out, seg_out], axis="f")
    out = nm.Conv(out, 30, (2, 3, 3), pool_shape=(1, 1, 1), activation_func=act, dropout_rate=dr, name="conv_adv")
    out = nm.Conv(out, 40, (2, 3, 3), pool_shape=(1, 2, 2), activation_func=act, dropout_rate=dr, name="conv_adv")
    out = nm.Conv(out, 60, (2, 3, 3), pool_shape=(1, 2, 2), activation_func=act, dropout_rate=dr, name="conv_adv")
    out = nm.Conv(out, 2, (1, 3, 3), pool_shape=(1, 1, 1), activation_func=act, dropout_rate=dr, name="conv_adv")
    dec  = nm.Perceptron(out, 2, flatten=True, activation_func='lin', name="perc_adv")
    adv_out = nm.Softmax(dec)

    # as in orig. GAN paper of Godfellow et al. only the positive label is taken
    # into account (i.e. adversarial network predicts input is ground truth)
    # for the adv. prediction loss when training the trainee
    # if training the adv. network only the binary cross-entropy of the adv.
    # prediction is backpropagated.
    # flip node: flips target if training the trainee network, in order to have stronger gradients (see above, or 'Generative adversarial nets' from Goodfellow et al. 2014)
    # goal: maximize probability of adv. net that it predicts an output of the trainee network as ground truth map
    flipped_adv_target = nm.FlipNode(adv_target)
    adv_loss = nm.MultinoulliNLL(adv_out, flipped_adv_target, target_is_sparse=True, name='nll_adversarial', class_weights=None)
    adv_loss = nm.AggregateLoss([adv_loss], name='loss_adversarial')
    loss = nm.AggregateLoss([adv_loss, trainee_loss] , mixing_weights=None, name='loss_total')

    model = nm.model_manager.getmodel()
    model.designate_nodes(
        input_node=inp,
        target_node=trainee.target_node,
        loss_node=loss,
        prediction_node=trainee.prediction_node,
        prediction_ext=trainee.prediction_ext,
        debug_outputs=[adv_loss, trainee_loss, adv_target]
    )
    return model

if __name__ == "__main__":
    model = create_model()
    # "Test" if model is saveable
    # model.save("/tmp/"+save_name)
