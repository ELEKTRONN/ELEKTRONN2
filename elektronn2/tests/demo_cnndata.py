from __future__ import absolute_import, division, print_function
from builtins import filter, hex, input, int, map, next, oct, pow, range, \
    super, zip

import time
import numpy as np
from matplotlib import pyplot as plt

from ..utils.plotting import scroll_plot
from ..neuromancer import Input, Input_like, Conv
from ..training.parallelisation import BackgroundProc


def test_cnndata():
    print("Testing CNNData")

    x = Input((1, 1, 5, 40, 40), 'b,f,z,x,y')
    y = Conv(x, 2, (5, 31, 31), (1, 1, 1))
    l = Input_like(y, dtype='int16')

    # data_path        = os.path.expanduser('~/devel/data/BirdGT/') # (*) Path to data dir
    # label_path       = data_path
    # d_files          = [('raw_cube0-crop.h5', 'raw')]
    # l_files          = [('cube0_barrier-int16.h5', 'labels')]

    data_init_kwargs = dict(d_path='~/lustre/mkilling/BirdGT/j0126_new/',
                            l_path='~/lustre/mkilling/BirdGT/j0126_new/',
                            d_files=[('raw_%i.h5' % ii, 'raw') for ii in
                                     range(3)],
                            l_files=[('barrier_int16_%i.h5' % ii, 'lab') for ii
                                     in range(3)], n_target=2,
                            anisotropic_data=True, data_order='xyz',
                            valid_cubes=[0, 1], nhood_targets=True)

    D = CNNData(x, l, **data_init_kwargs)

    kwargs = dict(batch_size=1, flip=True, grey_augment_channels=[0],
                  ret_ll_mask=True, warp_on=False, ignore_thresh=0.5)

    batch = D.getbatch(**kwargs)

    # ------------------------------------------------------------------------------

    if False:  # BG
        bg = BackgroundProc(D.getbatch, n_proc=2, target_args=(),
                            target_kwargs=kwargs, profile=False)
        try:
            for i in range(1):
                data, label, seg, seg_gt, info1, info2 = bg.get()
                # data, label, info1, info2 = D.getbatch(**kwargs)
                label = label[0]
                seg = seg[0]
                seg_gt = seg_gt[0]
                nhood = np.eye(3, dtype=np.int32)

                aff_pred = label * 0.95 + 0.05 * np.random.uniform(
                    size=label.shape)
                pos_counts, neg_counts = malis.get_malis_weights(aff_pred,
                                                                 label, seg,
                                                                 nhood)

                seg_color = seg * 1.0 / seg.max()
                seg_color = plt.cm.nipy_spectral(seg_color)
                seg_color_gt = seg_gt * 1.0 / seg_gt.max()
                seg_color_gt = plt.cm.nipy_spectral(seg_color_gt)

                aff_pred = np.transpose(aff_pred, (1, 2, 3, 0))
                pos_counts = np.transpose(pos_counts, (1, 2, 3, 0))
                neg_counts = np.log(neg_counts + 1)
                neg_counts = np.transpose(neg_counts, (1, 2, 3, 0))

                plt.ioff()
                fig, scroller = scroll_plot.scroll_plot(
                    [aff_pred, seg_color, seg_color_gt, neg_counts],
                    ['PRED', 'SEG', 'SEG_GT', 'NEG'])
                fig.show()
                plt.show()

        except KeyboardInterrupt:
            pass
        finally:
            bg.shutdown()

    if True:
        off = l.shape.offsets[0]
        data = batch[0]
        label = batch[1]
        img = data[0, 0,]
        for z in range(img.shape[0]):
            plt.imsave('/tmp/%i-img.png' % (z), img[z,], cmap='gray')

        lab = label[0, 0,]
        for z in range(lab.shape[0]):
            plt.imsave('/tmp/%i-lab.png' % (z + off),
                       lab[z].astype(np.uint8) * 255)

    # TIMING
    if False:
        t0 = time.clock()
        tt0 = time.time()
        for i in range(100):
            data, label, ll_mask1, ll_mask2 = D.getbatch(**kwargs)
        t1 = time.clock()
        tt1 = time.time()
        print("clock {}".format(t1 - t0))
        print("time {}".format(tt1 - tt0))

        fig, scroller = scroll_plot.scroll_plot(
            [aff_pred, seg_color, pos_counts, neg_counts],
            ['PRED', 'LAB', 'POS', 'NEG'])
        fig.show()


if __name__=="__main__":
    from elektronn2.data.cnndata import BatchCreatorImage
    from elektronn2.data.image import center_cubes
    from elektronn2.utils.plotting import _scroll_plot2, scroll_plot


    x = Input((1, 1, 8, 71, 71), 'b,f,z,x,y')
    y = Conv(x, 2, (1, 5, 5), (1, 1, 1))
    l = Input_like(y, dtype='int16')

    data_init_kwargs = dict(d_path='~/lustre/mkilling/BirdGT/j0126_new/',
                            l_path='~/lustre/mkilling/BirdGT/j0126_new/',
                            d_files=[('raw_%i.h5' % ii, 'raw') for ii in
                                     range(3)],
                            l_files=[('barrier_int16_%i.h5' % ii, 'lab') for ii
                                     in range(3)], aniso_factor=2.0,
                            valid_cubes=[1, 2], target_discrete_ix=[])

    data = BatchCreatorImage(x, y, **data_init_kwargs)
    data.train_d[0] = np.transpose(data.train_d[0], (0, 3, 2, 1))
    data.train_l[0] = np.transpose(data.train_l[0], (0, 3, 2, 1))
    #    data.train_d[0][:,:,::20,:] *= 0.5
    #    data.train_d[0][:,:,:,::20] *= 0.5
    #    data.train_l[0][:,:,::20,:] *= 0.5
    #    data.train_l[0][:,:,:,::20] *= 0.5
    #    data.train_d[0][:,::10,:,:] *= 0.5
    #    data.train_l[0][:,::10,:,:] *= 0.5

    sample_warp_params = dict(sample_aniso=True, lock_z=True, no_x_flip=False,
                              warp_amount=1, perspective=True)

    data_batch_args = dict(grey_augment_channels=[0], ret_ll_mask=False,
                           warp=1, warp_args=sample_warp_params,
                           ignore_thresh=0.5, force_dense=True)

    batch = data.getbatch(**data_batch_args)
    img, target = batch[0][0, 0], batch[1][0, 0]
    img, target = center_cubes(img, target, crop=False)
    fig = _scroll_plot2([img, (0.4 * target + 0.6 * img)], ['raw', 'target'])

    plt.ioff()
    for i in range(100):
        batch = data.getbatch(**data_batch_args)
        img, target = batch[0][0, 0], batch[1][0, 0]
        img, target = center_cubes(img, target, crop=False)
        fig = _scroll_plot2([img, (0.4 * target + 0.6 * img)],
                            ['raw', 'target'])
        plt.savefig("/tmp/%i.png" % i, bbox_inches='tight')
        plt.show()
        # fig2 = scroll_plot(data.train_l[0][0], 'l')
