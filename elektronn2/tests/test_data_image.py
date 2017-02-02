from __future__ import absolute_import, division, print_function
from builtins import filter, hex, input, int, map, next, oct, pow, range, \
    super, zip

from ..data import image
from ..utils import h5load, h5save, timeit
from ..utils.plotting import scroll_plot, _scroll_plot2, _scroll_plot4

import numpy as np
import matplotlib.pyplot as plt


def test_data_image():
    ids = h5load(
        '/home/mfk/axon/mkilling/devel/data/BirdGT/j0126_bird_new/j0126 cubeSegmentor category-volume_gt__016-kkiesl-20150913-205217-final.h5')
    ids, raw = ids[1], ids[0]

    ids = np.transpose(ids, (2, 0, 1))
    raw = np.transpose(raw, (2, 0, 1))

    ids2 = np.unique(ids, return_inverse=True)[1].reshape(ids.shape)
    barrier = image.ids2barriers(ids, dilute=[0, 0, 0], ecs_as_barr=True)

    barrier, raw = image.center_cubes(barrier, raw, crop=True)
    barrier2 = image.smearbarriers(barrier, kernel=(3, 5, 5))  # (z,x,y)

    # fig, scroller = _scroll_plot2([barrier2, (0.75*raw+0.25*255*(1-barrier))], ['ids', 'raw'])
    # fig, scroller = _scroll_plot2([barrier2.T, (0.75*raw.T+0.25*255*(1-barrier.T))], ['ids', 'raw'])
    # fig, scroller = _scroll_plot2([barrier2.T, barrier.T], ['ids', 'raw'])
    # fig, scroller = _scroll_plot2([barrier2, barrier], ['ids', 'raw'])
    fig, scroller = _scroll_plot4(
        [barrier2, barrier, (barrier2 > 0.5) - barrier, barrier2 > 0.5],
        ['bar1', 'bar2', 'ids', 'raw'])
    fig.show()

    seg = image.seg_old(barrier2 * 255)

    fig, scroller = _scroll_plot4([barrier2, seg, ids2, raw],
                                  ['bar1', 'seg', 'ids', 'raw'])
    fig.show()

    ri = timeit(image.rand_index)
    print(ri(ids2, seg))  # ~ 0.99

    fig, scroller = scroll_plot(raw.T, ['raw'])
    fig.show()
