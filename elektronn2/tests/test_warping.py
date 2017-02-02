from __future__ import absolute_import, division, print_function
from builtins import filter, hex, input, int, map, next, oct, pow, range, \
    super, zip

import numpy as np
from scipy.misc import imsave


def test_warping():
    im = wa.maketestimage((256, 256))
    imsave('/tmp/a.png', im)

    # 3D:
    im3 = np.ones((1, 256, 256))
    im3[0, :, :] = im
    wim3 = wa.warp2dFast(im3, (192, 192), rot=30, shear=10, scale=(2, .5),
                         stretch=(.9, .7))
    imsave('/tmp/b.png', wim3[0, :, :])

    # 4D:
    im4 = np.ones((1, 4, 256, 256))
    im4[0, 0, :, :] = im
    im4[0, 1, :, :] = im * im
    im4[0, 2, :, :] = im
    im4[0, 3, :, :] = im

    # wim4 = wa.warp3dFast(im4,
    #                      (4, 256, 256),
    #                      rot=20,
    #                      shear=50,
    #                      scale=(.7, .8, .9),
    #                      stretch=(.9, .7, .4))
    wim4 = wa.warp3dFast(im4, (4, 256, 256), rot=20, shear=10)

    # print(`wim4`)

    imsave('/tmp/c0.png', wim4[0, 0, :, :])
    imsave('/tmp/c1.png', wim4[0, 1, :, :])
    imsave('/tmp/c2.png', wim4[0, 2, :, :])
