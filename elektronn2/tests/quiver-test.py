# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 10:11:05 2016

@author: Marius Felix Killinger
"""
from __future__ import absolute_import, division, print_function
from builtins import filter, hex, input, int, map, next, oct, pow, range, \
    super, zip

import numpy as np
import matplotlib.pyplot as plt
import h5py

from ..data import transformations
from ..utils.plotting import scroll_plot


# from mayavi import mlab  # TODO: Dependency?

def my_quiver(x, y, img=None, c=None):
    """
    first dim of x,y changes along vertical axis
    second dim changes along horizontal axis
    x: vertical vector component
    y: horizontal vector component
    """
    figure = plt.figure(figsize=(7, 7))

    if img is not None:
        plt.imshow(img, interpolation='none', alpha=0.22, cmap='gray')

    plt.quiver(x, y, c, angles='xy', units='xy', cmap='spring', pivot='middle',
               scale=0.5)
    return figure


data = h5py.File(
    "/home/mfk/lustre/mkilling/BirdGT/skeleton/rel_direc+branch+barr+obj_zyx.h5")[
    'combi']
stack = data[:, 50:50 + 50, 50:300 + 50, 50:300 + 50]

# z_i =5
# barr = stack[4,z_i]
# x = stack[2,z_i]
# y = stack[1,z_i]
# z = stack[0,z_i]
# my_quiver(x, y, barr, z)



# stack has spatial shape (f,z,y,x)
# stack[0]: how the vec field changes along z, i.e. 0 spatial dim
# stack[1]: how the vec field changes along y, i.e. 1 spatial dim
# stack[2]: how the vec field changes along x, i.e. 2 spatial dim

direction_iso = [1, 0, 0]
direction_iso /= np.linalg.norm(direction_iso)

gamma = 0 * np.pi / 180

img_new, target_new, M = transformations.get_tracing_slice(stack[4:5],
                                                           [10, 80, 80],
                                                           [25, 150, 150],
                                                           direction_iso=direction_iso,
                                                           gamma=gamma,
                                                           target=stack,
                                                           target_ps=[10, 80,
                                                                      80],
                                                           target_vec_ix=[
                                                               [0, 1, 2]],
                                                           target_discrete_ix=[
                                                               3, 4, 5, 6, 7,
                                                               8])

z_i = 2
# barr = target_new[4,z_i]
# x = target_new[2,z_i]
# y = target_new[1,z_i]
# z = target_new[0,z_i]
# my_quiver(x, y, barr, z)
#
# zz,yy,xx = np.mgrid[:5:1,:40:1,:40:1]
# mlab.quiver3d(zz,yy,xx, target_new[0, ::2, ::2, ::2], target_new[1, ::2, ::2, ::2], target_new[2, ::2, ::2, ::2])

fig = scroll_plot(target_new[:, z_i], 'channels')
