# -*- coding: utf-8 -*-
"""
Created on Fri May  6 23:48:05 2016

@author: Marius Felix Killinger
"""
from __future__ import absolute_import, division, print_function
from builtins import filter, hex, input, int, map, next, oct, pow, range, \
    super, zip

import numpy as np

from .. import utils


# x = np.random.rand(21,2)
# kdt = utils.DynamicKDT(k=2, rebuild_thresh=3)
#
# kdt.append(x[0])
# kdt.append(x[1])
# kdt.append(x[2])
#
# dist, ind, co = kdt.get_knn(x[3])
# kdt.append(x[3])
#
# dist, ind, co = kdt.get_knn(x[4])
# kdt.append(x[4])
#
# kdt.append(x[5])
# kdt.append(x[6])
#
# dist, ind, co = kdt.get_knn(x[7])
# kdt.append(x[7])



for n in range(20, 100, 10):
    x = np.random.rand(500, 2)
    kdt = utils.DynamicKDT(k=4, rebuild_thresh=n)
    for i in range(5):
        kdt.append(x[i])

    for i in range(5, 500):
        dist, ind, co = kdt.get_knn(x[i])

        kdt_ref = utils.KDT(n_neighbors=4)
        kdt_ref.fit(x[:i])
        distances, indices = kdt_ref.kneighbors(x[i])

        err = abs(dist - distances).max()
        if err > 1e-7:
            print(n, i, err)
            dist, ind, co = kdt.get_knn(x[i])

        kdt.append(x[i])



# for n in range(100):
#    x = np.random.rand(21,2)
#    a = x[:10]
#    b = x[10:15]
#    kdt = utils.DynamicKDT(points=a, k=4)
#    for i in range(15,21):
#        kdt.append(x[i])
#
#    dist, ind, co = kdt.get_knn(b)





# x = np.random.rand(2000,2)
# distances = np.zeros((2000,20))
#
# tt = utils.Timer()
# for thresh in np.logspace(0.7,3.5,10).astype(np.int):
#    for i in range(5):
#        kdt = utils.DynamicKDT(points=x[:20], k=20, rebuild_thresh=thresh)
#        for j in range(20,2000):
#            ret = kdt.get_knn(x[j:j+10])
#            distances[j] = ret[0]
#            kdt.append(x[j])
#
#
#    tt.check("%i"%thresh)
#
# tt.plot()
