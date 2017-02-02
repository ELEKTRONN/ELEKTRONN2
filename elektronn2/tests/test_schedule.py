# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 16:57:36 2016

@author: Marius Felix Killinger
"""
from __future__ import absolute_import, division, print_function
from builtins import filter, hex, input, int, map, next, oct, pow, range, \
    super, zip

import numpy as np
import matplotlib.pyplot as plt
import theano

from ..training.trainutils import Schedule


def test_schedule():
    lr = theano.shared(0.1)
    sch = Schedule(dec=0.99,
                   updates=[(1000, 0.01), (10000, 0.01), (50000, 0.01)])
    sch.bind_variable(lr)

    vals = []
    for i in range(100000):
        vals.append(lr.get_value())
        if i==sch.next_update:
            sch.update(i)

    vals = np.array(vals)
    plt.plot(vals)


    class A(object):
        def __init__(self):
            self.n = 10
            self._i = 100

        @property
        def i(self):
            return self._i

        @i.setter
        def i(self, val):
            self._i = val


    a = A()

    sch = Schedule(dec=1.0,
                   updates=[(1000, 0.01), (10000, 0.01), (50000, 0.01)])
    sch.bind_variable(obj=a, prop_name='n')

    vals = []
    for i in range(100000):
        vals.append(a.n)
        if i==sch.next_update:
            sch.update(i)

    vals = np.array(vals)
    plt.plot(vals)
