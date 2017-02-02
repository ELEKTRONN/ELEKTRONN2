# -*- coding: utf-8 -*-
"""
Created on Wed Jun  1 15:12:45 2016

@author: Marius Felix Killinger
"""
from __future__ import absolute_import, division, print_function
from builtins import filter, hex, input, int, map, next, oct, pow, range, \
    super, zip

import numpy as np

from .. import neuromancer


act = 'tanh'

data = neuromancer.Input((30, 10, 20), 'r,b,f', name='data')
a, b = neuromancer.split(data, axis='r', index=1, name=['a', 'b'])
c, d = neuromancer.split(data, axis='r', index=1, strip_singleton_dims=True)

x = np.random.rand(30, 10, 20).astype(np.float32)

aa, bb = a(x), b(x)
cc, dd = c(x), d(x)

print(aa.shape, bb.shape, cc.shape, dd.shape)

model = neuromancer.model_manager.current
model.designate_nodes(input_node=data)

# , target_node=target, loss_node=loss,
#                                  prediction_node=pred,
#                                  prediction_ext=[loss, nll_barr, pred],
#                                  debug_outputs =[nll_barr, nll_obj, nll_branch, mse_skel])

model.save("/tmp/test.mdl")
model2 = neuromancer.model.modelload("/tmp/test.mdl")
