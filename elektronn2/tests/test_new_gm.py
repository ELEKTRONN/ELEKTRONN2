# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 19:00:16 2016

@author: Marius Felix Killinger
"""
from __future__ import absolute_import, division, print_function
from builtins import filter, hex, input, int, map, next, oct, pow, range, \
    super, zip

import theano.tensor as T

from .. import neuromancer
from ..neuromancer.model import modelload


def test_new_gm():
    mfp = False
    in_sh = (1, 1, 15, 198, 198) if mfp else (1, 1, 25, 171, 171)
    inp = neuromancer.Input(in_sh, 'b,f,z,x,y', name='raw')

    out = neuromancer.Conv(inp, 20, (1, 6, 6), (1, 1, 1), mfp=mfp,
                           batch_normalisation='train')
    out = neuromancer.Conv(out, 40, (1, 5, 5), (1, 2, 2), mfp=mfp)
    out = neuromancer.Conv(out, 50, (1, 4, 4), (1, 2, 2), mfp=mfp)
    out = neuromancer.Conv(out, 80, (1, 4, 4), (1, 1, 1), mfp=mfp)

    out = neuromancer.Conv(out, 80, (4, 1, 1), (2, 1, 1),
                           mfp=mfp)  # first z kernel, 2 pool
    out = neuromancer.Conv(out, 80, (3, 4, 4), (1, 1, 1), mfp=mfp)
    out = neuromancer.Conv(out, 80, (3, 4, 4), (1, 1, 1), mfp=mfp)
    out = neuromancer.Conv(out, 100, (2, 4, 4), (1, 1, 1), mfp=mfp)

    out = neuromancer.Conv(out, 120, (2, 4, 4), (1, 1, 1), mfp=mfp)
    out = neuromancer.Conv(out, 120, (1, 2, 2), (1, 1, 1), mfp=mfp)

    out = neuromancer.Conv(out, 120, (1, 1, 1), (1, 1, 1), mfp=mfp)
    out1, out2 = neuromancer.split(out, 1, n_out=2)

    probs = neuromancer.Conv(out1, 2, (1, 1, 1), (1, 1, 1), mfp=mfp,
                             activation_func='lin')
    probs = neuromancer.Softmax(probs, name='probs')
    discard, mode = neuromancer.split(probs, 1, n_out=2)

    concentration = neuromancer.Conv(out2, 1, (1, 1, 1), (1, 1, 1), mfp=mfp,
                                     activation_func='lin',
                                     name='concentration')
    t_sh = probs.shape.copy()
    t_sh.updateshape('f', 1)
    target = neuromancer.Input_like(t_sh, dtype='float32', name='target')

    loss_pix = neuromancer.BetaNLL(mode, concentration, target)
    loss = neuromancer.AggregateLoss(loss_pix)
    errors = neuromancer.Errors(probs, target, target_is_sparse=True)
    prediction = neuromancer.Concat([mode, concentration], axis=1,
                                    name='prediction')

    loss_std = neuromancer.ApplyFunc(loss_pix, T.std)

    model = neuromancer.model_manager.getmodel()
    model.designate_nodes(input_node=inp, target_node=target, loss_node=loss,
                          prediction_node=prediction,
                          prediction_ext=[loss, errors, prediction])

    ### --- ###

    model2 = neuromancer.model_manager.newmodel("second")
    inp2 = neuromancer.Input(in_sh, 'b,f,z,x,y', name='raw')

    out2 = neuromancer.Conv(inp2, 20, (1, 6, 6), (1, 1, 1), mfp=mfp)
    out2 = neuromancer.Conv(out2, 40, (1, 5, 5), (1, 2, 2), mfp=mfp)
    out2 = neuromancer.Conv(out2, 50, (1, 4, 4), (1, 2, 2), mfp=mfp)

    out2 = neuromancer.Conv(out2, 120, (2, 4, 4), (1, 1, 1), mfp=mfp)
    out2 = neuromancer.Conv(out2, 120, (1, 2, 2), (1, 1, 1), mfp=mfp)

    out2 = neuromancer.Conv(out2, 120, (1, 1, 1), (1, 1, 1), mfp=mfp)

    probs2 = neuromancer.Conv(out2, 2, (1, 1, 1), (1, 1, 1), mfp=mfp,
                              activation_func='lin')
    probs2 = neuromancer.Softmax(probs2, name='probs')
    t_sh = probs2.shape.copy()
    t_sh.updateshape('f', 1)
    target2 = neuromancer.Input_like(t_sh, dtype='float32', name='target')

    loss_pix2 = neuromancer.MultinoulliNLL(probs2, target2)
    loss2 = neuromancer.AggregateLoss(loss_pix2)
    errors2 = neuromancer.Errors(probs2, target2, target_is_sparse=True)
    model2.designate_nodes(input_node=inp2, target_node=target2,
                           loss_node=loss2, prediction_node=probs2,
                           prediction_ext=[loss2, errors2, probs2])

    model.save('/tmp/test.pkl')
    model2.save('/tmp/test2.pkl')
    model2_reloaded = modelload('/tmp/test2.pkl')
    model2_reloaded.save('/tmp/test2_reloaded.pkl')

    print(neuromancer.model_manager)
