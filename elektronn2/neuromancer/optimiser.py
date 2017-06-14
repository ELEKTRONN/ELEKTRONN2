# -*- coding: utf-8 -*-
# ELEKTRONN2 Toolkit
# Copyright (c) 2015 Marius Killinger
# All rights reserved

from __future__ import absolute_import, division, print_function
from builtins import filter, hex, input, int, map, next, oct, pow, range, super, zip

import logging
import numpy as np
import theano.tensor as T

from . import graphutils
from . import variables

logger = logging.getLogger('elektronn2log')


class Optimiser(object):
    global_lr = variables.VariableParam(value=1,
                                        name='lr',
                                        dtype=graphutils.floatX)
    global_weight_decay = variables.VariableParam(value=0,
                                                  name='weight_decay',
                                                  dtype=graphutils.floatX)

    global_mom = variables.VariableParam(value=0.9,
                                          name='mom',
                                          dtype=graphutils.floatX)

    @classmethod
    def setlr(cls, val):
        """
        Set learning rate (global to all optimisers)
        """
        val = graphutils.as_floatX(val)
        cls.global_lr.set_value(val)

    @classmethod
    def setwd(cls, val):
        """
        Set weight decay parameter (global to all optimisers)
        """
        val = graphutils.as_floatX(val)
        cls.global_weight_decay.set_value(val)

    @classmethod
    def setmom(cls, val):
        """
        Set momentum parameter (global to all optimisers)
        """
        val = graphutils.as_floatX(val)
        cls.global_mom.set_value(val)

    def __init__(self, inputs, loss, grads, params, additional_outputs):
        if additional_outputs is None:
            additional_outputs = []

        self.meta_params = dict(lr=self.global_lr,
                                mom=self.global_mom,
                                wd=self.global_weight_decay)
        self.input = inputs
        self.output = [loss,] + additional_outputs
        self.loss   = loss
        self.params = params
        self.grads = grads
        self.step = None
        self.last_exec_time = None
        self.last_dir = []
        # the higher the index the older the params
        self.params_cycler = [self.alloc_shared_grads(name_suffix='_lp_%i'%i)
                              for i in range(3)]

    def alloc_shared_grads(self, name_suffix='_lg', init_val=0.0):
        """Returns new shared variables matching the shape of params/gradients"""
        grads = []
        for i, p in enumerate(self.params):
            name = p.name+name_suffix
            value = np.ones_like(p.get_value()) * graphutils.as_floatX(init_val)
            g = variables.VariableParam(value=value, name=name)
            grads.append(g)

        return grads

    def set_opt_meta_params(self, value_dict):
        """
        Update the meta-parameters via value dictionary
        """
        for k,v in value_dict.items():
            try:
                self.meta_params[k].set_value(v)
            except AttributeError:
                raise AttributeError

    def clear_last_dir(self, last_dir=None):
        if last_dir is None:
            last_dir = self.last_dir

        for d in last_dir:
            d.set_value(np.zeros(d.get_value().shape, dtype=d.dtype))


    def get_rotational_updates(self):
        updates = []
        for x in zip(self.params, *self.params_cycler):
            new_param, param_queue, = x[0], x[1:]
            for i in range(len(param_queue)-1, 0, -1):
                updates.append((param_queue[i], param_queue[i-1]))

            updates.append((param_queue[0], new_param))

        return updates


    def repair(self):
        self.clear_last_dir()
        for p, p_old in zip(self.params, self.params_cycler[-1]):
            p.set_value(p_old.get_value())


    def __call__(self, *args):
        """
        Perform an update step
        [data (,labels etc...)] --> [loss (, add. outputs...)]
        """
        ret = list(self.step(*args))
        ret[0] = graphutils.as_floatX(ret[0]) # the scalar loss
        self.last_exec_time = self.step.last_exec_time
        return ret



###############################################################################

class SGD(Optimiser):
    """
    SGD optimiser (See https://en.wikipedia.org/wiki/Stochastic_gradient_descent).
    """
    def __init__(self,  inputs, loss, grads, params, extra_updates,
                 additional_outputs=None):
        super(SGD, self).__init__(inputs, loss, grads, params,
                                  additional_outputs)
        self.last_dir     = self.alloc_shared_grads() # last direction os update

        updates = []
        for g, d, p in zip(self.grads, self.last_dir, self.params):
            new_d = g + self.global_mom * d
            if p.apply_reg:
                if p.apply_reg > 1:
                    multiplier = graphutils.as_floatX(p.apply_reg)
                    new_p = p - self.global_lr * \
                                (new_d + self.global_weight_decay * p * multiplier)
                else:
                    new_p = p - self.global_lr * \
                                (new_d + self.global_weight_decay * p)
            else:
                new_p = p - self.global_lr *  new_d

            updates.append((d, new_d))
            updates.append((p, new_p))

        updates.extend(extra_updates)
        updates.extend(self.get_rotational_updates())
        self.step = graphutils.make_func(self.input, self.output,
                                 updates=updates, name='SGD step')


class AdaGrad(Optimiser):
    """
    AdaGrad optimiser (See http://jmlr.org/papers/v12/duchi11a.html).

    Tries to favor making faster progress on parameters with usually small
    gradients (but does somehow ignore their actual direction, i.e. a parameter
    which has a lot of small gradients in the same direction and one that has
    many small gradients in opposite directions have both a high LR !
    """
    def __init__(self, inputs, loss, grads, params, extra_updates,
                 additional_outputs=None):
        super(AdaGrad, self).__init__(inputs, loss, grads, params,
                                      additional_outputs)

        self._init_done = False
        self.hs  = self.alloc_shared_grads('_h', init_val=0.0)
        updates = []
        for g, h, p in zip(self.grads, self.hs, self.params):
            new_h = h + T.square(g)
            if p.apply_reg: # apply to W but not b
                new_p = p - self.global_lr / T.sqrt(new_h) * \
                            (g + self.global_weight_decay * p)
            else:
                new_p = p - self.global_lr / T.sqrt(new_h) * g

            updates.append((h, new_h))
            updates.append((p, new_p))

        updates.extend(extra_updates)
        self.step = graphutils.make_func(self.input, self.output,
                                 updates=updates, name='AdaGrad step')

        # Create init_func to init h from one gradient evaluation
        updates = []
        for g, h in zip(self.grads, self.hs):
            new_h = h + T.square(g)
            updates.append((h, new_h))

        self.init_func = graphutils.make_func(self.input, [], updates=updates,
                                     name='AdaGrad initialiser')

    def __call__(self, *args):
        if not self._init_done:
            self.init_func(*args)
            self._init_done = True

        return super(AdaGrad, self).__call__(*args)


    def repair(self):
        super(AdaGrad, self).repair()
        self.clear_last_dir(self.hs)
        self._init_done = False

class AdaDelta(Optimiser):
    """
    AdaDelta optimiser (See https://arxiv.org/abs/1212.5701).

    Like AdaGrad, but accumulate squared only over window
    The delta part is some diagonal hessian approximation.
    Claims to be robust against sudden large gradients because then the
    denominator explodes, but this explosion is persistent for a while...
    (and this argumentation is true for any method accumulating squared grads).
    """
    def __init__(self,  inputs, loss, grads, params, extra_updates,
                 additional_outputs=None):
        super(AdaDelta, self).__init__(inputs, loss, grads, params,
                                       additional_outputs)
        self.squared_accum = self.alloc_shared_grads("_sq") # last directions update
        self.delta_accum = self.alloc_shared_grads("_d") # last directions update
        epsilon = 1e-5

        updates = []
        for g, s, d, p in zip(self.grads, self.squared_accum,
                              self.delta_accum, self.params):
            new_s = self.global_mom * s + (1.0 - self.global_mom) * T.square(g)
            direction = (g * T.sqrt(d + epsilon) / T.sqrt(s + epsilon))
            new_d = self.global_mom * d + (1 - self.global_mom) * T.square(direction)
            if p.apply_reg:
                if p.apply_reg > 1:
                    multiplier = graphutils.as_floatX(p.apply_reg)
                    new_p = p - self.global_lr * \
                                (direction + self.global_weight_decay * p * multiplier)
                else:
                    new_p = p - self.global_lr * \
                                (direction + self.global_weight_decay * p)
            else:
                new_p = p - self.global_lr *  direction

            updates.append((s, new_s))
            updates.append((d, new_d))
            updates.append((p, new_p))

        updates.extend(extra_updates)
        updates.extend(self.get_rotational_updates())
        self.step = graphutils.make_func(self.input, self.output,
                                 updates=updates, name='AdaDelta step')


    def repair(self):
        super(AdaDelta, self).repair()
        self.clear_last_dir(self.squared_accum)
        self.clear_last_dir(self.delta_accum)


class Adam(Optimiser):
    """
    Adam optimiser (See https://arxiv.org/abs/1412.6980v9).

    Like AdaGrad with windowed squared_accum and with momentum and a bias for
    the initial phase (t).
    The normalisation of Adam and AdaGrad (and RMSProp) does not damp but
    exaggerates sudden steep gradients (their squared_accum is small and their
    current grad is large).
    """
    def __init__(self,  inputs, loss, grads, params, extra_updates,
                 additional_outputs=None):
        super(Adam, self).__init__(inputs, loss, grads, params,
                                   additional_outputs)
        self.squared_accum = self.alloc_shared_grads("_sq") # last directions update
        self.momentum = self.alloc_shared_grads("_m") # last directions update
        epsilon = 1e-5

        # self.beta1 = variables.VariableParam(value=0.9, name='beta1',
        #                                      dtype=graphutils.floatX)
        self.beta2 = variables.VariableParam(value=0.999, name='beta2',
                                             dtype=graphutils.floatX)
        #self.meta_params['beta1'] = self.beta1
        self.meta_params['beta2'] = self.beta2

        t_old = variables.VariableParam(value=0.0, name='beta2',
                                        dtype=graphutils.floatX)

        updates = []
        t = 1 + t_old
        updates.append((t_old, t))
        factor = T.sqrt(1-self.beta2**t)/(1-self.global_mom**t)
        for g, s, m, p in zip(self.grads, self.squared_accum,
                              self.momentum, self.params):
            new_m = self.global_mom * m + (1.0 - self.global_mom) * g
            new_s = self.beta2      * s + (1.0 - self.beta2) * T.square(g)

            direction = factor * new_m / T.sqrt(new_s + epsilon)
            if p.apply_reg:
                if p.apply_reg > 1:
                    multiplier = graphutils.as_floatX(p.apply_reg)
                    new_p = p - self.global_lr * \
                                (direction + self.global_weight_decay * p * multiplier)
                else:
                    new_p = p - self.global_lr * \
                                (direction + self.global_weight_decay * p)
            else:
                new_p = p - self.global_lr *  direction

            updates.append((s, new_s))
            updates.append((m, new_m))
            updates.append((p, new_p))

        updates.extend(extra_updates)
        updates.extend(self.get_rotational_updates())
        self.step = graphutils.make_func(self.input, self.output,
                                 updates=updates, name='Adam step')

    def repair(self):
        super(Adam, self).repair()
        self.clear_last_dir(self.squared_accum)
        self.clear_last_dir(self.momentum)

class CG(Optimiser):
    pass
