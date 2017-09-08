# -*- coding: utf-8 -*-
# ELEKTRONN2 Toolkit
# Copyright (c) 2015 Marius Killinger
# All rights reserved

from __future__ import absolute_import, division, print_function
from builtins import filter, hex, input, int, map, next, oct, pow, range, super, zip


import sys
import logging
import getpass
from collections import OrderedDict
import uuid

import numpy as np
import theano
import theano.tensor as T

from ..config import config
from .. import utils
from . import node_basic, neural, loss
from . import optimiser
from . import graphmanager
from . import graphutils

logger = logging.getLogger('elektronn2log')

user_name = getpass.getuser()

if sys.version_info.major >= 3:
    unicode = str

__all__ = ['Model', 'modelload', 'kernel_lists_from_node_descr',
           'params_from_model_file', 'rebuild_model', 'simple_cnn']
#TODO break all feed backs
#TODO docstrings for rebuild etc.

class Model(graphmanager.GraphManager):
    """
    Represents a neural network model and its current training state.

    The Model is defined and checked by running Model.designate_nodes() with
    appropriate Nodes as arguments (see example in examples/numa_mnist.py).

    Permanently saving a Model with its respective training state is possible
    with the Model.save() function.
    Loading a Model from a file is done by elektronn2.neuromancer.model.modelload().

    During training of a neural network, you can access the current Model
    via the interactive training shell as the variable "model"
    (see elektronn2.training.trainutils.user_input()).
    There are several statistics and hyperparameters of the Model that you
    can inspect and set directly in the shell, e.g. entering
    >>> model.lr = 1e-3
    and exiting the prompt again effectively sets the learning rate to 1e-3
    for further training.
    (This can also be done with the shortcut "setlr 1e-3".)
    """
    def __init__(self, name=""):
        super(Model, self).__init__(name=name)

        self.batch_size = None
        self.ndim = None
        self._desig_descr = dict()

        # Tracking / Monitoring etc
        self.iterations   = 0
        self.elapsed_time = 0
        self._last_exec_times = utils.CircularBuffer(
            config.time_per_step_smoothing_length)

        self._last_losses = utils.CircularBuffer(config.loss_smoothing_length)

        # Node stuff
        self.prediction_node = None
        self.prediction_ext = None
        self._prediction_ext_func = None

        self.loss_node = None
        self.target_node = None
        self.error_node = None
        self.input_node = None

        self.trainable_params = None
        self.nontrainable_params = None
        self._grad = None
        self._grad_inp = None
        self._grad_func = None
        self.optimisers = dict()
        self.debug_outputs = []

        self._activation_func = None

    def designate_nodes(self, input_node='input', target_node=None,
                        loss_node=None, prediction_node=None,
                        prediction_ext=None, error_node=None,
                         debug_outputs=None):
        """
        Register input, target and other special Nodes in the Model.

        Most of the Model's attributes are derived from the Nodes that are
        given as arguments here.
        """

        def designate_nodes(purpose, name): # Does nothing if name is None
            if isinstance(name, (list, tuple)):
                if purpose not in ['debug_outputs', 'prediction_ext']:
                    raise ValueError("Can only designate several nodes for "
                                     "debug outputs and prediction_ext")
                name = [n if isinstance(n, (str,unicode)) else n.name for n in name]
                nodes = []
                for n in name:
                    nodes.append(self.nodes[n])

                setattr(self, purpose, nodes)

            elif name:
                name = name if isinstance(name, (str, unicode)) else name.name
                setattr(self, purpose, self.nodes[name])

            self._desig_descr[purpose]=name # for reconstruction

        if debug_outputs is None:
            debug_outputs = []

        # Must use same names (second arg) as in kwargs of this function
        designate_nodes('input_node', input_node)
        designate_nodes('target_node', target_node)
        designate_nodes('loss_node', loss_node)
        designate_nodes('prediction_node', prediction_node)
        designate_nodes('error_node', error_node)

        designate_nodes('prediction_ext', prediction_ext)
        designate_nodes('debug_outputs', debug_outputs)

        # Prediction
        if self.prediction_node:
            self.batch_size = self.prediction_node.shape['b']
            self.ndim = self.prediction_node.shape.ndim
            if np.any(np.less(self.prediction_node.shape.fov, 0)): # If UpConvs contained
                in_sh = self.input_node.shape.spatial_shape
                sh = np.array(self.prediction_node.shape.spatial_shape)
                st = np.array(self.prediction_node.shape.strides)
                out_sh = st * (sh-1) + 1
                diff = np.subtract(in_sh, out_sh)
                if np.any(np.mod(diff, 2)):
                    raise ValueError("FOV is not centered. In_sh=%s, "
                    "out_sh*strides=%s, diff=%s"
                                     %(in_sh, out_sh, diff))
                self.prediction_node.shape._fov = diff # Hack
                self.target_node.shape._fov = diff     # Hack

            elif not self.prediction_node.shape.fov_all_centered:
                logger.warning("Not all field of views are centered (odd) "
                "this might cause problems for many setups")
            logger.info("Prediction properties:\n%s"\
                   %(self.prediction_node.shape.ext_repr))

            if not self.loss_node:
                n_param = self.prediction_node.all_params_count
                n_comp  = self.prediction_node.all_computational_cost
                n_comp_specific = float(n_comp)/self.prediction_node.shape.spatial_size
                logger.info("\nTotal Computational Cost of Model: {0:s}\n"
                            "Total number of trainable parameters: {1:,d}.\n"
                            "Computational Cost per pixel: {2:s}\n".
                            format(utils.pretty_string_ops(n_comp), n_param,
                                   utils.pretty_string_ops(n_comp_specific)))

        # Multiple output extended prediction
        if self.prediction_ext:
            outs = [n.output for n in self.prediction_ext]
            inp = graphutils.getinput_for_multioutput(self.prediction_ext)
            self._prediction_ext_func = graphutils.make_func(inp, outs,
                                                    name='Predictor Extended')
        # Loss, Model Parameters and their gradients, Optimisation
        if self.loss_node:
            self.trainable_params = list(self.loss_node.all_trainable_params.values()) # values of dict, not of shared!
            self.nontrainable_params = self.loss_node.all_nontrainable_params
            extra_updates          = self.loss_node.all_extra_updates

            self._grad = T.grad(self.loss_node.output, self.trainable_params,
                                disconnected_inputs="warn")
            self._grad_inp = self.loss_node.input_tensors
            self._grad_func = graphutils.make_func(self._grad_inp, self._grad,
                                                name='Gradient Func')

            debug_outputs = [n.output for n in self.debug_outputs]

            # Init *all* optimisers
            opt_init = (self._grad_inp, self.loss_node.output, self._grad,
                        self.trainable_params, extra_updates, debug_outputs)


            # TODO: self.optimisers should automatically register all subclasses of Optimiser
            self.optimisers = dict(SGD=optimiser.SGD(*opt_init),
                                   AdaGrad=optimiser.AdaGrad(*opt_init),
                                   AdaDelta=optimiser.AdaDelta(*opt_init),
                                   Adam=optimiser.Adam(*opt_init))

            n_param = self.loss_node.all_params_count
            n_comp  = self.loss_node.all_computational_cost
            if self.prediction_node:
                n_comp_specific = float(n_comp) / self.prediction_node.shape.spatial_size
            else:
                n_comp_specific = float(n_comp)/self.loss_node.shape.spatial_size
            logger.info("\nTotal Computational Cost of Model: {0:s}\n"
                        "Total number of trainable parameters: {1:,d}.\n"
                        "Computational Cost per pixel: {2:s}\n".
                        format(utils.pretty_string_ops(n_comp), n_param,
                               utils.pretty_string_ops(n_comp_specific)))


        # Activations (outputs os all intermediate layers
        activations = []
        activations_tt = []
        for node in self.nodes.values():
            # Split descriptors are stored as strings amongst the nodes
            if not isinstance(node, str) and not node.is_source:
                # multi output will apper in from-tensor nodes
                if not isinstance(node.output, (list,tuple)):
                    activations.append(node)
                    activations_tt.append(node.output)

        inp = graphutils.getinput_for_multioutput(activations)
        self._activation_func = graphutils.make_func(inp, activations_tt,
                                                        name='Activations')

    def save(self, file_name):
        """
        Save a Model (including its training state) to a pickle file.
        :param file_name: File name to save the Model in.
        """
        descriptors = self.serialise()
        utils.picklesave((descriptors, self._desig_descr), file_name)

    def loss(self, *args, **kwargs):
        return self.loss_node(*args, **kwargs)

    def gradients(self, *args, **kwargs):
        return self._grad_func(*args, **kwargs)

    def activations(self, *args, **kwargs):
       return self._activation_func(*args, **kwargs)

    def predict(self, *args, **kwargs):
        return self.prediction_node(*args, **kwargs)

    def predict_ext(self, *args, **kwargs):
        return self._prediction_ext_func(*args, **kwargs)

    def predict_dense(self, raw_img, as_uint8=False, pad_raw=False):
        return self.prediction_node.predict_dense(raw_img,
                                                  as_uint8=as_uint8,
                                                  pad_raw=pad_raw)

    def paramstats(self):
        print("Parameter statistics")
        for k, W in self.loss_node.all_trainable_params.items():
            W=W.get_value()
            print("Param %s:\tshape=%s,\tmean=%f,\tstd=%f,\tmedian(abs)=%f"\
                 %(k, W.shape, W.mean(), W.std(), np.median(np.abs(W))))


    def gradstats(self, *args, **kwargs):
        grads = self.gradients(*args, **kwargs)
        print("Gradient statistics")
        for g in grads:
            print("\tshape=%s,\tmean=%f,\tstd=%f,\tmedian(abs)=%f"\
                  %(g.shape, np.mean(g), np.std(g), np.median(np.abs(g))))

    def actstats(self, *args, **kwargs):
        acts = self.activations(*args, **kwargs)
        print("Activation statistics")
        for a in acts:
            print("\tshape=%s,\tmean=%f,\tstd=%f,\tmedian(abs)=%f"\
                  %(a.shape, np.mean(a), np.std(a), np.median(np.abs(a))))

    def set_opt_meta_params(self, opt_name, value_dict):
        self.optimisers[opt_name].set_opt_meta_params(value_dict)

    @property
    def lr(self):
        """
        Get learning rate.
        """
        return list(self.optimisers.values())[0].global_lr.get_value()

    @lr.setter
    def lr(self, val):
        """
        Set learning rate.
        """
        list(self.optimisers.values())[0].setlr(val)

    @property
    def mom(self):
        """
        Get momentum.
        """
        return list(self.optimisers.values())[0].global_mom.get_value()

    @mom.setter
    def mom(self, val):
        """
        Set momentum.
        """
        list(self.optimisers.values())[0].setmom(val)

    @property
    def wd(self):
        """
        Get weight decay.
        """
        return list(self.optimisers.values())[0].global_weight_decay.get_value()

    @wd.setter
    def wd(self, val):
        """
        Set weight decay.
        """
        list(self.optimisers.values())[0].setwd(val)

    @property
    def mixing(self):
        """
        Get mixing weights.
        """
        if isinstance(self.loss_node, loss.AggregateLoss):
            return self.loss_node.mixing_weights.get_value()
        else:
            try:
                for n in self.nodes.values():
                    if isinstance(n, loss.AggregateLoss):
                        break

                logger.info("model.loss_node is not of type AggregateLoss and "
                               "hence has no mixing_weights. But '%s' is, so this is "
                               "used." %n.name)
                return n.mixing_weights.get_value()
            except:
                logger.error("no mixing_weights found in model")

    @mixing.setter
    def mixing(self, val):
        """
        Set mixing weights.
        """
        val = graphutils.as_floatX(val)
        if isinstance(self.loss_node, loss.AggregateLoss):
            self.loss_node.mixing_weights.set_value(val)
        else:
            try:
                for n in self.nodes.values():
                    if isinstance(n, loss.AggregateLoss):
                        break

                logger.info("model.loss_node is not of type AggregateLoss and "
                               "hence has no mixing_weights. But '%s' is, so this is "
                               "used." % n.name)
                n.mixing_weights.set_value(val)
            except:
                logger.error("no mixing_weights found in model")

    @property
    def dropout_rates(self):
        """
        Get dropout rates.
        """
        dropout_rates = []
        for node in self.nodes.values():
            r = node.params.get('dropout_rate', None)
            if r:
                dropout_rates.append(r.get_value())

        return np.array(dropout_rates)

    @dropout_rates.setter
    def dropout_rates(self, rates):
        """
        Set dropout rates.

        If the argument is a number, all Nodes that support it are given this
        dropout rate.
        If the argument is a tuple, list or array, the Nodes in the Model
        that support it are given the rates in their respective ordering.
        """
        i = 0
        for node in self.nodes.values():
            r = node.params.get('dropout_rate', None)
            if r:
                if isinstance(rates, (tuple, list, np.ndarray)):
                    r.set_value(graphutils.as_floatX(rates[i]))
                else:
                    r.set_value(graphutils.as_floatX(rates))
                i += 1

    @property
    def gradnet_rates(self):
        """
        Get gradnet rates.

        Description: https://arxiv.org/abs/1511.06827
        """
        gradnet_rates = []
        for node in self.nodes.values():
            r = node.params.get('gradnet_rate', None)
            if r:
                gradnet_rates.append(r.get_value())

        return gradnet_rates

    @gradnet_rates.setter
    def gradnet_rates(self, rates):
        """
        Set gradnet rates.

        Description: https://arxiv.org/abs/1511.06827

        If the argument is a number, all Nodes that support it are given this
        gradnet rate.
        If the argument is a tuple, list or array, the Nodes in the Model
        that support it are given the rates in their respective ordering.
        """
        i = 0
        for node in self.nodes.values():
            r = node.params.get('gradnet_rate', None)
            if r:
                if isinstance(rates, (tuple, list, np.ndarray)):
                    r.set_value(graphutils.as_floatX(rates[i]))
                else:
                    r.set_value(graphutils.as_floatX(rates))
                i += 1

    @property
    def batch_normalisation_active(self):
        """
        Check if batch normalisation is active in any Node in the Model.
        """
        active = 0
        for node in self.nodes.values():
            if hasattr(node, 'batch_normalisation'):
                if node.batch_normalisation in ['train', 'fadeout']:
                    active += 1

        return (active > 0)


    @property
    def debug_output_names(self):
        """
        If debug_outputs is set, a list of all debug output names is returned.
        """
        if self.debug_outputs:
            return [x.name for x in self.debug_outputs]
        else:
            return None


    @property
    def prediction_feature_names(self):
        """
        If a prediction node is set, return its feature names.
        """
        ret = None
        if self.prediction_node:
            ret = self.prediction_node.feature_names

        return ret




    def get_param_values(self, skip_const=False, as_list=False):
        """
        Only use this to save/load parameters!

        Returns a dict of mapping the values of the params
        (such that they can be saved to disk)
        :param skip_const: whether to exclude constant parameters
        """
        p_dict = OrderedDict()
        for name,node in self.nodes.items():
            p_dict[name] = node.get_param_values(skip_const)

        ret = list(p_dict.values()) if as_list else p_dict
        return ret


    def set_param_values(self, value_dict, skip_const=False):
        """
        Only use this to save/load parameters!

        Sets new values for non constant parameters
        :param value_dict: dict mapping values by parameter name / or file name thereof
        :param skip_const: if dict also maps values for constants, these
        can be skipped, otherwise an exception is raised.
        """
        logger.info("Setting model parameters")
        if isinstance(value_dict, str):
            value_dict = utils.pickleload(value_dict)

        if isinstance(value_dict, dict):
            for k,v in value_dict.items():
                if k not in self.nodes:
                    raise KeyError("Graph Manager has no node %s" %(k,))
                self.nodes[k].set_param_values(v, skip_const)
        else:
            for p,n in zip(value_dict, self.nodes.values()):
                if not isinstance(n, str):
                    n.set_param_values(p, skip_const)

    @property
    def loss_input_shapes(self):
        """
        Get shape(s) of loss nodes' input node(s).

        The return value is either a shape (if one input) or a list of shapes
        (if multiple inputs).
        """
        sources = self.loss_node.input_nodes
        if len(sources)==0:
            return sources[0].shape
        else:
            return [s.shape for s in sources]

    @property
    def time_per_step(self):
        """
        Get average run time per training step.

        The average is calculated over the last n steps, where n is defined by
        the config variable time_per_step_smoothing_length (default: 50).
        """
        return self._last_exec_times.mean() + 1e-6

    @property
    def loss_smooth(self):
        """
        Get average loss during the last training steps.

        The average is calculated over the last n steps, where n is defined by
        the config variable time_per_step_smoothing_length (default: 50).
        """
        return self._last_losses.mean()


    def trainingstep(self, *args, **kwargs):
        """
        Perform one optimiser iteration.
        Optimisers can be chosen by the kwarg ``optimiser``.

        **Signature**: ``trainingstep(data, target(, *aux)(, **kwargs**))``

        Parameters
        ----------
        *args
            * data: floatX array
                  input [bs, ch (, z, y, x)]
            * targets: int16 array
                  [bs,((z,)y,x)] (optional)
            * (optional) other inputs: np.ndarray
                  depending in model

        **kwargs
            * optimiser: str
                  Name of the chosen optimiser class in
                  :py:mod:`elektronn2.neuromancer.optimiser`
            * update_loss: Bool
                 determine current loss *after* update step
                 (e.g. needed for queue, but ``get_loss`` can also be
                 called explicitly)

        Returns
        -------
        loss: floatX
            Loss (nll, squared error etc...)
        t: float
            Time spent on the GPU per step
        """

        opt_name = kwargs.get('optimiser', 'SGD')
        if opt_name not in self.optimisers:
            logger.warning("No optimiser '%s'. Falling back to SGD"%(opt_name,))

        ret = self.optimisers[opt_name](*args)
        loss = ret[0]
        if kwargs.get('update_loss', False):
            loss = self.loss(*args)

        t  = self.optimisers[opt_name].last_exec_time
        self.elapsed_time += t
        self._last_exec_times.append(t + 1e-10) # add some epsilon to ensure > 0
        self._last_losses.append(loss)
        self.iterations += 1

        if len(ret)>1:
            return loss, t, ret[1:]
        else:
            return loss, t, None

    def test_run_prediction(self):
        """
        Execute test run on the prediction node.
        """
        self.prediction_node.test_run()

    def measure_exectimes(self, n_samples=5, n_warmup=4, print_info=True):
        """
        Return an OrderedDict that maps node names to their estimated execution times in milliseconds.

        Parameters are the same as in elektronn2.neuromancer.node_basic.Node.measure_exectime()
        """
        exectimes = OrderedDict()
        for name, node in self.nodes.items():
            exectime = node.measure_exectime(n_samples=n_samples, n_warmup=n_warmup, print_info=print_info)
            # exec_time = node.local_exec_time
            exectimes[name] = exectime
        return exectimes

###############################################################################

def modelload(file_name, override_mfp_to_active=False,
              imposed_patch_size=None, imposed_batch_size=None,
              name=None, **model_load_kwargs):
    """
    Load a Model from a pickle file (created by Model.save()).

    model_load_kwargs: remove_bn, make_weights_constant (True/False)
    """  # TODO: Document params

    logger.info("Loading model from %s" %file_name)
    node_descr, desig_descr = utils.pickleload(file_name)

    if 'input_node' not in desig_descr:
        if override_mfp_to_active or imposed_patch_size is not None or\
            imposed_batch_size is not None:
            raise("To use 'override_mfp_to_active' or 'imposed_patch_size', the"
                  " saved model must have a designated 'input_node'")
    else:
        input_node_descr = node_descr[desig_descr['input_node']][0]
        if isinstance(input_node_descr.args[1], graphutils.TaggedShape):
            old_shape = input_node_descr.args[1] # Input node is "FromTensor"-Type
        else:
            old_shape = graphutils.TaggedShape(input_node_descr.args[0],
                                       input_node_descr.args[1],
                                       strides=input_node_descr.kwargs.get('strides', None),
                                       fov=input_node_descr.kwargs.get('fov', None))
        old_patch_size = old_shape.spatial_shape

        ### Changing the shape ###
        changed = False
        if imposed_batch_size is not None:
            changed = True
            old_shape = old_shape.updateshape('b', imposed_batch_size)

        filter_shapes, pool_shapes, mfps = kernel_lists_from_node_descr(node_descr)
        if override_mfp_to_active:

            if imposed_patch_size is None:
                imposed_patch_size = old_patch_size
                # raise ValueError("'override_mfp_to_active' is True but the "
                #                  "patch_size is not updated. This does not work.")

            mfps = [True for mfp in mfps]
            model_load_kwargs['override_mfp_to_active'] = True

            ### Inject a FragmentsToDense node prior to softmax ###
            from elektronn2.neuromancer.neural import FragmentsToDense
            pred_name = desig_descr['prediction_node']
            pred_descr = node_descr[pred_name][0]
            ###TODO inject after prediction_node because this does not work for Concat nodes as prediction node. But then we need to change desig_descr???
            # create descriptor for a FragmentsToDense prior to softmax node
            # make the 'old' parent of pred, the parent of the reshape
            guard = str(uuid.uuid4())[:5]
            reshape_descr = graphmanager.NodeDescriptor([pred_descr.args[0]],
                                           dict(name='to_dense'+guard),
                                           FragmentsToDense, None)
            reshape_descr = [reshape_descr, OrderedDict()]
            pred_descr.args[0] = graphmanager.NodePointer('to_dense'+guard) # mutate parent

            new_model_descr = OrderedDict()
            for k,v in node_descr.items():
                if k == pred_name:
                    new_model_descr['to_dense'+guard] = reshape_descr

                new_model_descr[k] = v

            node_descr = new_model_descr

        if imposed_patch_size is not None:
            if len(imposed_patch_size)!=len(old_patch_size):
                raise ValueError("The dimensionality of the model and the "
                                 "imposed patchsize do not match.")

            if not np.array_equal(imposed_patch_size, old_patch_size) or override_mfp_to_active:
                changed = True
                ndim = len(filter_shapes[0])
                if not filter(lambda x: x[0].cls==neural.UpConv,
                              filter(lambda x: not isinstance(x, graphmanager.SplitDescriptor),
                                     node_descr.values())):

                    valid_patch_size = utils.get_cloesest_valid_patch_size(filter_shapes,
                                                     pool_shapes,
                                                     desired_patch_size=imposed_patch_size,
                                                     mfp=mfps,
                                                     ndim=ndim)

                else:
                    valid_patch_size = imposed_patch_size
                    logger.warning("Imposed patch size is not failsafe for UpConvs")
                for i, sh in enumerate(valid_patch_size):
                    old_shape = old_shape.updateshape(old_shape.spatial_axes[i], sh)

        if changed:
            input_node_descr.args[0] = tuple(old_shape.shape)
            logger.warning("Warning: the input shape of the image input node "
            "was changed during modelload. This change is NOT reflected in "
            "the printed shapes of other nodes (e.g. labels), but they "
            "have changed accordingly! This should be fixed in future, "
            "but the model can now be used for predictions nonetheless.")


    model = node_basic.model_manager.newmodel(name)
    model.restore(node_descr, **model_load_kwargs)
    if desig_descr:
        model.designate_nodes(**desig_descr)

    return model


def inject_source(node_descr, child_name, shape, tags, strides=None,
                  fov=None, dtype=theano.config.floatX, name='input', parent_no=0):
    logger.warning("'inject_source' is unsafe: it works only for nodes that have"
                   "1 child and this child must have 1 parent. You are trying"
                   "to insert a source node prior to %s" %child_name)
    from elektronn2.neuromancer.node_basic import Input
    #name = name + '_' + str(uuid.uuid4())[:5] maybe not needed...?
    input_descr = graphmanager.NodeDescriptor((shape, tags),
                                dict(strides=strides, fov=fov, dtype=dtype,
                                     name=name),
                                Input, None)
    input_descr = [input_descr, OrderedDict()]

    child_descr = node_descr[child_name][0]
    # mutate parent of child
    if hasattr(child_descr.args[0], '__len__'):
        p_name = child_descr.args[0][parent_no]
        child_descr.args[0][parent_no] = graphmanager.NodePointer(name)
    else:
        p_name = child_descr.args[parent_no]
        child_descr.args[parent_no] = graphmanager.NodePointer(name)

    logger.warning("'inject_source' replacing %s input of node %s" %(p_name, child_name))
    new_model_descr = OrderedDict()
    for k, v in node_descr.items():
        if k==child_name:
            new_model_descr[name] = input_descr

        new_model_descr[k] = v

    return new_model_descr, name

def inject_source_split(node_descr, split_name, shape, tags, strides=None,
                      fov=None, dtype=theano.config.floatX, name='input'):

    logger.warning("'inject_source_split' is unsafe")
    from elektronn2.neuromancer.node_basic import Input

    input_descr = graphmanager.NodeDescriptor((shape, tags),
                                              dict(strides=strides, fov=fov,
                                                   dtype=dtype,
                                                   name=name),
                                              Input, None)
    input_descr = [input_descr, OrderedDict()]

    split_descr = node_descr[split_name]
    split_descr.node_id = name

    new_model_descr = OrderedDict()
    for k, v in node_descr.items():
        if k==split_name:
            new_model_descr[name] = input_descr

        new_model_descr[k] = v

    return new_model_descr, name

def rebuild_rnn(model):
    """
    Rebuild an RNN by saving it to a file and reloading it from there.

    :param model: Model object.
    :return: Rebuilt Model.
    """
    img_shape = model.input_node.shape.shape
    img_tags  = model.input_node.shape.tags
    trace_it = model['scan'].n_steps
    descriptors = model.serialise()

    if 'conv_r' in descriptors:
        descriptors, input_name = inject_source_split(descriptors, 'split', img_shape,img_tags, name='img')
    else:
        descriptors, input_name = inject_source(descriptors, 'conv', img_shape, img_tags, name='img')


    if 'trace_join' in model.nodes:
        descriptors, _ = inject_source(descriptors, 'trace_join', (1,3*trace_it), 'b,f', name='trace_feedback')
    if isinstance(model['mem_hid'], node_basic.ValueNode):
        sh = model['mem_hid'].shape.shape
        descriptors, _ = inject_source(descriptors, 'gru', sh, 'b,f', name='mem_hid_feedback', parent_no=1)

    desig_descr = dict(model._desig_descr) # copy because we destroy
    desig_descr['input_node'] = input_name
    utils.picklesave((descriptors, desig_descr),
                     '/tmp/%s_tempmodel.pkl'%user_name)
    new_model = modelload('/tmp/%s_tempmodel.pkl' %user_name)
    return new_model


def rebuild_decoder(model, state_name, state_child_name):
    shape = model[state_name].shape.shape
    tags  = model[state_name].shape.tags
    descriptors = model.serialise()
    descriptors, input_name = inject_source(descriptors, state_child_name,
                                            shape, tags, name='state_input')

    desig_descr = dict(model._desig_descr) # copy because we destroy
    desig_descr['input_node'] = input_name

    # new_model_descr = OrderedDict()
    # skip = True
    # for k, v in descriptors.items():
    #     if k==input_name:
    #         skip = False
    #
    #     if not skip:
    #         new_model_descr[k] = v
    # Works only if a stuff from desig_descr is removed

    utils.picklesave((descriptors, desig_descr),
                     '/tmp/%s_tempmodel.pkl'%user_name)
    new_model = modelload('/tmp/%s_tempmodel.pkl' %user_name)
    return new_model



def rebuild_model(model, override_mfp_to_active=False,
                 imposed_patch_size=None, name=None, **model_load_kwargs):
    """
    Rebuild a Model by saving it to a file and reloading it from there.

    :param model: Model object.
    :param override_mfp_to_active: (See elektronn2.neuromancer.model.modelload()).
    :param imposed_patch_size: (See elektronn2.neuromancer.model.modelload()).
    :param name: New model name.
    :param model_load_kwargs:  Additional kwargs for restoring Model
        (see elektronn2.neuromancer.graphmanager.GraphManager.restore()).
    :return: Rebuilt Model.
    """
    model.save('/tmp/%s_tempmodel.pkl'%user_name)
    new_model = modelload('/tmp/%s_tempmodel.pkl' %user_name,
                          override_mfp_to_active=override_mfp_to_active,
                          imposed_patch_size=imposed_patch_size,
                          name=name, **model_load_kwargs)

    return new_model


def kernel_lists_from_node_descr(model_descr):
    """
    Extract the tuple (filter_shapes, pool_shapes, mfp) from a model description.

    :param model_descr: Model description OrderedDict.
    :return: Tuple (filter_shapes, pool_shapes, mfp).
    """
    filter_shapes = []
    pool_shapes   = []
    mfp           = []
    logger.warning("Warning [kernel_lists_from_model_descr]:"
                   "kernel properties are only considered for 'Conv' "
                   "nodes (excluding 'UpConv'!)")
    for name, descr in model_descr.items():
        if isinstance(descr, (list, tuple)):
          if descr[0].cls.__name__ == 'Conv':
              filter_shapes.append(descr[0].args[2])
              pool_shapes.append(descr[0].args[3])
              mfp.append(descr[0].kwargs.get('mfp', False))

    return filter_shapes, pool_shapes, mfp

def params_from_model_file(file_name):
    """
    Load parameters from a model file.

    :param file_name: File name of the pickled Model.
    :return: OrderedDict of model parameters.
    """
    logger.info("Extracting parameters from %s" %file_name)
    node_descr, desig_descr = utils.pickleload(file_name)
    params = OrderedDict()
    for name, descr in node_descr.items():
        if isinstance(descr, (list, tuple)) and len(descr[1]):
          params[name] = descr[1]

    return params


def simple_cnn(batch_size, n_ch, n_lab, desired_input, filters, nof_filters,
               activation_func, pools, mfp=False, tags=None,name=None):
    """
    Create a simple Model of a convolutional neural network.
    :param batch_size: Batch size (how many data samples are used in one
        update step).
    :param n_ch: Number of channels.
    :param n_lab: Number of distinct labels (classes).
    :param desired_input: Desired input image size. (Must be smaller than the
        size of the training images).
    :param filters: List of filter sizes in each layer.
    :param nof_filters: List of number of filters for each layer.
    :param activation_func: Activation function.
    :param pools: List of maxpooling factors for each layer.
    :param mfp: List of bools that tell if max fragment pooling should be used
        in each layer (only intended for prediction).
    :param tags: Tuple of tags for Input node (see docs of
        elektronn2.neuromancer.node_basic.Input).
    :param name: Name of the model.
    :return: Network Model.
    """  # TODO: params in docstring.

    ndim = len(desired_input)
    dimcalc = utils.cnncalculator(filters, pools, desired_input,
                         mfp=mfp, force_center=True, ndim=ndim)

    patch_size = dimcalc.input
    input_size = (batch_size, n_ch) + tuple(patch_size)

    node_basic.model_manager.newmodel(name)
    inp = node_basic.Input(input_size, tags)
    conv = list(zip(nof_filters, filters, pools, activation_func))
    for i,(n, f, p, act) in enumerate(conv):
        inp = neural.Conv(inp, n, f, p, mfp=mfp,
                         activation_func=act, )

    # last Layer
    out = neural.Conv(inp, config.n_lab, (1,)*ndim, (1,)*ndim,
                     activation_func='lin')
    if mfp:
        out = neural.FragmentsToDense(out)

    probs = loss.Softmax(out, n_class=n_lab, name='class_probabilities')

    # Label layer
    l_sh = out.shape.copy()
    l_sh.updateshape('f', 1)
    l     = node_basic.Input_like(l_sh, dtype='int16', name='labels')

    # Loss
    loss_pix  = loss.MultinoulliNLL(out, l, target_is_sparse=True)
    loss_scalar = loss.AggregateLoss(loss_pix, name='nll')

    # Model
    model = node_basic.model_manager.getmodel()
    model.designate_nodes(input_node=inp, target_node=l, loss_node=loss_scalar,
                                  prediction_node=probs,
                                  prediction_ext=[loss_scalar, loss_scalar, probs])
    return model
