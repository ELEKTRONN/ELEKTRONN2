# -*- coding: utf-8 -*-
# ELEKTRONN2 Toolkit
# Copyright (c) 2015 Marius Killinger
# All rights reserved

from __future__ import absolute_import, division, print_function
from builtins import filter, hex, input, int, map, next, oct, pow, range, super, zip

from numpy import ndarray
import logging
from collections import OrderedDict
from functools import reduce
import theano.tensor as T

from . import node_basic

logger = logging.getLogger('elektronn2log')

__all__ = ["GraphManager"]

class NodePointer(object):
    def __init__(self, target_id):
        self.target_id = target_id

    def __repr__(self):
        return "<NodePointer> %s"%self.target_id

class ParamPointer(object):
    def __init__(self, target_id, param_name):
        self.target_id = target_id
        self.param_name = param_name

    def __repr__(self):
        return "<ParamPointer> %s of node %s" %(self.param_name, self.target_id)

class SplitDescriptor(object):
    def __init__(self, node_id, func, args, kwargs):
        self.node_id = node_id
        self.args    = args
        self.kwargs  = kwargs
        self.func    = func

    def restore(self, gm):
        node = gm.nodes[self.node_id]
        return self.func(node, *self.args, **self.kwargs)



class NodeDescriptor(object):
    @staticmethod
    def _is_node_iterable(arg):
        if hasattr(arg, '__iter__') and not isinstance(arg, T.Variable):
            return reduce(lambda x,y: x*isinstance(y, node_basic.Node), arg, True)
        else:
            return False

    @staticmethod
    def _is_node_pointer_iterable(arg):
        if hasattr(arg, '__iter__'):
            return reduce(lambda x,y: x*isinstance(y, NodePointer), arg, True)
        else:
            return False

    def __init__(self, args, kwargs, cls, gm):
        new_args = []
        for arg in args:
            if isinstance(arg, node_basic.Node): # Replace node instances by id
                arg = NodePointer(arg.name)
            elif self._is_node_iterable(arg):
                new_arg = []
                for a in arg:
                   new_arg.append(NodePointer(a.name))
                arg = new_arg

            elif isinstance(arg, T.Variable): # For T.Variable search their origin
                found = False
                for name, node in gm.nodes.items(): # Search in all nodes
                    for p_name,p in node.params.items():
                        if p is arg:
                            arg = ParamPointer(name, p_name)
                            found = True
                            break
                if not found:
                    if cls==node_basic.FromTensor:
                        logger.debug("Could not find TensorVariable positional "
                        "arg to %s node. Saving/Loading might be broken" %cls)
                    else:
                         logger.warning("Could not find TensorVariable positional "
                    "arg to %s node. Saving/Loading might be broken" %cls)

                    arg = "TensorVariable not found"

            new_args.append(arg)

        new_kwargs = dict()
        for k,kwarg in kwargs.items():
            if isinstance(kwarg, node_basic.Node): # Replace node instances by id
                kwarg = NodePointer(kwarg.name)
            elif isinstance(kwarg, T.Variable): # For T.Variable search their origin
                found = False
                for name,node in gm.nodes.items():
                    for p_name,p in node.params.items():
                        if p is kwarg:
                            kwarg = ParamPointer(name, p_name)
                            found = True
                            break
                if not found:
                    logger.warning("Could not find shared param!")

            elif isinstance(kwarg, ndarray):
                continue

            new_kwargs[k] = kwarg

        self.args = new_args
        self.kwargs = new_kwargs
        self.cls = cls


    def restore(self, param_values, gm, override_mfp_to_active=False,
                make_weights_constant=False, unlock_weights=False,
                replace_bn=False):
        if self.cls==node_basic.FromTensor:
            return None

        args = self.args
        kwargs = self.kwargs

        new_args = []
        for arg in args:
            if isinstance(arg, NodePointer): # Replace nodepointers by nodes of graph_manager
                arg = gm.nodes[arg.target_id]
            elif self._is_node_pointer_iterable(arg):
                new_arg = []
                for a in arg:
                    new_arg.append(gm.nodes[a.target_id])
                arg = new_arg
            elif isinstance(arg, ParamPointer): # lookup parameter
                arg = gm.nodes[arg.target_id].params[arg.param_name]

            new_args.append(arg)

        new_kwargs = dict()
        for k,kwarg in kwargs.items():
            if isinstance(kwarg, NodePointer): # Replace nodepointers by nodes of graph_manager
                kwarg = gm.nodes[kwarg.target_id]
            elif isinstance(kwarg,ParamPointer): # For T.Variable search their origin
                 kwarg = gm.nodes[kwarg.target_id].params[kwarg.param_name]

            new_kwargs[k] = kwarg

        if make_weights_constant: # This relies on the rule that the keyword name for a
        # param of the __init__ of a node is equal to the name that param has in
        # node.params dict (param_values is create from that)!
            for name, param in param_values.items():
                if name in ['w', 'b', 'gamma', 'mean', 'std']:
                    new_kwargs[name] = [param, 'const']

        elif unlock_weights:
            if make_weights_constant:
                raise ValueError("make_weights_constant and unlock_weights"
                                 "cannot both be True")
            for name, param in param_values.items():
                if name in ['w', 'b', 'gamma', 'mean', 'std']:
                    new_kwargs[name] = param


        if replace_bn:
            if new_kwargs.get('batch_normalisation', None):
                if replace_bn=='remove':
                    new_kwargs['batch_normalisation'] = False
                elif replace_bn=='predict':
                    new_kwargs['batch_normalisation'] = 'predict'
                    #new_kwargs['mean'] = param_values['mean'] # setting is done already below
                    #new_kwargs['std']  = param_values['std']
                elif replace_bn=='const':
                    new_kwargs['batch_normalisation'] = 'predict'
                    new_kwargs['mean'] = [param_values['mean'], 'const']
                    new_kwargs['std']  = [param_values['std'], 'const']
                else:
                    raise ValueError("replace_bn can be 'remove', 'predict' or 'const'.")

        if override_mfp_to_active:
            if self.cls.__name__ == 'Conv':
                new_kwargs['mfp'] = True

        node = self.cls(*new_args, **new_kwargs)
        node.set_param_values(param_values, skip_const=True)
        return node


class GraphManager(object):
    @staticmethod
    def get_lock_names(node_descriptors, count):
        names = []
        for name,descr in node_descriptors.items():
            if (not isinstance(descr, SplitDescriptor)) and \
            (isinstance(descr[0], NodeDescriptor)):
                if len(descr[1]): # param values
                    names.append(name)
                    if len(names)>=count:
                        break
        return names


    def __init__(self, name=""):
        self.name = name
        self.nodes = OrderedDict()
        self.node_descriptors = OrderedDict()

    def __repr__(self):
        return repr(list(self.nodes.keys()))

    def __getitem__(self, slice):
        if isinstance(slice, str):
            return self.nodes[slice]
        else:
            return list(self.nodes.values())[slice]

    def reset(self):
        self.nodes = OrderedDict()
        self.node_descriptors= OrderedDict()
        logger.debug("Cleared all nodes from the graph_manager. IDs starting from 0 again")

    def register_node(self, node, name, args, kwargs):
        self.node_descriptors[name] = NodeDescriptor(args, kwargs, node.__class__, self)
        self.nodes[name] = node # Add node only after descriptor has been created:
        # descriptor may check all existing nodes, but this not has not yet finished
        # initialisation, and so look-ups might fail

    def register_split(self, node, func, name, args, kwargs):
        name = node_basic.choose_name(name, self.node_descriptors.keys())
        self.node_descriptors[name] = SplitDescriptor(node.name, func, args, kwargs)
        return name

    def serialise(self):
        descriptors = OrderedDict()
        for name,descr in self.node_descriptors.items():
            if isinstance(descr, NodeDescriptor):
                param_values = self.nodes[name].get_param_values()
                descriptor = [descr, param_values]
            else:
                descriptor = descr

            descriptors[name] = descriptor

        return descriptors


    def restore(self, descriptors, override_mfp_to_active=False,
                make_weights_constant=False, replace_bn=False,
                lock_trainable_params=False, unlock_all_params=False):

        self.reset() # This clears this gm and when the nodes are restored
        # they are registered to the current active model in the model_manager
        # so make sure that the current active model is this one!
        names_to_lock = []
        if lock_trainable_params:
            assert not unlock_all_params
            if make_weights_constant:
                logger.info("'lock_trainable_params' has no effect because"
                            "'make_weights_constant' locks all anyway.")
            if isinstance(lock_trainable_params, int):
                names_to_lock = self.get_lock_names(descriptors, lock_trainable_params)
            elif isinstance(lock_trainable_params, (list, tuple)):
                names_to_lock = list(lock_trainable_params)

            logger.info("locking parameters: %s" %(names_to_lock,))
            names_to_lock = set(names_to_lock)

        for name, descr in descriptors.items():
            # logger.debug("Processing descriptor %s:\n %s\n" % (name, rec))
            if isinstance(descr, SplitDescriptor):
                descr.restore(self)
            elif isinstance(descr[0], NodeDescriptor):
                param_values = descr[1]
                descr = descr[0]
                const = make_weights_constant or name in names_to_lock
                descr.restore(param_values, self,
                              override_mfp_to_active,
                              const, unlock_all_params, replace_bn)
            else:
                raise ValueError("Unknown descriptor: %s." %(descr))


    @property
    def sources(self):
        ret = []
        for node in self.nodes.values():
            if node.is_source:
                ret.append(node)

        return ret

    @property
    def sinks(self):
        ret = []
        for node in self.nodes.values():
            if len(node.children)==0:
                ret.append(node)

        return ret

    @property
    def node_count(self):
        return len(self.nodes)

    def plot(self):
        raise NotImplementedError()


    def function_code(self):
        raise NotImplementedError()
