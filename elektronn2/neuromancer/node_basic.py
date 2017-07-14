# -*- coding: utf-8 -*-
# ELEKTRONN2 Toolkit
# Copyright (c) 2015 Marius Killinger
# All rights reserved

from __future__ import absolute_import, division, print_function
from builtins import filter, hex, input, int, map, next, oct, pow, range, super, zip


__all__ = ['Node', 'Input', 'Input_like', 'Concat', 'ApplyFunc',
           'FromTensor', 'split', 'model_manager', 'GenericInput', 'ValueNode',
           'MultMerge', 'InitialState_like']


import sys
import inspect
import logging
import re
import time
import functools
import uuid
import gc
import getpass
from collections import OrderedDict
from functools import reduce

import numpy as np
import theano
import theano.tensor as T
import tqdm

from theano import gof
from theano import printing
from future.utils import raise_with_traceback
from future.utils import with_metaclass

from .variables import VariableWeight, ConstantParam
from .. import utils
from . import graphutils

if sys.version_info.major >= 3:
    unicode = str

logger = logging.getLogger('elektronn2log')

floatX = theano.config.floatX

user_name = getpass.getuser()

###############################################################################

class ModelContainer(object):
    """
    Keeps track of the network models.

    This class describes the global model_manager, which can contain multiple models that
    were defined by a network configuration.
    Each model is represented by a Model object.
    """
    _count = 0

    def __init__(self):
        if self._count > 0:
            raise RuntimeError("There may exist only one ModelContainer ("
                               "which may contain several models)")

        self._count += 1
        self._default_set = False
        self.models = OrderedDict()
        self.last = None
        self.current = None


    def __getitem__(self, item):
        return self.models[item]

    def __repr__(self):
        return repr(list(self.models.keys()))

    def setdefault(self):
        """
        If it is not already set, a new model named "default" is created and
        registered as the default model.
        """
        if self._default_set:
            raise RuntimeError("The default model has already been set.")
        from elektronn2.neuromancer.model import Model
        self.models["default"] = Model(name="default")
        self.current = self["default"]
        self.last = self["default"]
        self._default_set = True

    def newmodel(self, name):
        """
        Create a new Model in the ModelContainer and return the new Model
        object.

        Parameters
        ----------
        name: str
            Name of the new Model. If a Model with this name exists,
            a random UUID is used instead.

        Returns
        -------
        Model
            The new Model object.
        """
        from elektronn2.neuromancer.model import Model
        if name in self.models:
            raise ValueError("Model of the same name %s exists already." %(name))
        elif name is None:
            name = str(uuid.uuid4())
        self.models[name] = Model(name=name)
        self.last = self.current
        self.current = self[name]
        return self[name]

    def getmodel(self, *args):
        """
        Return a Model and register it as the current one.

        Parameters
        ----------
        *args
            If empty, the default Model is returned. If it is a string, the
            model with this name is returned.

        Returns
        -------
        Model
            The Model object.
        """
        if len(args)==1:
            current = self[args[0]]
        elif len(args)==0:
            current =  self["default"]
        else:
            raise ValueError("Either provide name or nothing!")

        self.last = self.current
        self.current = current
        return current

    def togglemodel(self):
        """
        Swap last and current model.

        :return: current Model
        """
        self.last, self.current = self.current, self.last
        return self.current

# Create a quasi-global, unique GraphManagerContainer and bind it the node
# registration decorator. This implies a mutable state, because the
model_manager = ModelContainer()

def choose_name(proposal, names):
    """
    Choose an appropriate unique name for a Node.

    If the proposed name is not already taken, it is directly returned.
    If it is taken and it does not end with a number, a "1" is appended to it.
    If it ends with a number and is taken, the number is incremented by 1
    until the proposed name is unique.

    Parameters
    ----------
    proposal: str
        Proposed name for the Node.
    names: str
        Names that were already given to other nodes in this Model.

    Returns
    -------
    str
        Appropriate and unique name for the Node.

    Examples
    --------
    This function makes it possible to write a network config in the following style:
    >>> out = neuromancer.Conv(inp, 12,  (3,3), (2,2))
    >>> out = neuromancer.Conv(out, 36,  (3,3), (2,2))
    >>> out = neuromancer.Conv(out, 64,  (3,3), (1,1))
    , which will automatically assign the names "conv", "conv1" and "conv2" to
    the three subsequent Conv nodes, since the proposed default name for Conv
    nodes is "conv".
    """
    if proposal in names:
        numbers = re.findall('(\d+)$', proposal)
        if not len(numbers):
           proposal = proposal+str(1)

        while proposal in names:
            proposal = re.sub('(\d+)$', lambda x: str(int(x.group(0)) + 1), proposal)

    return proposal


class MetaNode(type):
    """
    Metaclass for the ``Node`` class that manages automatic registration in the
    current ``model_manager``.

    This metaclass makes sure that the global ``model_manager``, which keeps track
    of the network models, is updated every time a new ``Node`` object is created.
    """
    def __new__(mcs, name, bases, clsdict):
        if name=='Node': # Don't do anything to base class
            pass
        else:
            for attr, attribute in clsdict.items():
                if attr == "__init__":
                    attribute = mcs.init_register(attribute)
                if attr == "__new__":
                    pass

                # Inherit class docstring
                if not attribute.__doc__:
                    for mro_cls in (mro_cls for base in bases for mro_cls in
                                    base.mro()
                                    if hasattr(mro_cls, attr)):
                        doc = getattr(getattr(mro_cls, attr), '__doc__')
                        if doc:
                            attribute.__doc__ = doc+"        NOTE: docstring was inherited\n"
                            break

                clsdict[attr] = attribute

        return super(MetaNode, mcs).__new__(mcs, name, bases, clsdict)

    @staticmethod
    def init_register(old_init):
        """
        Wrap the Node class constructor with with model management routines.

        This makes sure that every Node object that is created is registered
        within the current network model.

        Parameters
        ----------
        old_init: function
            Original constructor of the Node

        Returns
        -------
        function
            New constructor that wraps the original one
        """
        @functools.wraps(old_init)
        def new_init(*args, **kwargs):
            if model_manager.current is None:
                n_pre = 0
            else:
                n_pre = len(model_manager.current.nodes)

            node = args[0]
            args = args[1:]
            if model_manager.current is None:
                model_manager.setdefault()

            default_name = ''
            arg_names, varargs, keywords, defaults = inspect.getargspec(old_init)
            try: # extract the default name from kwargs of class initialiser
                i = arg_names.index('name')
                i = i - len(arg_names) + len(defaults)
                default_name = defaults[i]
                assert isinstance(default_name, str)
            except: # If fails just use an empty default name
                pass

            # Check to register node not twice. This might happen if a node
            # calls an inherited __init__ which has already a register
            # decorator, but the actual __init__ has already done the
            # registration. It is important that the registration is done
            # prior to the call to __init__!
            if not node in model_manager.current.nodes.values():
                # Replace given name by unique identifier (if not unique)
                name = kwargs.get('name', default_name)
                name = choose_name(name, model_manager.current.node_descriptors.keys())
                kwargs['name'] = name
                model_manager.current.register_node(node, name, args, kwargs)

            old_init(node, *args, **kwargs)
            node._finalize_init()

            n_post = len(model_manager.current.nodes)
            diff = n_post - n_pre
            if diff > 1:
                raise RuntimeError("Initialisation of %s failed: instead of 1"
                                   " %i were registered to the model_manager. It is"
                                   "not permitted to create other nodes within"
                                   "the class definiton of a node. Use a factory"
                                   "function to create several nodes at once"
                                   %(args[0].__class__, diff))
            elif diff == 0:
                logger.debug("Initialisation of %s registered 0 instead of 1 nodes. "
                            "This might be the case, if this function is "
                            "called from a derived node that has already registered."
                                   % (args[0].__class__,))

        return new_init


###############################################################################

class Node(with_metaclass(MetaNode, object)):
    """
    Basic node class. All neural network nodes should inherit from ``Node``.

    Parameters
    ----------
    parent: Node or list[Node]
        The input node(s).
    name: str
        Given name of the Node, may be an empty string.
    print_repr: bool
        Whether to print the node representation upon initialisation.


    Models are built from the interplay of *Nodes* to form a (directed,
    acyclic) computational graph.

    The **ELEKTRONN2** framework can be seen as an intelligent abstraction level
    that hides the raw theano-graph and manages the involved symbolic variables.
    The overall goal is the intuitive, flexible and **easy** creation of
    complicated graphs.

    A ``Node`` has one or several inputs, called ``parent``, (unless it is
    a *source*, i.e. a node where external data is feed into the graph).
    The inputs are node objects themselves.

    Layers automatically keep track of their previous inputs,
    parameters, computational cost etc. This allows to compile the
    theano-functions without manually specifying the inputs, outputs and
    parameters.
    In the most simple case, any node, which might be part of a more
    complicated graph, can be called as a function (passing suitable
    numpy arrays):

     >>> import elektronn2.neuromancer.utils
     >>> inp = neuromancer.Input((batch_size, in_dim))
     >>> test_data = elektronn2.neuromancer.utils.as_floatX(np.random.rand(batch_size, in_dim))
     >>> out = inp(test_data)
     >>> np.allclose(out, test_data)
     True

    At the first time the theano function is compiled and cached
    for re-use in future calls.

    Several properties (with respect to the sub-graph the node depends on, or
    only from the of the node itself)
    These can also be looked up externally (e.g. required sources,
    parameter count, computational count).

    The theano variable that represents the output of a node is kept
    in the attribute ``output``. Subsequent Nodes must use this attribute of
    their inputs to perform their calculation and write the result to their own
    output (this happens in the method ``_calc_output``, which is hidden
    because it must be called only internally at initialisation).

    A divergence in the computational graph is created by passing the *parent*
    to several *children* as input:

     >>> inp = neuromancer.Input((1,10), name='Input_1')
     >>> node1 = neuromancer.ApplyFunc(inp, func1)
     >>> node2 = neuromancer.ApplyFunc(inp, func2)


    A convergence in the graph is created by passing several inputs to a node
    that performs a reduction:

     >>> out = neuromancer.Concat([node1, node2])

    Although the node "out" has two immediate inputs, it is detected that the
    required sources is only a single object:

      >>> print(out.input_nodes)
      Input_1

    Computations that result in more than a single output for a Node must be
    broken apart using divergence and individual nodes for the several outputs.
    Alternatively the function ``split`` can be used to create two
    dummy nodes of the output of a previous Node by splitting along the
    specified axis.
    Note that possible redundant computations in Nodes are most likely
    eliminated by the theano graph optimiser.


    **Instructions for subclassing:**

    Overriding ``__init__``:
        At the very first the base class' initialiser must be called, which just
        assigns the names and emtpy default values for attributes.
        Then node specific initialisations are made e.g. initialisation of shared
        parameters / weights.
        Finally the ``_finialise_init`` method of the base class is automatically
        called:
        This evokes the execution of the methods: ``_make_output``,
        ``_calc_shape`` and ``self._calc_comp_cost``. Each of those updates
        the corresponding attributes.
        NOTE: if a Node (except for the base ``Node``) is subclassed and the
        derived calls ``__init__`` of the base Node, this will also call
        ``_finialise_init`` exactly right the call to the superclass'
        ``__init__``.

    For the graph serialisation and restoration to work, the following
    conditions must additionally be met:

    * The name of of a node's trainable parameter in the parameter dict must
      be the same as the (optional) keyword used to initialise this parameter
      in ``__init__``;
      moreover, parameters must not be initialised/shared from positional
      arguments.
    * When serialising only the current state of parameters is kept, parameter
      value arrays given for initialisation are never kept.

    Depending on the purpose of the node, the latter methods and others
    (e.g. ``__repr__``) must be overridden. The default behaviour of the base
    class is: output = input, outputs shape = input shape,
    computational cost = tensor size (!) ...
    """

    def __init__(self, parent, name="", print_repr=False):
        self.parent             = parent
        self.children           = OrderedDict()
        self.name               = name
        self._features_names    = None
        # weights, Dropout rate, mixing rates, batch normalisation etc.
        self.params             = OrderedDict()
        self.computational_cost = 0

        self.is_source   = False
        self.output       = None # theano tensor variable
        self.shape        = None # The *output* shape
        self._output_func = None # theano function (compiled on demand)
        self._debug_outputs = []
        self._output_func_debug = None
        self._local_exec_time = None
        self._total_exec_time = None
        self._prof = None

        self._finalized = False
        self._print_repr = print_repr


    def __repr__(self):
        if self.name=='':
            s  = "<%s-Node>\n"%(self.__class__.__name__,)
        else:
            s  = "<%s-Node> '%s' \n"%(self.__class__.__name__, self.name)
        s += '  '
        if self.param_count>0:
            s += "#Params={0:,d} ".format(self.param_count)
        if self.computational_cost>0:
            s += "Comp.Cost=%s, " %(utils.pretty_string_ops(self.computational_cost))
        s += "Out:%s" %(str(self.shape))
        if len(self.input_nodes)>1:
            sources = [node.name for node in self.input_nodes]
            s+= "\n  Order of sources=%s, "%(str(sources))
        return s


    def __call__(self, *args):
        """
        Compute output of node. If called for the first time the theano-function
        must be compiled which might take a while.

        Parameters
        ----------

        *args: numpy arrays
            Graph input

        Returns
        -------
        np.ndarray
            Node output.
        """
        try:
            return self._output_func(*args)
        except TypeError as e:
            tb = sys.exc_info()[2]
            if len(args)!=len(self.input_nodes):
                add_info = "\n %i inputs required, %i were given." %(len(self.input_nodes), len(args))
            else:
                try:
                    add_info = '\nShapes (required - given):\n'
                    for ni, ar in zip(self.input_nodes, args):
                        add_info += " %s %s\t%s %s\n" %(tuple(ni.shape.shape), ni.dtype, ar.shape, ar.dtype)
                except:
                    add_info = ""

            raise_with_traceback(type(e)(str(e) + add_info), tb)

    def get_debug_outputs(self, *args):
        if self._output_func_debug:
            return self._output_func_debug(*args)
        else:
            logger.warning("Layer '%s' has no debug outputs defined." %(self.name,))

    def _finalize_init(self):
        """
        Calculates output, computational cost and shape, prints node repr (if
        not turned off)
        """
        if self._finalized:
            logger.debug("Attempt to call _finalize_init twice on %s. This "
            "might be the case, if this function is called from a derived node "
            "and the base has already called it. This call is IGNORED." %self.name)
            return

        # Don't change the order of this!
        self._make_output()
        self._calc_shape()
        self._calc_comp_cost()
        if isinstance(self.parent, (list, tuple)):
            for p in self.parent:
                p._register_child(self)
                # logger.debug('{} {} {}'.format(id(p), p.name, p.children.keys())
        else:
            if self.parent is not None:
                self.parent._register_child(self)

        self._output_func = graphutils.make_func(self.input_tensors,
                                                 self.output, name=self.name)

        if self._debug_outputs:
            self._output_func_debug = graphutils.make_func(self.input_tensors,
                                                     self._debug_outputs, name=self.name+"_dbg")

        if self._print_repr:
            logger.info("-" * 87)
            logger.info(self)
        else:
            logger.debug("-" * 87)
            logger.debug(self)

        self._finalized = True

    def _register_child(self, child):
        """
        Add child to dict of children
        """
        self.children[child.name] = child
        #print("Adding Child %s to node %s"%(child.name, self.name))


    def _make_output(self):
        """
        Computation of output variable (symbolic)
        """
        self.output = self.parent.output


    def _calc_shape(self):
        """
        Computation of output shape and tags
        """
        if isinstance(self.parent, (list, tuple)):
            self.shape = self.parent[0].shape.copy()
        else:
            self.shape = self.parent.shape.copy()


    def _calc_comp_cost(self):
        """
        Computation of computational cost
        """
        if isinstance(self.parent, (list, tuple)):
            self.computational_cost = self.parent[0].shape.stripnone_prod
        else:
            self.computational_cost = self.parent.shape.stripnone_prod


    def get_param_values(self, skip_const=False):
        """
        Returns a dict that maps the values of the params.
        (such that they can be saved to disk)

        Parameters
        ----------
        skip_const: bool
            whether to exclude constant parameters.

        Returns
        -------
        dict
            Dict that maps the values of the params.
        """
        p_dict = OrderedDict()
        for k,v in self.params.items():
            if v.constant and skip_const:
                logger.debug("Skipping %s because it is constant" % (k,))
            else:
                p_dict[k] = v.get_value()

        return p_dict

    def set_param_values(self, value_dict, skip_const=False):
        """
        Sets new values for non constant parameters.

        Parameters
        ----------
        value_dict: dict
            A dict that maps values by parameter name.
        skip_const: bool
            if dict also maps values for constants, these can be skipped,
            otherwise an exception is raised.
        """
        for k,v in value_dict.items():
            if k not in self.params:
                if k in ['gamma', 'std', 'mean']:
                    logger.info("Skipping set_value of BN parameter %s" %(k,))
                    continue
                raise KeyError("Layer has no parameter %s" %(k,))

            if self.params[k].constant:
                if skip_const:
                    logger.debug("Skipping %s because it is constant" % (k,))
                    continue
                else:
                    raise ValueError("Cannot set value of constant "
                                     "parameter %s in node %s" %(k, self.name))

            self.params[k].set_value(v)

    @property
    def all_parents(self):
        """
        List all nodes that are involved in the computation of the output
        of this node (incl. ``self``). The list contains no duplicates.
        The return is a dict, the keys of which are the layers, the values are
        just all ``True``
        """
        parents = OrderedDict()

        if self.parent is None: # For an input/source node
            pass
        elif isinstance(self.parent, (tuple, list)):
            for node in self.parent:
                parents.update(node.all_parents)
        else:
            parents.update(self.parent.all_parents)

        parents[self.name] = self # insert this node itself
        return parents


    @property
    def input_nodes(self):
        """
        Contains the all parent nodes that are sources, i.e. inputs that are
        required to compute the result of this node.
        """
        inputs = []
        for node in self.all_parents.values(): # this does not contain duplicates
            if node.is_source:
                inputs.append(node)
        return inputs

    @property
    def input_tensors(self):
        """
        The same as ``input_nodes`` but contains the theano tensor variables
        instead of the node objects. May be used as input to compile theano
        functions.
        """
        return [s.output for s in self.input_nodes]

    @property
    def all_params(self):
        """
        Dict of the all parameters of all parent nodes. They are theano variable
        """
        p_dict = OrderedDict()
        for node in self.all_parents.values():
            for k,v in node.params.items():
                p_dict[str(node.name)+'_'+k] = v
        return p_dict


    @property
    def all_trainable_params(self):
        """
        Dict of the trainable parameters (weights) of all parent nodes. They
        are theano shared variables.
        """
        p_dic = OrderedDict()
        for node in self.all_parents.values():
            for k,v in node.params.items():
                if v.apply_train:
                    p_dic[str(node.name)+'_'+k] = v
        return p_dic


    @property
    def all_nontrainable_params(self):
        """
        Dict of the trainable parameters (weights) of all parent nodes. They
        are theano shared variables.
        """
        p_dic = OrderedDict()
        for node in self.all_parents.values():
            for k,v in node.params.items():
                if not v.apply_train:
                    p_dic[str(node.name)+'_'+k] = v
        return p_dic

    @property
    def all_extra_updates(self):
        """
        List of the parameters updates of all parent nodes. They
        are tuples.
        """
        updates = []
        for node in self.all_parents.values():
            for param in node.params.values():
                if param.updates:
                    updates.append(param.updates)
        return updates

    @property
    def param_count(self):
        """
        Count of trainable parameters in this node
        """
        if self.params:
            param_count = reduce(lambda x,y: x+y.get_value().size,
                                 [x for x in self.params.values() if x.apply_train],
                                 0)
        else:
            param_count = 0

        return param_count

    @property
    def all_params_count(self):
        """
        Count of all trainable parameters in the entire sub-graph used to
        compute the output of this node
        """
        if self.all_trainable_params:
            param_count = reduce(lambda x,y: x+y.get_value().size,
                                 self.all_trainable_params.values(), 0)
        else:
            param_count = 0

        return param_count

    @property
    def all_computational_cost(self):
        return reduce(lambda x,y: x+y.computational_cost,
                      self.all_parents.values(), 0)

    @property
    def all_children(self):
        children = OrderedDict()
        for child in self.children.values():
            children.update(child.all_children)

        return children

    @property
    def last_exec_time(self):
        """
        Last function execution time in seconds.
        """
        return self._output_func.last_exec_time

    @property
    def local_exec_time(self):
        if self._local_exec_time is None:
            self.measure_exectime(print_info=False, local=True)

        return self._local_exec_time

    @property
    def total_exec_time(self):
        if self._total_exec_time is None:
            self.measure_exectime(print_info=False, local=True)

        return self._total_exec_time


    @property
    def feature_names(self):
        if self._features_names:
            return self._features_names
        else:
            return None

    @feature_names.setter
    def feature_names(self, value):
        if 'f' not in self.shape.tags:
            raise ValueError("Shape has no feature tag")
        if len(value)!= self.shape['f']:
            raise ValueError("Shape has %i features, but %i names were"
                             "given" %(self.shape['f'], len(value)))

        self._features_names = tuple(value)


    def _predict_densetile(self, raw_img, out_arr):
        """
        Parameters
        ----------
        raw_img: np.ndarray
          raw image (ch, x, y) or (ch, z, y, x)to be predicted
          The shape must be cnn.patch_size + cnn.output_strides - 1 (elwise)
        out_arr: np.ndarray
          The shape is cnn.patch_size + cnn.mfp_strides - floor(cnn.offset) - 1 (elwise)

        Returns
        -------
        np.ndarray
          Prediction (n_lab, z, y, x).
          The shape is cnn.patch_size + cnn.mfp_strides - floor(cnn.offset) - 1 (elwise)

        """
        strides = self.shape.strides
        patch_size = self.input_nodes[0].shape.spatial_shape

        if np.all(np.equal(strides,1)):
            if self.shape.ndim==2:
                out_arr[:,0] = self(raw_img[None])[0] # (ch,x,y)
            else:
                out_arr[:] = self(raw_img[None])[0] # (ch,z,y,x)

        else:
            for x_off in range(strides[-2]):
                for y_off in range(strides[-1]):
                    if self.shape.ndim==2:
                        cut_img = raw_img[None, :,
                                  x_off: x_off+patch_size[0],
                                  y_off: y_off+patch_size[1]]

                        # insert prob(ch, x, y) into out_arr(ch,z,y,x)
                        out_arr[:,
                        0,
                        x_off::strides[0],
                        y_off::strides[1]] = self(cut_img)[0]

                    elif self.shape.ndim==3:
                        for z_off in range(strides[0]):
                            cut_img = raw_img[None,
                                      :,
                                      z_off: z_off+patch_size[0],
                                      x_off: x_off+patch_size[1],
                                      y_off: y_off+patch_size[2]]

                            out_arr[:,
                            z_off::strides[0],
                            x_off::strides[1],
                            y_off::strides[2]] = self(cut_img)[0]

        return out_arr

    def predict_dense(self, raw_img, as_uint8=False, pad_raw=False):
        """
        Core function that performs the inference

        Parameters
        ----------
        raw_img : np.ndarray
            raw data in the format (ch, (z,) y, x)
        as_uint8: Bool
            Return class probabilites as uint8 image (scaled between 0 and 255!)
        pad_raw: Bool
            Whether to apply padding (by mirroring) to the raw input image
            in order to get predictions on the full image domain.

        Returns
        -------
        np.ndarray
            Predictions.
        """

        # TODO: Fix "negative dimensions are not allowed" errors with raw_img.shape < patch size

        # determine normalisation depending on int or float type
        if self.shape.ndim<2:
            logger.error("'predict_dense' works only for nodes with 2 or 3 "
                         "spatial axes, this node has shape %s."%(self.shape))
            return None

        if self.shape.ndim==2 and raw_img.ndim==3 and \
                        raw_img.shape[0]!=self.input_nodes[0].shape['f']:
            logger.error("If 3d input is given to 2d CNNs, the first axis must"
                         "contain the features/channel and have size %i. Note"
                         "that also 4d input can be given to 2d CNNs in the "
                         "axis order (ch, z, y, x), where for each z-slice a"
                         "2d prediction image is created"
                         %self.input_nodes[0].shape['f'])
            return None

        offset = self.shape.offsets
        if np.any(np.less(offset, 0)):
            raise ValueError("Cannot predict dense because the CNN contains "
                             "UpConvs which cause unknown FOVs. If you use "
                             "UpConvs you should not need predict dense anyway!")

        if raw_img.dtype in [np.int, np.int8, np.int16, np.int32, np.uint32, np.uint,
                             np.uint8, np.uint16, np.uint32, np.uint32]:
            m = 255
        else:
            m  = 1

        raw_img = graphutils.as_floatX(raw_img) / m

        time_start = time.time()
        strip_z = False
        if raw_img.ndim==3:
            strip_z = True
            raw_img = raw_img[:,None] # add singleton z-channel  # TODO: Correct order?

        n_lab      = self.shape['f']
        cnn_out_sh = self.shape.spatial_shape
        ps         = self.input_nodes[0].shape.spatial_shape
        strides    = self.shape.strides


        if self.shape.ndim==2:
            cnn_out_sh = np.concatenate([[1,], cnn_out_sh])
            ps         = np.concatenate([[1,], ps])
            strides    = np.concatenate([[1,], strides])
            offset     = np.concatenate([[0,], offset])

        if pad_raw:
            raw_img = np.pad(raw_img,
                             [(0,0),
                              (offset[0],offset[0]),
                              (offset[1],offset[1]),
                              (offset[2],offset[2])],
                             mode='symmetric')

        raw_sh     = raw_img.shape[1:] # only spatial, not channels
        tile_sh = np.add(ps, strides) - 1
        prob_sh = np.multiply(cnn_out_sh, strides)
        prob_arr = np.zeros(np.concatenate([[n_lab,], prob_sh]), dtype=floatX)

        pred_sh = np.array([raw_sh[0]-2*offset[0], raw_sh[1]-2*offset[1], raw_sh[2]-2*offset[2] ])
        if as_uint8:
            predictions = np.zeros(np.concatenate(([n_lab,], pred_sh)), dtype=np.uint8)
        else:
            predictions = np.zeros(np.concatenate(([n_lab,], pred_sh)), dtype=floatX)


        # Calculate number of tiles (in 3d: blocks) that need to be performed
        z_tiles = int(np.ceil(float(pred_sh[0])/prob_sh[0]))
        x_tiles = int(np.ceil(float(pred_sh[1])/prob_sh[1]))
        y_tiles = int(np.ceil(float(pred_sh[2])/prob_sh[2]))
        total_nb_tiles = np.product([x_tiles, y_tiles, z_tiles])
        if self._output_func.func is None:
            self._output_func.compile()

        logger.info("Predicting img %s in %i Blocks: (%i, %i, %i)" \
                    %(raw_img.shape, total_nb_tiles, z_tiles, x_tiles, y_tiles))
        self() # This compiles the function
        pbar = tqdm.tqdm(total=total_nb_tiles, ncols=80, leave=False)
        for z_t in range(z_tiles):
            for x_t in range(x_tiles):
                for y_t in range(y_tiles):
                    # For every z_tile a slice of thickness cnn_out_sh[2] is
                    # collected and then collectively written to the output_data
                    raw_tile = raw_img[:,
                               z_t*prob_sh[0]:z_t*prob_sh[0]+tile_sh[0],
                               x_t*prob_sh[1]:x_t*prob_sh[1]+tile_sh[1],
                               y_t*prob_sh[2]:y_t*prob_sh[2]+tile_sh[2]]

                    this_is_end_tile = False if np.all(np.equal(raw_tile.shape[1:],  tile_sh)) else True

                    if this_is_end_tile: # requires 0-padding
                        right_pad = np.subtract(tile_sh, raw_tile.shape[1:]) # (ch,z,x,y)
                        right_pad = np.concatenate(([0,], right_pad)) # for channel dimension
                        left_pad  = np.zeros(raw_tile.ndim, dtype=np.int)
                        pad_with  = list(zip(left_pad,right_pad))
                        raw_tile  = np.pad(raw_tile, pad_with, mode='constant')

                    if self.shape.ndim==2:
                        # slice from raw_tile(ch,z,x,y) --> (ch,x,y)
                        prob_arr = self._predict_densetile(raw_tile[:,0], prob_arr) # returns (ch,z=1,x,y)
                        prob = prob_arr
                    else:
                        prob_arr = self._predict_densetile(raw_tile, prob_arr)
                        prob = prob_arr

                    if this_is_end_tile: # cut away padded range
                        prob = prob[:, :prob_sh[0]-right_pad[1], :prob_sh[1]-right_pad[2], :prob_sh[2]-right_pad[3]]

                    if as_uint8:
                        prob *= 255

                    predictions[:,
                    z_t*prob_sh[0]:(z_t+1)*prob_sh[0],
                    x_t*prob_sh[1]:(x_t+1)*prob_sh[1],
                    y_t*prob_sh[2]:(y_t+1)*prob_sh[2]] = prob

                    pbar.update()

        pbar.close()
        dtime = time.time() - time_start
        speed = np.product(predictions.shape[1:]) * 1.0/1000000/dtime
        dtime = utils.pretty_string_time(dtime)
        logger.info(" Inference speed: %.3f MB or MPix /s, time %s" \
                    %(speed, dtime))

        if strip_z:
            predictions = predictions[:,0,:,:]

        return predictions

    def test_run(self, on_shape_mismatch='warn', debug_outputs=False):
        """
        Test execution of this Node with random (but correctly shaped) data.

        Parameters
        ----------
        on_shape_mismatch: str
            If this is "warn", a warning is emitted if there is a mismatch
            between expected and calculated output shapes.

        Returns
        -------
            Debug output of the Theano function.
        """
        inp = self.input_nodes
        test_vals = []
        for i in inp:
            sh = [1 if s is None else s for s in i.shape.shape]
            test_vals.append(np.random.rand(*sh).astype(i.dtype))

        if debug_outputs:
            return self.get_debug_outputs(*test_vals)

        self() # compile before timing
        y = self(*test_vals)
        shape_match = tuple(y.shape) == tuple(self.shape.shape)
        if not shape_match:
            if np.prod(y.shape) == self.shape.stripnone_prod:
                shape_match = True

        if shape_match:
            sh_str = "Shapes agree"
        else:
            sh_str = 'Shapes do not agree (Outshape should be %s)' % self.shape.shape

        dtime = self._output_func.last_exec_time
        speed = self.shape.spatial_size * 1.0/1000000/dtime
        #dtime = utils.pretty_string_time(dtime)

        print("Node '%s' compiled and computation tested\n"
              "Outshape: %s\n"
              "Runtime: %g s\n"
              "Speed: %.3f MB or MPix /s\n"
              "%s"%(self.name, y.shape, dtime, speed, sh_str))

        if not shape_match:
            if on_shape_mismatch=='warn':
                logger.warning("Shape of computed output and shape calculated "
                               "by layer definition not match")
        return y

    def plot_theano_graph(self, outfile=None, compiled=True, **kwargs):
        """
        Plot the execution graph of this Node's Theano function to a file.

        If "outfile" is not specified, the plot is saved in "/tmp/<user>_<name>.png"

        Parameters
        ----------
        outfile: str or None
            File name for saving the plot.
        compiled: bool
            If True, the function is compiled before plotting.
        kwargs
            kwargs (plotting options) that get directly passed to
            theano.printing.pydotprint().
        """
        if outfile is None:
            outfile ='/tmp/{}_{}.png'.format(user_name, self.name)
        kwargs['outfile'] = outfile
        if 'var_with_name_simple' not in kwargs:
            kwargs['var_with_name_simple'] = True
        if compiled:
            self()
            printing.pydotprint(self._output_func.func, **kwargs)
        else:
            printing.pydotprint(self.output, **kwargs)

    def measure_exectime(self, n_samples=5, n_warmup=4, print_info=True, local=True, nonegative=True):
        """
        Measure how much time the node needs for its calculation (in milliseconds).

        Parameters
        ----------
        n_samples: int
            Number of independent measurements of which the median is taken.
        n_warmup: int
            Number of warm-up runs before each measurement (not taken into account for median calculation).
        print_info: bool
            If True, print detailed info about measurements while running.
        local: bool
            Only compute exec time for this node by subtracting its parents' times.
        nonegative: bool
            Do not return exec times smaller than zero.

        Returns
        -------
        np.float
            median of execution time measurements.
        """
        # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
        inp = self.input_nodes
        if not inp:
            logger.info('Node {} has no inputs -> skipping node, giving it zero execution time.'.format(self.name))
            return 0.0
        self._prof = theano.ProfileStats(atexit_print=False)
        self._output_func.func = None
        self._output_func.compile(profile=self._prof)
        test_vals = [np.random.rand(*i.shape.shape).astype(i.dtype) for i in inp]
        for _ in range(n_warmup):  # TODO: Warmup doesn't work this way with the new theano profiling. Warmups are currently fully counted.
            self(*test_vals)

        def noprofile_op(op, unwanted_ops=None):
            if unwanted_ops is None:
                unwanted_ops = ['HostFromGpu', 'GpuFromHost']
            return str(op) in unwanted_ops

        def measure_once():
            self(*test_vals)
            # exec_time = self.last_exec_time  # TODO: Is n_f already taken into account here?
            # exec_time = self._output_func.func.profile.fct_call_time / self._output_func.func.profile.fct_callcount
            exec_time = sum(t for op, t in self._prof.op_time().items()
                            if not noprofile_op(op)
                            ) / self._prof.fct_callcount
            return exec_time * 1000  # in ms

        subgraph_exec_times = np.array([measure_once() for _ in range(n_samples)])
        sort_ix = np.argsort(subgraph_exec_times)
        # TODO: If the profile.op_times() / profile.fct_callcount() approach stays, mean or median don't make sense here
        median_exec_time = subgraph_exec_times[sort_ix[-3:]].mean() #  np.sort(subgraph_exec_times)

        self._total_exec_time = median_exec_time

        if local:
            if self.is_source:
                parents = []
            elif isinstance(self.parent, (list,tuple)):
                parents = self.parent
            else:
                parents = [self.parent,]

            parents_exec_time = sum([par.total_exec_time for par in parents if par.input_nodes])

            # parents = [par for par in self.all_parents.values() if par is not self]
            # parents_exec_time = sum([par.local_exec_time for par in parents if par.input_nodes])

            median_exec_time -= parents_exec_time
            self._local_exec_time = median_exec_time

        if nonegative and median_exec_time < 0:
            logger.warning('{}: Local execution time smaller than 0. "Correcting" to 0.'.format(self.name))
            median_exec_time = 0

        # Explicitly free memory since GPU memory tends to overflow here after multiple calls
        self._output_func.func = None
        for _ in range(3):  # Use this hack until https://github.com/Theano/Theano/issues/4275 is resolved.
            gc.collect()

        if print_info:
            logger.info('{0} samples in ms:\n{1}\n{0}: median execution time: {2} ms\n'
                        .format(self.name, subgraph_exec_times, median_exec_time))

        return median_exec_time

###############################################################################
###############################################################################
###############################################################################

class Input(Node):
    """
    Input Node

    Parameters
    ----------
    shape: list/tuple of int
        shape of input array, unspecified shapes are ``None``
    tags: list/tuple of strings or comma-separated string
        tags indicate which purpose the dimensions of the tensor serve. They are
        sometimes used to decide about reshapes. The maximal tensor has tags:
        "r, b, f, z, y, x, s" which denote:
        * r: perform recurrence along this axis
        * b: batch size
        * f: features, filters, channels
        * z: convolution no. 3 (slower than 1,2)
        * y: convolution no. 1
        * x: convolution no. 2
        * s: samples of the same instance (over which expectations are calculated)
        Unused axes are to be removed from this list, but ``b`` and ``f`` must
        always remain.
        To avoid bad memory layout, the order must not be changed.
        For less than 3 convolutions conv1,conv2 are preferred for performance reasons.
        Note that CNNs can mix nodes with 2d and 3d convolutions as 2d is a special
        case of 3d with filter size 1 on the respective axis. In this case conv3
        should be used for the axis with smallest filter size.
    strides
    fov
    dtype: str
        corresponding to numpy dtype (e.g., 'int64').
        Default is floatX from theano config
    hardcoded_shape
    name: str
        Node name.
    print_repr: bool
        Whether to print the node representation upon initialisation.
    """

    def __init__(self, shape, tags, strides=None, fov=None, dtype=theano.config.floatX,
                 hardcoded_shape=False, name='input', print_repr=True):
        super(Input, self).__init__(None, name, print_repr)

        self.is_source = True
        self._shape = graphutils.TaggedShape(shape, tags, strides, fov=fov)
        if type(dtype) not in [str, unicode]:
            raise ValueError("dtype must be a string.")
        self.dtype = dtype
        self.hardcoded_shape = hardcoded_shape
        self._local_exec_time = 0


    def _make_output(self):
        broadcast = (False,)*len(self._shape)
        self.output = T.TensorType(self.dtype, broadcast)(name=self.name)
        if self.hardcoded_shape and not None in self._shape.shape:
            self.output = self.output.reshape(self._shape.shape)

    def _calc_shape(self):
        self.shape = self._shape

    def _calc_comp_cost(self):
        self.computational_cost = 0


def Input_like(ref, dtype=None, name='input',
               print_repr=True, override_f=False, hardcoded_shape=False):
    if isinstance(ref, Node):
        shape = list(ref.shape.shape)
        tags  = ref.shape.tags
        strides = ref.shape.strides
        fov = ref.shape.fov
        if override_f:
            shape = list(shape)
            shape[ref.shape.tag2index('f')] = override_f

        if dtype is None:
            dtype = ref.output.dtype

    elif isinstance(ref, graphutils.TaggedShape):
        shape = ref.shape
        tags  = ref.tags
        strides = ref.strides
        fov = ref.fov
        assert dtype is not None
        assert override_f is None
    else:
        raise ValueError("ref must be Node or TaggedShape.")
    return Input(shape, tags, strides, fov=fov, dtype=dtype,
                 name=name, print_repr=print_repr,
                 hardcoded_shape=hardcoded_shape)


class GenericInput(Node):
    """
    Input Node for arbitrary oject.

    Parameters
    ----------
    name: str
        Node name.
    print_repr: bool
        Whether to print the node representation upon initialisation.
    """

    def __init__(self, name='generic_input',
                 print_repr=True):
        super(GenericInput, self).__init__(None, name, print_repr)
        self.is_source = True
        self._shape = None

    def _make_output(self):
        self.output = gof.type.Generic()(name=self.name)

    def _calc_shape(self):
        self.shape = self._shape


    def _calc_comp_cost(self):
        self.computational_cost = 0


class FromTensor(Node):
    """
    Dummy Node to be used in the split-function.

    Parameters
    ----------
    tensor: T.Tensor
    tensor_shape
    tensor_parent: T.Tensor
    name: str
        Node name.
    print_repr: bool
        Whether to print the node representation upon initialisation.
    """

    def __init__(self, tensor, tensor_shape, tensor_parent, name='from_tensor',
                 print_repr=False):
        super(FromTensor, self).__init__(tensor_parent, name, print_repr)

        self._shape = tensor_shape
        self._output = tensor

    def _make_output(self):
        self.output = self._output

    def _calc_shape(self):
        self.shape = self._shape

    def _calc_comp_cost(self):
        self.computational_cost = 0


def split(node, axis='f', index=None, n_out=None, strip_singleton_dims=False, name='split'):
    args = ()
    kwargs = dict(axis=axis, index=index, n_out=n_out,
                  strip_singleton_dims=strip_singleton_dims, name=name)
    if isinstance(name, (list, tuple)):
        name_list = list(name)
        name = 'split'
    else:
        name_list = None
    model_manager.current.register_split(node, split, name, args, kwargs)

    sh = node.shape
    try:
        sh[axis]
    except:
        raise ValueError("%s has no index %s."%(sh, axis))

    if index is not None:
        index = utils.as_list(index)

    if (index is not None) and (np.max(index) >= sh[axis]):
        if sh[axis] is not None:
            raise ValueError("Index %s larger than size of axis %s (%s)."%(sh, axis, sh[axis]))

    if (n_out is not None) and (sh[axis]%n_out!=0):
        raise ValueError("Cannot split axis of size %i into %i parts."%(sh[axis], n_out))

    if isinstance(axis, (str, unicode)):
        axis = sh.tag2index(axis)

    if index is None:
        splits = [sh[axis]//n_out * i for i in range(n_out+1)]
    else:
        splits = [0,] + index + [sh[axis],]

    tensor = node.output
    out_nodes = []
    for i in range(len(splits)-1):
        if splits[i+1] is None:
            length = None
        else:
            length = splits[i + 1] - splits[i]

        sl = [slice(None),] * tensor.ndim
        if strip_singleton_dims and length==1:
            sl[axis] = splits[i]
        else:
            sl[axis] = slice(splits[i],splits[i+1])
        sub_tensor = tensor[tuple(sl)]

        if strip_singleton_dims and length==1:
            sub_sh = sh.delaxis(axis)
        else:
            sub_sh = sh.updateshape(axis, length)

        if name_list is not None:
            assert len(name_list)==len(splits)-1
            name_i = name_list[i]
        else:
            name_i = name
        out_node = FromTensor(sub_tensor, sub_sh, node, name=name_i)
        out_nodes.append(out_node)

    return tuple(out_nodes)


###############################################################################

class Concat(Node):
    """
    Node to concatenate the inputs. The inputs must have the same shape,
    except in the dimension corresponding to ``axis``. This is not checked as
    shapes might be unspecified prior to compilation!

    Parameters
    ----------
    parent_nodes: list of Node
        Inputs to be concatenated.
    axis: int
        Join axis.
    name: str
        Node name.
    print_repr: bool
        Whether to print the node representation upon initialisation.
    """

    def __init__(self, parent_nodes, axis='f', name="concat", print_repr=True):
        super(Concat, self).__init__(parent_nodes, name, print_repr)

        if not isinstance(parent_nodes, (tuple, list)):
            raise ValueError("Can only join list/tuple of nodes")

        if isinstance(axis, str):
            self.axis = parent_nodes[0].shape.tag2index(axis)
        else:
            self.axis = axis


    def _make_output(self):
        # It is assumed that all other dimensions are matching
        inputs = [inp.output for inp in self.parent]
        self.output = T.concatenate(inputs, axis=self.axis)

    def _calc_shape(self):
        joint_axis_size = reduce(lambda x,y: x+y.shape[self.axis],
                                 self.parent, 0)
        # assuming all other dimensions are equal
        sh = self.parent[0].shape.updateshape(self.axis, joint_axis_size)
        self.shape = sh

    def _calc_comp_cost(self):
        self.computational_cost = 0

    ###############################################################################

class MultMerge(Node):
    """
    Node to concatenate the inputs. The inputs must have the same shape,
    except in the dimension corresponding to ``axis``. This is not checked as
    shapes might be unspecified prior to compilation!

    Parameters
    ----------
    n1: Node
        First input node.
    n2: Node
        Second input node.
    name: str
        Node name.
    print_repr: bool
        Whether to print the node representation upon initialisation.
    """

    def __init__(self, n1, n2, name="multmerge",
                 print_repr=True):
        super(MultMerge, self).__init__((n1, n2), name, print_repr)
        assert n1.shape.shape==n2.shape.shape

    def _make_output(self):
        self.output = T.mul(self.parent[0].output, self.parent[1].output)

    def _calc_shape(self):
        self.shape = self.parent[0].shape.copy()

    def _calc_comp_cost(self):
        self.computational_cost = self.parent[0].shape.stripnone_prod

###############################################################################

class ApplyFunc(Node):
    """
    Apply function to the input. If the function changes the output shape,
    this node should not be used.

    Parameters
    ----------
    parent: Node
        Input (single).
    functor: function
        Function that acts on theano variables (e.g. ``theano.tensor.tanh``).
    args: tuple
        Arguments passed to ``functor`` **after** the input.
    kwargs: dict
        kwargs for ``functor``.
    """

    def __init__(self, parent, functor, args=(), kwargs={}, name="apply",
                 print_repr=False):
        super(ApplyFunc, self).__init__(parent, name, print_repr)

        self._functor = functor
        self._args = args
        self._kwargs = kwargs


    def _make_output(self):
        self.output = self._functor(self.parent.output,
                                    *self._args, **self._kwargs)


class ValueNode(Node):
    """
    (Optionally) trainable Value Node

    Parameters
    ----------
    shape: list/tuple of int
        shape of input array, unspecified shapes are ``None``
    tags: list/tuple of strings or comma-separated string
        tags indicate which purpose the dimensions of the tensor serve. They are
        sometimes used to decide about reshapes. The maximal tensor has tags:
        "r, b, f, z, y, x, s" which denote:
            * r: perform recurrence along this axis
            * b: batch size
            * f: features, filters, channels
            * z: convolution no. 3 (slower than 1,2)
            * y: convolution no. 1
            * x: convolution no. 2
            * s: samples of the same instance (over which expectations are calculated)
        Unused axes are to be removed from this list, but ``b`` and ``f`` must
        always remain.
        To avoid bad memory layout, the order must not be changed.
        For less than 3 convolutions conv1,conv2 are preferred for performance reasons.
        Note that CNNs can mix nodes with 2d and 3d convolutions as 2d is a special
        case of 3d with filter size 1 on the respective axis. In this case conv3
        should be used for the axis with smallest filter size.
    strides
    fov
    dtype: str
        corresponding to numpy dtype (e.g., 'int64').
        Default is floatX from theano config
    apply_train: bool
    value
    init_kwargs: dict
    name: str
        Node name.
    print_repr: bool
        Whether to print the node representation upon initialisation.
    """

    def __init__(self, shape, tags, strides=None, fov=None, dtype=theano.config.floatX, apply_train=False,
                 value=None, init_kwargs=None, name='value_state', print_repr=True):
        super(ValueNode, self).__init__(None, name, print_repr)

        if isinstance(value, list): # Handle locking of params
            if value[1]=='const' and len(value==2):
                value = value[0]
                apply_train = False

        self.is_source = False
        self._shape = graphutils.TaggedShape(shape, tags, strides, fov=fov)
        if type(dtype)!=str:
            raise ValueError("dtype must be a string.")
        self._dtype = dtype
        self._apply_train = apply_train
        self._value = value
        self._init_kwargs = init_kwargs

    def _make_output(self):
        broadcast = (False,)*len(self._shape)
        p_name = '<%s%s>' % ('state_', tuple(self._shape))
        if self._apply_train:
            self.output = VariableWeight(self._shape, dtype=self._dtype, value=self._value,
                                         init_kwargs=self._init_kwargs, name=p_name, broadcastable=broadcast)
        else:
            if self._init_kwargs is not None:
                raise NotImplementedError()

            if self._value is None:
                self._value = np.zeros(self._shape)
            else:
                if not isinstance(self._value, np.ndarray):
                    self._value = np.array(self._value)

            self.output = ConstantParam(self._value, name=p_name,
                                        dtype=self._dtype,
                                        make_singletons_broadcastable=False)
        self.params['value'] = self.output

    def _calc_shape(self):
        self.shape = self._shape

    def _calc_comp_cost(self):
        self.computational_cost = 0

    def get_value(self):
        return self.output.get_value()


class InitialState_like(Node):
    """

    Parameters
    ----------
    parent
    override_f
    dtype
    name
    print_repr
    init_kwargs
    """

    def __init__(self, parent, override_f, dtype=None, name='initial_state',
               print_repr=True, init_kwargs=None):
        assert isinstance(parent, Node)
        super(InitialState_like, self).__init__(parent, name, print_repr)

        if init_kwargs is None:
            init_kwargs = dict(mode='const', scale=0.0)

        self._init_kwargs = init_kwargs
        shape = list(parent.shape.shape)
        tags  = parent.shape.tags
        strides = parent.shape.strides
        fov = parent.shape.fov
        if dtype is None:
            dtype = parent.output.dtype

        self.axis = parent.shape.tag2index('f')

        shape[self.axis] = override_f
        self._shape = graphutils.TaggedShape(shape, tags, strides, fov=fov)
        self._dtype = dtype
        self.override_f = override_f

    def _make_output(self):
        p_name = '<%s%s>' % ('initial_state_', self.override_f)
        single = VariableWeight(shape=self.override_f, dtype=self._dtype,
                                     init_kwargs=self._init_kwargs,
                                     name=p_name)
        self.params['value'] = single
        pattern = ['x' for i in self.parent.shape]
        pattern[self.axis] = 0
        single = single.dimshuffle(pattern)
        sh = list(self.parent.output.shape)
        sh[self.axis] = 1
        broadcaster = T.ones(sh, dtype=self._dtype)
        self.output = single * broadcaster

    def _calc_shape(self):
        self.shape = self._shape

    def _calc_comp_cost(self):
        self.computational_cost = 0

    def get_value(self):
        return self.params['value'].get_value()


