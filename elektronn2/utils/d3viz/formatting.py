"""
Visualisation code taken form Theano
Original Author: Christof Angermueller <cangermueller@gmail.com>
Adapted with permission for the ELEKTRONN2 Toolkit by Marius Killinger 2016
Note that this code is licensed under the original terms of Theano (see license
containing directory).
"""

from __future__ import absolute_import, division, print_function
from builtins import filter, hex, input, int, map, next, oct, pow, range, \
    super, zip

import future.utils
import re
import os
import shutil
import logging
from collections import OrderedDict

from ...neuromancer.graphutils import TaggedShape


logger = logging.getLogger('elektronn2log')


__path__ = os.path.dirname(os.path.realpath(__file__))

pydot_imported = False
try:
    # pydot2 supports py3
    import pydotplus as pd


    if pd.find_graphviz():
        pydot_imported = True
except ImportError:
    try:
        # fall back on pydot if necessary
        import pydot as pd


        if pd.find_graphviz():
            pydot_imported = True
    except ImportError:
        pass  # tests should not fail on optional dependency

if not pydot_imported:
    logger.warning('Failed to import pydot/pydotplus. You must install '
                   'graphviz and either pydot or pydotplus for '
                   '`PyDotFormatter` to work.')


def sort(model, select_outputs):
    graph_sorted = OrderedDict()
    graph_unsorted = OrderedDict()
    for n in model.nodes.values():
        if n.is_source:
            graph_sorted[n] = True
        else:
            graph_unsorted[n] = True

    for n in graph_unsorted:
        if not len(n.children) and select_outputs:  # output node
            ### TODO
            raise NotImplementedError(
                "Select outputs does not work. Make sure "
                "that outputs which are needed from scan "
                "are not removed, that edges to removed "
                "nodes are not drawn and and that after "
                "removing unwanted output nodes in one sort "
                "there will be new unwanted terminal nodes")
            if n.name in select_outputs:
                graph_sorted[n] = True
        else:
            graph_sorted[n] = True

    return graph_sorted


class PyDotFormatter2(object):
    """Create `pydot` graph object from Theano function.

    Parameters
    ----------
    compact : bool
        if True, will remove intermediate variables without name.

    Attributes
    ----------
    node_colors : dict
        Color table of node types.
    apply_colors : dict
        Color table of apply nodes.
    shapes : dict
        Shape table of node types.
    """

    def __init__(self, compact=True):
        """Construct PyDotFormatter object."""
        if not pydot_imported:
            raise ImportError(
                'Failed to import pydot/pydotplus. You must install '
                'graphviz and either pydot or pydotplus for '
                '`PyDotFormatter` to work.')

        self.compact = compact
        self.__node_prefix = 'n'

    def __add_node(self, node):
        """Add new node to node list and return unique id.

        Parameters
        ----------
        node : Theano graph node
            Apply node, tensor variable, or shared variable in compute graph.

        Returns
        -------
        str
            Unique node id.
        """
        assert node not in self.__nodes
        _id = '%s%d' % (self.__node_prefix, len(self.__nodes) + 1)
        self.__nodes[node] = _id
        return _id

    def __node_id(self, node):
        """Return unique node id.

        Parameters
        ----------
        node : Theano graph node
            Apply node, tensor variable, or shared variable in compute graph.

        Returns
        -------
        str
            Unique node id.
        """
        if node in self.__nodes:
            return self.__nodes[node]
        else:
            return self.__add_node(node)

    def get_node_props(self, node):
        cls_name = node.__class__.__name__
        if cls_name.startswith('_'):
            cls_name = cls_name[1:]
        __node_id = self.__node_id(node)

        node_type = 0  # normal
        if node.is_source:
            node_type = 1
        elif len(node.children)==0:
            node_type = 2

        nparams = {}
        nparams['name'] = __node_id
        nparams['label'] = "%s - %s" % (cls_name, node.name)
        nparams['profile'] = [0, 1e-5]
        nparams['style'] = 'filled'
        nparams['type'] = 'colored'

        nparams['shape'] = 'ellipse'
        nparams['fillcolor'] = '#008000'  # 'green'
        if cls_name=='Conv':
            nparams['shape'] = 'invtrapezium'
        if cls_name=='UpConv':
            nparams['shape'] = 'trapezium'
        if cls_name=='Perceptron':
            nparams['shape'] = 'octagon'
        if cls_name in ['GRU', 'LSTM']:
            nparams['shape'] = 'doubleoctagon'
        if cls_name=='Concat':
            nparams['shape'] = 'house'
        if cls_name=='ScanN':
            nparams['shape'] = 'doublecircle'
            nparams['fillcolor'] = 'red'
        if 'loss' in cls_name.lower() or 'nll' in cls_name.lower():
            nparams['shape'] = 'diamond'
            nparams['fillcolor'] = '#FFAA22'
        if cls_name=='ValueNode':
            nparams['shape'] = 'box'

        if node_type==2:
            nparams['fillcolor'] = 'blue'
            nparams['shape'] = 'box'
        elif node_type==1:
            nparams['fillcolor'] = 'yellow'
            nparams['shape'] = 'box'

        if isinstance(node.shape, TaggedShape) and node.shape.ndim:
            nparams['dtype'] = node.shape  # .ext_repr
        elif node.shape:
            nparams['dtype'] = node.shape

        nparams['tag'] = None  # 'tag' # None Not needed?
        nparams['node_type'] = 'node type'  # not needed?
        nparams['apply_op'] = 'apply_op'  # not needed?
        return nparams

    def __call__(self, model, select_outputs=None):
        """Create pydot graph from function.

        Parameters
        ----------
        model: model object

        Returns
        -------
        pydot.Dot
            Pydot graph of `fct`
        """
        graph = pd.Dot()
        self.__nodes = {}
        if select_outputs is not None and isinstance(select_outputs, str):
            select_outputs = [select_outputs, ]

        nodes = sort(model, select_outputs)
        # Create nodes
        for node in nodes:
            nparams = self.get_node_props(node)
            pd_node = dict_to_pdnode(nparams)
            graph.add_node(pd_node)

        # Create edges
        for node in nodes:
            for i, c in enumerate(node.children.values()):
                if c.__class__.__name__=="ScanN":
                    if node in c.in_memory:
                        # print("Skipping",node,'for',c)
                        continue
                p_id = self.__node_id(node)
                c_id = self.__node_id(c)
                edge_params = {}
                edge_params['color'] = 'black'
                edge_label = " "  # str(i)
                pdedge = pd.Edge(p_id, c_id, label=edge_label, **edge_params)
                graph.add_edge(pdedge)

            if node.__class__.__name__=='ScanN':
                self.add_scan_edges(node, graph, nodes)

        return graph

    def add_scan_edges(self, scan, graph, nodes):
        n = str(scan.n_steps) if scan.n_steps else "variable"
        # if scan.out_memory:
        #     out = scan.out_memory
        # else:
        #     out = scan.step_result
        out = []
        for i in scan.out_memory_sl:
            name = scan.output_names[i]
            for nd in nodes:
                if nd.name==name:
                    out.append(nd)
                    break
            else:
                out.append("NotFound")

        # out = [nodes[name] for name in scan.output_name]

        for p, c in zip(out, scan.in_memory):
            for ci in c.children.values():
                if ci.__class__.__name__=="ScanN":
                    continue
                    # if c in c.in_memory:

                p_id = self.__node_id(p)
                c_id = self.__node_id(ci)
                edge_params = {}
                edge_params['color'] = 'red'
                edge_params['constraint'] = False
                edge_params['penwidth'] = 3
                edge_label = n + "x recur.\nreplace %s" % c.name
                pdedge = pd.Edge(p_id, c_id, label=edge_label, **edge_params)
                graph.add_edge(pdedge)

        if scan.in_iterate:
            for p, c in zip(scan.in_iterate, scan.in_iterate_0):
                p_id = self.__node_id(p)
                c_id = self.__node_id(c)
                edge_params = {}
                edge_params['color'] = 'red'
                edge_params['constraint'] = False
                edge_params['penwidth'] = 3
                edge_label = n + "x recur.\niteration"
                pdedge = pd.Edge(p_id, c_id, label=edge_label, **edge_params)
                graph.add_edge(pdedge)


def dict_to_pdnode(d):
    """Create pydot node from dict."""
    e = dict()
    for k, v in d.items():
        if v is not None:
            if isinstance(v, list):
                v = '\t'.join([str(x) for x in v])
            else:
                v = str(v)
            v = str(v)
            v = v.replace('"', '\'')
            e[k] = v
    pynode = pd.Node(**e)
    return pynode


def replace_patterns(x, replace):
    """Replace `replace` in string `x`.

    Parameters
    ----------
    s : str
        String on which function is applied
    replace : dict
        `key`, `value` pairs where key is a regular expression and `value` a
        string by which `key` is replaced
    """
    for from_, to in replace.items():
        x = x.replace(str(from_), str(to))
    return x


def escape_quotes(s):
    """Escape quotes in string.

    Parameters
    ----------
    s : str
        String on which function is applied
    """
    s = re.sub(r'''(['"])''', r'\\\1', s)
    return s


def visualise_model(model, outfile, copy_deps=True, select_outputs=None,
                    image_format='png', *args, **kwargs):
    """
    Parameters
    ----------
    model : model object
    outfile : str
        Path to output HTML file.
    copy_deps : bool, optional
        Copy javascript and CSS dependencies to output directory.

    Notes
    -----
    This function accepts extra parameters which will be forwarded to
    :class:`theano.d3viz.formatting.PyDotFormatter`.

    """

    outfile = os.path.expanduser(outfile)

    # Create DOT graph
    formatter = PyDotFormatter2(*args, **kwargs)
    graph = formatter(model, select_outputs=select_outputs)
    graph.write(outfile + '.' + image_format, prog='dot', format=image_format)

    dot_graph_raw = graph.create_dot()
    if not future.utils.PY2:
        dot_graph_raw = dot_graph_raw.decode('utf8')
    dot_graph = escape_quotes(dot_graph_raw).replace('\n', '').replace('\r',
                                                                       '')

    # Create output directory if not existing
    outdir = os.path.dirname(outfile)
    if not outdir=='' and not os.path.exists(outdir):
        os.makedirs(outdir)

    # Read template HTML file
    template_file = os.path.join(__path__, 'html', 'template.html')
    with open(template_file) as f:
        template = f.read()

    # Copy dependencies to output directory
    src_deps = __path__
    if copy_deps:
        dst_deps = 'd3viz'
        for d in ['js', 'css']:
            dep = os.path.join(outdir, dst_deps, d)
            if not os.path.exists(dep):
                shutil.copytree(os.path.join(src_deps, d), dep)
    else:
        dst_deps = src_deps

    # Replace patterns in template
    replace = {'%% JS_DIR %%': os.path.join(dst_deps, 'js'),
               '%% CSS_DIR %%': os.path.join(dst_deps, 'css'),
               '%% DOT_GRAPH %%': dot_graph,}
    html = replace_patterns(template, replace)

    # Write HTML file
    with open(outfile + '.html', 'w') as f:
        f.write(html)

    graph.write(outfile + '.' + image_format, prog='dot', format=image_format)
