# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 21:03:47 2015

@author: Marius Felix Killinger
s4."""
from __future__ import absolute_import, division, print_function

import numpy as np
import matplotlib.pyplot as plt

from .. import neuromancer as gr
from .. import utils as ut
import elektronn2.utils.cnncalculator
from ..neuromancer.neural import Conv, FragmentsToDense, Perceptron, UpConv

# gm = gr.node_basic.graph_manager
# from ..neuromancer.node_basic import model_manager as graph_manager
from ..neuromancer.node_basic import model_manager


def test_restore():
    # records = ut.pickleload('/tmp/model.pkl')
    # graph_manager.restore(records)
    # print "="*50
    # print "="*50
    # out = graph_manager.sinks
    # print out
    # x_val = np.linspace(0,1, num=64*784).astype(np.float32).reshape((64, 784))
    # loss = graph_manager.nodes['loss']
    # enc_mu = graph_manager.nodes['enc mu']
    # print loss(x_val)
    # print enc_mu(x_val)

    # pack = False
    # if pack:
    #    inp = gr.Input((2,6,20,20), 'b,f,x,y')
    #    out1, out2 = gr.split(inp, 'f', 2, name='test_split')
    #    x_val = np.random.rand(2,6,20,20).astype(np.float32)
    #    print out1(x_val).shape, out2(x_val).shape
    #    ut.picklesave(graph_manager.get_records(), '/tmp/model2.pkl')
    #    print graph_manager.nodes.keys()
    #
    # if not pack:
    #    graph_manager.reset()
    #    records = ut.pickleload('/tmp/model2.pkl')
    #    graph_manager.restore(records)
    #    print "="*50
    #    print "="*50
    #    out = graph_manager.sinks
    #    print out
    #    out1 = graph_manager.nodes['test_split1']
    #    out2 = graph_manager.nodes['test_split2']
    #    x_val = np.random.rand(2,6,20,20).astype(np.float32)
    #    print out1(x_val).shape, out2(x_val).shape
    #    print graph_manager.nodes.keys()

    in_sh = (1, 1, 183, 183, 31)
    x = gr.Input(in_sh, 'b,f,z,x,y')
    in_val = np.random.rand(*in_sh).astype(np.float32)

    y1 = Conv(x, 12, (1, 6, 6)[::-1], (1, 2, 2)[::-1], mfp=False)
    y2 = Conv(y1, 24, (4, 4, 4)[::-1], (2, 2, 2)[::-1], mfp=False)
    y3 = Conv(y2, 64, (4, 4, 4)[::-1], (1, 2, 2)[::-1], mfp=False)
    y4 = Conv(y3, 64, (4, 4, 4)[::-1], (1, 1, 1)[::-1], mfp=False)
    # z4 = Perceptron(y4, 12)
    s4 = UpConv(y4, 64, (4, 4, 4), (2, 2, 2))
    d4 = y4.make_dual(y4, activation_func='lin')
    p4 = gr.Softmax(d4)

    lab = gr.Input_like(p4, name='lab')
    loss = gr.loss.MultinoulliNLL(p4, lab)

    # y5 = FragmentsToDense(y4, name='MFP-reshape')
    #
    # utils.cnncalculator.cnncalculator([6,4,4,4], [2,2,2,1])
