# -*- coding: utf-8 -*-
"""
Created on Tue Aug 30 13:43:50 2016

@author: Marius Felix Killinger
"""
from __future__ import absolute_import, division, print_function
from builtins import filter, hex, input, int, map, next, oct, pow, range, super, zip


import os
import numpy as np
import matplotlib.pyplot as plt


from elektronn2 import utils
from elektronn2.utils.plotting import scroll_plot
from elektronn2.data.image import center_cubes
from elektronn2 import neuromancer
from elektronn2.data import knossos_array

from subprocess import call
import argparse, ast


def parseargs():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default=None)
    parsed = parser.parse_args()
    return parsed.model_path,

model_path, = parseargs()

if 'knossos_raw' not in locals():
     knossos_raw = knossos_array.KnossosArray(os.path.expanduser("~/lustre/kornfeld/cubedStacks/j0126_cubed/"), max_ram=7000, n_preload=0)
    
def colorize_10ch(pred):
    pred_color = np.zeros(pred.shape[1:]+(3,), dtype=np.float32)
    r = np.array([1,0,0])[None,None,None]
    g = np.array([0,1,0])[None,None,None]
    b = np.array([0,0,1])[None,None,None]
    pred_color += pred[1][...,None] * (r+g+b) # membrane
    pred_color += pred[2][...,None] * (r+g+0.85*b) # ecs
    pred_color += pred[4][...,None] * (0.35*r+0.64*g+b) # mito
    pred_color = np.clip(pred_color + pred[5][...,None] * (1*r+0.65*g-2*b), 0, 1) # ves
    pred_color = np.clip(pred_color + pred[6][...,None] * (1*r-0.5*g-0.5*b), 0, 1) # syn
    pred_color = np.clip(pred_color + pred[8][...,None] * (+0.3*r-0.8*g-0.2*b), 0, 1) # my_out, green
    pred_color = np.clip(pred_color + pred[9][...,None] * (0.5*r+0.3*g-0.2*b), 0, 1) # my_in
    return pred_color
    


def predict_centered(knossos_raw_, model, pos, excess=[0, 2, 2]):
    pos = np.array(pos)
    excess = np.array(excess)
    in_sh = model.input_node.shape.spatial_shape
    out_sh = model.prediction_node.shape.spatial_shape
    add = np.multiply(out_sh, excess)
    # diff = np.array(np.subtract(in_sh, out_sh))

    half = np.divide(in_sh, 2).astype(np.int)
    img = knossos_raw_.cut_slice(in_sh + 2 * add, pos - half - add)
    pred = model.predict_dense(img[None])

    return pred, img



if __name__ == '__main__':
    # models = [
    #         "/home/mfk/axon/mkilling/CNN_Training/3D/DS-1-0-fusion/Backup/DS-1-0-fusion-73.0h.mdl",
    #         "/home/mfk/axon/mkilling/CNN_Training/3D/DS-1-6-fusion-lr+/Backup/DS-1-6-fusion-lr+-94.0h.mdl",
    #         "/home/mfk/axon/mkilling/CNN_Training/3D/DS-1-13-fusion-pool-Conv/Backup/DS-1-13-fusion-pool-Conv-73.0h.mdl",
    #         "/home/mfk/axon/mkilling/CNN_Training/3D/DS-1-14-fusion-pool-Conv-new/Backup/DS-1-14-fusion-pool-Conv-new-73.0h.mdl",
    #         "/home/mfk/axon/mkilling/CNN_Training/3D/DS-2-0-normal/Backup/DS-2-0-normal-34.0h.mdl",
    #         "/home/mfk/axon/mkilling/CNN_Training/3D/DS-2-1-2xConv/Backup/DS-2-1-2xConv-40.0h.mdl",
    #         "/home/mfk/axon/mkilling/CNN_Training/3D/DS-2-2-2xConv-barr-side/Backup/DS-2-2-2xConv-barr-side-34.0h.mdl",
    #         "/home/mfk/CNN_Training/3D/DS-2-3-1xConv-barr-side/Backup/DS-2-3-1xConv-barr-side-34.0h.mdl"
    #         ]
    # if False:
    #     if model_path is None:
    #         for model_path in models:
    #             fp = "/home/mfk/axon/mkilling/investigation/Figures/ModelPred.py"
    #             call("python %s --model_path=%s"%(fp, model_path), shell=True, executable="/bin/bash")
    #
    #     else:
    #         raw = utils.h5load("/home/mfk/BirdGT/test_raw_6452_6408_2926.h5")
    #         model = neuromancer.model.modelload(model_path)
    #         pred = model.predict_dense(raw)
    #         pred_color = colorize_10ch(pred)
    #
    #         utils.h5save(pred, "/home/mfk/BirdGT/pred_%s.h5"%os.path.split(model_path)[1][:-4])
    #         utils.h5save(pred_color, "/home/mfk/BirdGT/pred_color_%s.h5"%os.path.split(model_path)[1][:-4])
    #
    #         #plt.imsave("/home/mfk/BirdGT/pred_color_s10_%s.png"%os.path.split(model_path)[1][:-4], pred_color[10])
    #         #plt.imsave("/home/mfk/BirdGT/pred_color_s17_%s.png"%os.path.split(model_path)[1][:-4], pred_color[17])
    #         #plt.imsave("/home/mfk/BirdGT/pred_color_s24_%s.png"%os.path.split(model_path)[1][:-4], pred_color[24])
    #         #plt.imsave("/home/mfk/BirdGT/pred_color_s27_%s.png"%os.path.split(model_path)[1][:-4], pred_color[27])
    #
    #         #z = pred.shape[1]//2
    #         #fig0 = scroll_plot(pred_color, 'pred')
    #         #fig1 = scroll_plot(pred[1,:]+pred[2,:], 'mem')
    # if True:
    #     preds = []
    #     models = list(map(lambda x: os.path.split(x)[1][:-4], models))
    #     for model_path in models:
    #         pred = utils.h5load("/home/mfk/BirdGT/pred_color_%s.h5"%model_path)
    #         preds.append(pred)
    #
    #     i = np.argmin(list(map(lambda x: x.shape[-1], preds)))
    #     preds_new = []
    #
    #     for p in preds:
    #         p, _ = center_cubes(p, preds[i])
    #         preds_new.append(p)
    #
    #     fig1 = scroll_plot(preds_new[:4], models[:4])
    #     fig2 = scroll_plot(preds_new[4:], models[4:])
#
#    data = knossos_array.KnossosArrayMulti('~/lustre/',
#           ["kornfeld/cubedStacks/j0126_cubed/","sdorkenw/j0126_3d_rrbarrier/"],
#                                           max_ram=10000, n_preload=0)
#
#    model_path = "/home/mfk/axon/mkilling/CNN_Training/3D/DualScale/DS-2-1-2xConv/Backup/DS-2-1-2xConv-88.0h.mdl"
#
#    p0 = ""#"/home/mfk/axon/mkilling/CNN_Training/3D/"
#    models = ["/home/mfk/CNN_Training/3D/DS-3-0/Backup/DS-3-0-410k.mdl",
#              "DualScale/DS-2-1-2xConv/Backup/DS-2-1-2xConv-88.0h.mdl",
#              "DS-2-0-normal-aftertrain-lr-/DS-2-0-normal-aftertrain-lr--LAST.mdl",
#              "DS-2-0-normal-aftertrain-moremy/DS-2-0-normal-aftertrain-moremy-LAST.mdl",
#              "DS-2-0-normal-aftertrain-moremy-SGD/Backup/DS-2-0-normal-aftertrain-moremy-SGD-70k.mdl",
#
#              "DS-2-0-normal-aftertrain/DS-2-0-normal-aftertrain-LAST.mdl",
#              "DS-2-0-normal-aftertrain-noneg/DS-2-0-normal-aftertrain-noneg-LAST.mdl",
#              ]
#    preds = []
#    figs = []
#    for m in models[0:1]:
#        model = neuromancer.model.modelload(p0+m, imposed_patch_size=[59-16,385-32,385-32])
#        #pred, img = predict_centered(knossos_raw, model, [6106, 2403, 3952][::-1], excess=[6,1,1])
#        pred, img = predict_centered(knossos_raw, model, [8788, 3984, 2886-15][::-1], excess=[10,1,1]) # ranvier
#        pred, img = predict_centered(knossos_raw, model, [4673, 3941, 1780][::-1], excess=[5,3,3]) # ranvier
#        #pred, img = predict_centered(knossos_raw, model, [8371, 3856, 2595][::-1], excess=[6,2,2]) # bad closed
#        #pred, img = center_cubes(pred, img)
#        barr = pred[1]+pred[2]
#        pc = colorize_10ch(pred)
#        preds.append((pred, barr, pc))
#        #fig2 = scroll_plot(np.swapaxes(pc, 0,1), 'pred')
#        #fig = scroll_plot(pc, 'pred', 0)
#        #fig2 = scroll_plot(pred[9], 'pred', 0)
#        #figs.append(fig)
#        #figs.append(fig2)
#
#    pics = [p[2] for p in preds]
#    fig = scroll_plot(preds[0][2], '1234')
#    nws = [[900, 900, 450], ] * 2
#    nwpos = [[4619, 5865, 2445], [3719, 5636, 2361]]
#
#    stuff = []
#    for off,sz in zip(nwpos, nws):
#        #img = data.cut_slice(sz[::-1], off[::-1])
#        ##raw = utils.h5load("/home/mfk/BirdGT/skeleton/skeleton_cube_raw-zyx-larger.h5")
#        #pred = model.predict_dense(img[0:1])
#        pred, img = predict_centered(model, [8349, 5820, 3213][::-1], excess=[0,6,6])
#        pred, img = center_cubes(pred, img)
#        pc = colorize_10ch(pred)
#        #barr = pred[1]+pred[2]
#        #barr_small, rbar_small = center_cubes(barr, img[1])
#        #fig = scroll_plot([barr_small, rbar_small ], ['bar', 'rbar'])
#        #stuff.append([img, pred, fig])
#
#[59+32,385+32,385+32] = 0.27
#[59+32,385+32+32,385+32+32] = 0.28
#[59+32,385+3*32,385+3*32] = 0.29
#import numpy as np
#from elektronn2.utils import gpu
#gpu.initgpu('auto')
#from elektronn2 import neuromancer
#model_path = "~/axon/mkilling/CNN_Training/3D/DualScale/DS-2-1-2xConv/Backup/DS-2-1-2xConv-88.0h.mdl"
#model = neuromancer.model.modelload(model_path, imposed_patch_size=[59+32,385+3*32,385+3*32])
#x = np.random.rand(*model.input_node.shape.shape[1:]).astype(np.float32)
#y = model.predict_dense(x)
#y = model.predict_dense(x)
#y = model.predict_dense(x)

    model = neuromancer.model.modelload("/home/mfk/CNN_Training/3D/DS-3-0/Backup/DS-3-0-400k.mdl",
                                        imposed_patch_size=[59-16,385-32,385-32])
    
    pred, img = predict_centered(knossos_raw, model, [6092, 5551, 3135][::-1], excess=[0,4,4])
    barr = pred[1]+pred[2]
    pc = colorize_10ch(pred)
    fig = scroll_plot(pc, '1234')
    img = pc[3,407:-408,407:-408]
    plt.imsave("/home/mfk/axon/mkilling/investigation/MA-TEX/figures/sXY_6092_5551_3135_dscnn2.png", img)
