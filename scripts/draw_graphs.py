# Copyright (c) 2019, IRIS-HEP
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
import argparse
import numpy as np
import os
from scipy.sparse import csr_matrix, find
from scipy.spatial import cKDTree
from tqdm import tqdm_notebook as tqdm

from datasets.graph import draw_sample
import torch
import matplotlib.pyplot as plt
from matplotlib import collections  as mc


import glob
import mlflow

def draw_sample(X, Ri, Ro, y, out,
                cmap='bwr_r',
                skip_false_edges=True,
                alpha_labels=False,
                sim_list=None,
                plot_prefix="plot"):
    # let's draw only the non-noise edges
    out_mask = out > 0
    Ri = Ri[out_mask]
    Ro = Ro[out_mask]
    good_outs = out[out_mask]

    # Select the i/o node features for each segment
    feats_o = X[Ro]
    feats_i = X[Ri]
    # Prepare the figure
    fig, (ax0, ax1, ax2) = plt.subplots(3, 1, dpi=400, figsize=(10, 20))
    cmap = plt.get_cmap(cmap)

    # if sim_list is None:
    # Draw the hits (layer, x, y)
    #    ax0.scatter(X[:,0], X[:,2], c='k')
    #    ax1.scatter(X[:,1], X[:,2], c='k')
    # else:

    # e_max = np.max(X[:,4])
    # e_normed = np.tanh(X[:,4]/e_max)#1. / (1. + np.exp(-X[:,4]/e_max))
    e_max = np.max(1000 * X[:, 2])
    e_normed = np.tanh(1000 * X[:, 2] / e_max)  # 1. / (1. + np.exp(-X[:,4]/e_max))

    ax0.scatter(np.pi * X[:, 1], 1000 * X[:, 0], s=(e_normed), c='k')
    ax1.scatter(1000 * X[:, 2], 1000 * X[:, 0], s=(e_normed), c='k')
    ax2.scatter(1000 * X[:, 2], np.pi * X[:, 1], s=(e_normed), c='k')

    lines0 = []
    lines1 = []
    lines2 = []
    colors = []
    # Draw the segments
    if out is not None:
        # t = tqdm.tqdm()
        color_map = {0: (1, 1, 1, 1),
                     1: (0, 0, 1, 1),
                     2: (1, 0, 0, 1),
                     3: (0, 1, 0, 1)}

        for j in tqdm(range(good_outs.shape[0])):
            lines0.append([(np.pi * feats_o[j, 1], 1000 * feats_o[j, 0]),
                           (np.pi * feats_i[j, 1], 1000 * feats_i[j, 0])])
            lines1.append([(1000 * feats_o[j, 2], 1000 * feats_o[j, 0]),
                           (1000 * feats_i[j, 2], 1000 * feats_i[j, 0])])
            lines2.append([(1000 * feats_o[j, 2], np.pi * feats_o[j, 1]),
                           (1000 * feats_i[j, 2], np.pi * feats_i[j, 1])])
            colors.append(color_map[good_outs[j]])

            # _ = ax0.plot([feats_o[j,0], feats_i[j,0]],
            #             [feats_o[j,2], feats_i[j,2]], '-', **seg_args)
            # _ = ax1.plot([feats_o[j,1], feats_i[j,1]],
            #             [feats_o[j,2], feats_i[j,2]], '-', **seg_args)
    else:
        t = tqdm.tqdm(range(y.shape[0]))
        for j in t:
            if y[j]:
                seg_args = dict(c='b', alpha=0.4)
            elif not skip_false_edges:
                seg_args = dict(c='black', alpha=0.4)
            else:
                continue

            ax0.plot([np.pi * feats_o[j, 1], np.pi * feats_i[j, 1]],
                     [1000 * feats_o[j, 0], 1000 * feats_i[j, 0]], '-', **seg_args)
            ax1.plot([1000 * feats_o[j, 2], 1000 * feats_i[j, 2]],
                     [1000 * feats_o[j, 0], 1000 * feats_i[j, 0]], '-', **seg_args)
            ax2.plot([1000 * feats_o[j, 2], 1000 * feats_i[j, 2]],
                     [np.pi * feats_o[j, 1], np.pi * feats_i[j, 1]], '-', **seg_args)

    col_arr = np.array(colors)
    lc0 = mc.LineCollection(lines0, colors=col_arr, linewidths=1)
    lc1 = mc.LineCollection(lines1, colors=col_arr, linewidths=1)
    lc2 = mc.LineCollection(lines2, colors=col_arr, linewidths=1)

    ax0.add_collection(lc0)
    ax1.add_collection(lc1)
    ax2.add_collection(lc2)
    # Adjust axes
    ax0.set_xlabel('$Phi$')
    ax0.set_ylabel('$R [mm]$')
    ax0.set_xlim(-3.5, 3.5)
    ax0.set_ylim(0, 1050)

    ax1.set_xlabel('$Z [mm]$')
    ax1.set_ylabel('$R [mm]$')
    ax1.set_xlim(-1200, 1200)
    ax1.set_ylim(0, 1050)

    ax2.set_xlabel('$Z [mm]$')
    ax2.set_ylabel('$Phi$')
    ax2.set_xlim(-1200, 1200)
    ax2.set_ylim(-3.5, 3.5)

    plt.tight_layout()
    plot_file = plot_prefix + 'edges.png'
    fig.savefig(plot_file)
    return plot_file


def draw_sample_barrel(X, Ri, Ro, y, out,
                       cmap='bwr_r',
                       skip_false_edges=True,
                       alpha_labels=False,
                       sim_list=None):
    # let's draw only the non-noise edges
    out_mask = out > 0
    Ri = Ri[out_mask]
    Ro = Ro[out_mask]
    good_outs = out[out_mask]

    # Select the i/o node features for each segment    
    feats_o = X[Ro]
    feats_i = X[Ri]
    # Prepare the figure
    fig, (ax0, ax1, ax2) = plt.subplots(3, 1, dpi=400, figsize=(10, 30))
    cmap = plt.get_cmap(cmap)

    # if sim_list is None:    
    # Draw the hits (layer, x, y)
    #    ax0.scatter(X[:,0], X[:,2], c='k')
    #    ax1.scatter(X[:,1], X[:,2], c='k')
    # else:        

    # e_max = np.max(X[:,4])
    # e_normed = np.tanh(X[:,4]/e_max)#1. / (1. + np.exp(-X[:,4]/e_max))
    e_max = np.max(1000 * X[:, 2])
    e_normed = np.tanh(1000 * X[:, 2] / e_max)  # 1. / (1. + np.exp(-X[:,4]/e_max))

    ax0.scatter(1000 * X[:, 0] * np.cos(np.pi * X[:, 1]),
                1000 * X[:, 0] * np.sin(np.pi * X[:, 1]), s=(e_normed), c='k')
    ax1.scatter(1000 * X[:, 0] * np.cos(np.pi * X[:, 1]),
                1000 * X[:, 0] * np.sin(np.pi * X[:, 1]), s=(e_normed), c='k')
    ax2.scatter(1000 * X[:, 0] * np.cos(np.pi * X[:, 1]),
                1000 * X[:, 0] * np.sin(np.pi * X[:, 1]), s=(e_normed), c='k')

    lines0 = []
    lines1 = []
    lines2 = []
    colors = []
    # Draw the segments    
    if out is not None:
        # t = tqdm.tqdm()
        color_map = {0: (1, 1, 1, 1),
                     1: (0, 0, 1, 1),
                     2: (1, 0, 0, 1),
                     3: (0, 1, 0, 1)}

        for j in tqdm(range(good_outs.shape[0])):
            lines0.append([(1000 * feats_o[j, 0] * np.cos(np.pi * feats_o[j, 1]),
                            1000 * feats_o[j, 0] * np.sin(np.pi * feats_o[j, 1])),
                           (1000 * feats_i[j, 0] * np.cos(np.pi * feats_i[j, 1]),
                            1000 * feats_i[j, 0] * np.sin(np.pi * feats_i[j, 1]))])
            lines1.append([(1000 * feats_o[j, 0] * np.cos(np.pi * feats_o[j, 1]),
                            1000 * feats_o[j, 0] * np.sin(np.pi * feats_o[j, 1])),
                           (1000 * feats_i[j, 0] * np.cos(np.pi * feats_i[j, 1]),
                            1000 * feats_i[j, 0] * np.sin(np.pi * feats_i[j, 1]))])
            lines2.append([(1000 * feats_o[j, 0] * np.cos(np.pi * feats_o[j, 1]),
                            1000 * feats_o[j, 0] * np.sin(np.pi * feats_o[j, 1])),
                           (1000 * feats_i[j, 0] * np.cos(np.pi * feats_i[j, 1]),
                            1000 * feats_i[j, 0] * np.sin(np.pi * feats_i[j, 1]))])

            colors.append(color_map[good_outs[j]])

            # _ = ax0.plot([feats_o[j,0], feats_i[j,0]],
            #             [feats_o[j,2], feats_i[j,2]], '-', **seg_args)
            # _ = ax1.plot([feats_o[j,1], feats_i[j,1]],
            #             [feats_o[j,2], feats_i[j,2]], '-', **seg_args)
    else:
        t = tqdm.tqdm(range(y.shape[0]))
        for j in t:
            if y[j]:
                seg_args = dict(c='b', alpha=0.4)
            elif not skip_false_edges:
                seg_args = dict(c='black', alpha=0.4)
            else:
                continue

            ax0.plot([1000 * feats_o[j, 0] * np.cos(np.pi * feats_o[j, 1]),
                      1000 * feats_i[j, 0] * np.cos(np.pi * feats_i[j, 1])],
                     [1000 * feats_o[j, 0] * np.sin(np.pi * feats_o[j, 1]),
                      1000 * feats_i[j, 0] * np.sin(np.pi * feats_i[j, 1])], '-',
                     **seg_args)

            ax1.plot([1000 * feats_o[j, 0] * np.cos(np.pi * feats_o[j, 1]),
                      1000 * feats_i[j, 0] * np.cos(np.pi * feats_i[j, 1])],
                     [1000 * feats_o[j, 0] * np.sin(np.pi * feats_o[j, 1]),
                      1000 * feats_i[j, 0] * np.sin(np.pi * feats_i[j, 1])], '-',
                     **seg_args)

            ax2.plot([1000 * feats_o[j, 0] * np.cos(np.pi * feats_o[j, 1]),
                      1000 * feats_i[j, 0] * np.cos(np.pi * feats_i[j, 1])],
                     [1000 * feats_o[j, 0] * np.sin(np.pi * feats_o[j, 1]),
                      1000 * feats_i[j, 0] * np.sin(np.pi * feats_i[j, 1])], '-',
                     **seg_args)

    col_arr = np.array(colors)
    lc0 = mc.LineCollection(lines0, colors=col_arr, linewidths=1)
    lc1 = mc.LineCollection(lines1, colors=col_arr, linewidths=1)
    lc2 = mc.LineCollection(lines2, colors=col_arr, linewidths=1)

    ax0.add_collection(lc0)
    # ax1.add_collection(lc1)
    ax2.add_collection(lc2)
    # Adjust axes
    ax0.set_xlabel('$X [mm]$')
    ax0.set_ylabel('$Y [mm]$')
    ax0.set_xlim(-1050, 1050)
    ax0.set_ylim(-1050, 1050)

    ax1.set_xlabel('$X [mm]$')
    ax1.set_ylabel('$Y [mm]$')
    ax1.set_xlim(-1050, 1050)
    ax1.set_ylim(-1050, 1050)

    ax2.set_xlabel('$X [mm]$')
    ax2.set_ylabel('$Y [mm]$')
    ax2.set_xlim(-1050, 1050)
    ax2.set_ylim(-1050, 1050)

    for k, clus in had_clusters_sel.items():
        clu = X[clus]
        # ax0.scatter(1000*clu[:,0]*np.cos(np.pi*clu[:,1]), 1000*clu[:,0]*np.sin(np.pi*clu[:,1]), marker = '^')
        ax1.scatter(1000 * clu[:, 0] * np.cos(np.pi * clu[:, 1]),
                    1000 * clu[:, 0] * np.sin(np.pi * clu[:, 1]), marker='^')
        ax2.scatter(1000 * clu[:, 0] * np.cos(np.pi * clu[:, 1]),
                    1000 * clu[:, 0] * np.sin(np.pi * clu[:, 1]), marker='^')

    plt.tight_layout()
    fig.savefig(plot_prefix + 'barrel.png')


def draw_sample_R_Phi(X, Ri, Ro, y, out,
                      cmap='bwr_r',
                      skip_false_edges=True,
                      alpha_labels=False,
                      sim_list=None):
    # let's draw only the non-noise edges
    out_mask = out > 0
    Ri = Ri[out_mask]
    Ro = Ro[out_mask]
    good_outs = out[out_mask]

    # Select the i/o node features for each segment    
    feats_o = X[Ro]
    feats_i = X[Ri]
    # Prepare the figure
    fig, (ax0, ax1, ax2) = plt.subplots(3, 1, dpi=400, figsize=(10, 20))
    cmap = plt.get_cmap(cmap)

    # if sim_list is None:    
    # Draw the hits (layer, x, y)
    #    ax0.scatter(X[:,0], X[:,2], c='k')
    #    ax1.scatter(X[:,1], X[:,2], c='k')
    # else:        

    # e_max = np.max(X[:,4])
    # e_normed = np.tanh(X[:,4]/e_max)#1. / (1. + np.exp(-X[:,4]/e_max))
    e_max = np.max(1000 * X[:, 2])
    e_normed = np.tanh(1000 * X[:, 2] / e_max)  # 1. / (1. + np.exp(-X[:,4]/e_max))

    ax0.scatter(np.pi * X[:, 1], 1000 * X[:, 0], s=(e_normed), c='k')
    ax1.scatter(np.pi * X[:, 1], 1000 * X[:, 0], s=(e_normed), c='k')
    ax2.scatter(np.pi * X[:, 1], 1000 * X[:, 0], s=(e_normed), c='k')

    lines0 = []
    lines1 = []
    lines2 = []
    colors = []
    # Draw the segments    
    if out is not None:
        # t = tqdm.tqdm()
        color_map = {0: (1, 1, 1, 1),
                     1: (0, 0, 1, 1),
                     2: (1, 0, 0, 1),
                     3: (0, 1, 0, 1)}

        for j in tqdm(range(good_outs.shape[0])):
            lines0.append([(np.pi * feats_o[j, 1], 1000 * feats_o[j, 0]),
                           (np.pi * feats_i[j, 1], 1000 * feats_i[j, 0])])
            lines1.append([(np.pi * feats_o[j, 1], 1000 * feats_o[j, 0]),
                           (np.pi * feats_i[j, 1], 1000 * feats_i[j, 0])])
            lines2.append([(np.pi * feats_o[j, 1], 1000 * feats_o[j, 0]),
                           (np.pi * feats_i[j, 1], 1000 * feats_i[j, 0])])
            colors.append(color_map[good_outs[j]])

            # _ = ax0.plot([feats_o[j,0], feats_i[j,0]],
            #             [feats_o[j,2], feats_i[j,2]], '-', **seg_args)
            # _ = ax1.plot([feats_o[j,1], feats_i[j,1]],
            #             [feats_o[j,2], feats_i[j,2]], '-', **seg_args)
    else:
        t = tqdm.tqdm(range(y.shape[0]))
        for j in t:
            if y[j]:
                seg_args = dict(c='b', alpha=0.4)
            elif not skip_false_edges:
                seg_args = dict(c='black', alpha=0.4)
            else:
                continue

            ax0.plot([np.pi * feats_o[j, 1], np.pi * feats_i[j, 1]],
                     [1000 * feats_o[j, 0], 1000 * feats_i[j, 0]], '-', **seg_args)
            ax1.plot([np.pi * feats_o[j, 1], np.pi * feats_i[j, 1]],
                     [1000 * feats_o[j, 0], 1000 * feats_i[j, 0]], '-', **seg_args)
            ax2.plot([np.pi * feats_o[j, 1], np.pi * feats_i[j, 1]],
                     [1000 * feats_o[j, 0], 1000 * feats_i[j, 0]], '-', **seg_args)

    col_arr = np.array(colors)
    lc0 = mc.LineCollection(lines0, colors=col_arr, linewidths=1)
    lc1 = mc.LineCollection(lines1, colors=col_arr, linewidths=1)
    lc2 = mc.LineCollection(lines2, colors=col_arr, linewidths=1)

    ax0.add_collection(lc0)
    # ax1.add_collection(lc1)
    ax2.add_collection(lc2)
    # Adjust axes
    ax0.set_xlabel('$Phi$')
    ax0.set_ylabel('$R [mm]$')
    ax0.set_xlim(-3.5, 3.5)
    ax0.set_ylim(0, 1050)

    ax1.set_xlabel('$Phi$')
    ax1.set_ylabel('$R [mm]$')
    ax1.set_xlim(-3.5, 3.5)
    ax1.set_ylim(0, 1050)

    ax2.set_xlabel('$Phi$')
    ax2.set_ylabel('$R [mm]$')
    ax2.set_xlim(-3.5, 3.5)
    ax2.set_ylim(0, 1050)

    for k, clus in had_clusters_sel.items():
        clu = X[clus]
        # ax0.scatter(np.pi*clu[:,1], 1000*clu[:,0], marker = '^')
        ax1.scatter(np.pi * clu[:, 1], 1000 * clu[:, 0], marker='^')
        ax2.scatter(np.pi * clu[:, 1], 1000 * clu[:, 0], marker='^')

    plt.tight_layout()
    fig.savefig(plot_prefix + 'R_Phi.png')
    # return fig;


# %%
# %matplotlib inline

# import matplotlib.pyplot as plt
# from matplotlib import collections  as mc

def draw_sample_R_Z(X, Ri, Ro, y, out,
                    cmap='bwr_r',
                    skip_false_edges=True,
                    alpha_labels=False,
                    sim_list=None):
    # let's draw only the non-noise edges
    out_mask = out > 0
    Ri = Ri[out_mask]
    Ro = Ro[out_mask]
    good_outs = out[out_mask]

    # Select the i/o node features for each segment    
    feats_o = X[Ro]
    feats_i = X[Ri]
    # Prepare the figure
    fig, (ax0, ax1, ax2) = plt.subplots(3, 1, dpi=400, figsize=(10, 20))
    cmap = plt.get_cmap(cmap)

    # if sim_list is None:    
    # Draw the hits (layer, x, y)
    #    ax0.scatter(X[:,0], X[:,2], c='k')
    #    ax1.scatter(X[:,1], X[:,2], c='k')
    # else:        

    # e_max = np.max(X[:,4])
    # e_normed = np.tanh(X[:,4]/e_max)#1. / (1. + np.exp(-X[:,4]/e_max))
    e_max = np.max(1000 * X[:, 2])
    e_normed = np.tanh(1000 * X[:, 2] / e_max)  # 1. / (1. + np.exp(-X[:,4]/e_max))

    ax0.scatter(1000 * X[:, 2], 1000 * X[:, 0], s=(e_normed), c='k')
    ax1.scatter(1000 * X[:, 2], 1000 * X[:, 0], s=(e_normed), c='k')
    ax2.scatter(1000 * X[:, 2], 1000 * X[:, 0], s=(e_normed), c='k')

    lines0 = []
    lines1 = []
    lines2 = []
    colors = []
    # Draw the segments    
    if out is not None:
        # t = tqdm.tqdm()
        color_map = {0: (1, 1, 1, 1),
                     1: (0, 0, 1, 1),
                     2: (1, 0, 0, 1),
                     3: (0, 1, 0, 1)}

        for j in tqdm(range(good_outs.shape[0])):
            lines0.append([(1000 * feats_o[j, 2], 1000 * feats_o[j, 0]),
                           (1000 * feats_i[j, 2], 1000 * feats_i[j, 0])])
            lines1.append([(1000 * feats_o[j, 2], 1000 * feats_o[j, 0]),
                           (1000 * feats_i[j, 2], 1000 * feats_i[j, 0])])
            lines2.append([(1000 * feats_o[j, 2], 1000 * feats_o[j, 0]),
                           (1000 * feats_i[j, 2], 1000 * feats_i[j, 0])])
            colors.append(color_map[good_outs[j]])

            # _ = ax0.plot([feats_o[j,0], feats_i[j,0]],
            #             [feats_o[j,2], feats_i[j,2]], '-', **seg_args)
            # _ = ax1.plot([feats_o[j,1], feats_i[j,1]],
            #             [feats_o[j,2], feats_i[j,2]], '-', **seg_args)
    else:
        t = tqdm.tqdm(range(y.shape[0]))
        for j in t:
            if y[j]:
                seg_args = dict(c='b', alpha=0.4)
            elif not skip_false_edges:
                seg_args = dict(c='black', alpha=0.4)
            else:
                continue

            ax0.plot([1000 * feats_o[j, 2], 1000 * feats_i[j, 2]],
                     [1000 * feats_o[j, 0], 1000 * feats_i[j, 0]], '-', **seg_args)
            ax1.plot([1000 * feats_o[j, 2], 1000 * feats_i[j, 2]],
                     [1000 * feats_o[j, 0], 1000 * feats_i[j, 0]], '-', **seg_args)
            ax2.plot([1000 * feats_o[j, 2], 1000 * feats_i[j, 2]],
                     [1000 * feats_o[j, 0], 1000 * feats_i[j, 0]], '-', **seg_args)

    col_arr = np.array(colors)
    lc0 = mc.LineCollection(lines0, colors=col_arr, linewidths=1)
    lc1 = mc.LineCollection(lines1, colors=col_arr, linewidths=1)
    lc2 = mc.LineCollection(lines2, colors=col_arr, linewidths=1)

    ax0.add_collection(lc0)
    # ax1.add_collection(lc1)
    ax2.add_collection(lc2)
    # Adjust axes
    ax0.set_xlabel('$Z [mm]$')
    ax0.set_ylabel('$R [mm]$')
    ax0.set_xlim(-1200, 1200)
    ax0.set_ylim(0, 1050)

    ax1.set_xlabel('$Z [mm]$')
    ax1.set_ylabel('$R [mm]$')
    ax1.set_xlim(-1200, 1200)
    ax1.set_ylim(0, 1050)

    ax2.set_xlabel('$Z [mm]$')
    ax2.set_ylabel('$R [mm]$')
    ax2.set_xlim(-1200, 1200)
    ax2.set_ylim(0, 1050)

    for k, clus in had_clusters_sel.items():
        clu = X[clus]
        # ax0.scatter(1000*clu[:,2],  1000*clu[:,0], marker = '^')
        ax1.scatter(1000 * clu[:, 2], 1000 * clu[:, 0], marker='^')
        ax2.scatter(1000 * clu[:, 2], 1000 * clu[:, 0], marker='^')

    plt.tight_layout()
    fig.savefig(plot_prefix + 'R_Z.png')
    # return fig;


# %%
# %matplotlib inline

# import matplotlib.pyplot as plt
# from matplotlib import collections  as mc

def draw_sample_Z_Phi(X, Ri, Ro, y, out,
                      cmap='bwr_r',
                      skip_false_edges=True,
                      alpha_labels=False,
                      sim_list=None):
    # let's draw only the non-noise edges
    out_mask = out > 0
    Ri = Ri[out_mask]
    Ro = Ro[out_mask]
    good_outs = out[out_mask]

    # Select the i/o node features for each segment    
    feats_o = X[Ro]
    feats_i = X[Ri]
    # Prepare the figure
    fig, (ax0, ax1, ax2) = plt.subplots(3, 1, dpi=400, figsize=(10, 20))
    cmap = plt.get_cmap(cmap)

    # if sim_list is None:    
    # Draw the hits (layer, x, y)
    #    ax0.scatter(X[:,0], X[:,2], c='k')
    #    ax1.scatter(X[:,1], X[:,2], c='k')
    # else:        

    # e_max = np.max(X[:,4])
    # e_normed = np.tanh(X[:,4]/e_max)#1. / (1. + np.exp(-X[:,4]/e_max))
    e_max = np.max(1000 * X[:, 2])
    e_normed = np.tanh(1000 * X[:, 2] / e_max)  # 1. / (1. + np.exp(-X[:,4]/e_max))

    ax0.scatter(1000 * X[:, 2], np.pi * X[:, 1], s=(e_normed), c='k')
    ax1.scatter(1000 * X[:, 2], np.pi * X[:, 1], s=(e_normed), c='k')
    ax2.scatter(1000 * X[:, 2], np.pi * X[:, 1], s=(e_normed), c='k')

    lines0 = []
    lines1 = []
    lines2 = []
    colors = []
    # Draw the segments    
    if out is not None:
        # t = tqdm.tqdm()
        color_map = {0: (1, 1, 1, 1),
                     1: (0, 0, 1, 1),
                     2: (1, 0, 0, 1),
                     3: (0, 1, 0, 1)}

        for j in tqdm(range(good_outs.shape[0])):
            lines0.append([(1000 * feats_o[j, 2], np.pi * feats_o[j, 1]),
                           (1000 * feats_i[j, 2], np.pi * feats_i[j, 1])])
            lines1.append([(1000 * feats_o[j, 2], np.pi * feats_o[j, 1]),
                           (1000 * feats_i[j, 2], np.pi * feats_i[j, 1])])
            lines2.append([(1000 * feats_o[j, 2], np.pi * feats_o[j, 1]),
                           (1000 * feats_i[j, 2], np.pi * feats_i[j, 1])])
            colors.append(color_map[good_outs[j]])

            # _ = ax0.plot([feats_o[j,0], feats_i[j,0]],
            #             [feats_o[j,2], feats_i[j,2]], '-', **seg_args)
            # _ = ax1.plot([feats_o[j,1], feats_i[j,1]],
            #             [feats_o[j,2], feats_i[j,2]], '-', **seg_args)
    else:
        t = tqdm.tqdm(range(y.shape[0]))
        for j in t:
            if y[j]:
                seg_args = dict(c='b', alpha=0.4)
            elif not skip_false_edges:
                seg_args = dict(c='black', alpha=0.4)
            else:
                continue

            ax0.plot([1000 * feats_o[j, 2], 1000 * feats_i[j, 2]],
                     [np.pi * feats_o[j, 1], np.pi * feats_i[j, 1]], '-', **seg_args)
            ax1.plot([1000 * feats_o[j, 2], 1000 * feats_i[j, 2]],
                     [np.pi * feats_o[j, 1], np.pi * feats_i[j, 1]], '-', **seg_args)
            ax2.plot([1000 * feats_o[j, 2], 1000 * feats_i[j, 2]],
                     [np.pi * feats_o[j, 1], np.pi * feats_i[j, 1]], '-', **seg_args)

    col_arr = np.array(colors)
    lc0 = mc.LineCollection(lines0, colors=col_arr, linewidths=1)
    lc1 = mc.LineCollection(lines1, colors=col_arr, linewidths=1)
    lc2 = mc.LineCollection(lines2, colors=col_arr, linewidths=1)

    ax0.add_collection(lc0)
    # ax1.add_collection(lc1)
    ax2.add_collection(lc2)
    # Adjust axes
    ax0.set_xlabel('$Z [mm]$')
    ax0.set_ylabel('$Phi$')
    ax0.set_xlim(-1200, 1200)
    ax0.set_ylim(-3.5, 3.5)

    ax1.set_xlabel('$Z [mm]$')
    ax1.set_ylabel('$Phi$')
    ax1.set_xlim(-1200, 1200)
    ax1.set_ylim(-3.5, 3.5)

    ax2.set_xlabel('$Z [mm]$')
    ax2.set_ylabel('$Phi$')
    ax2.set_xlim(-1200, 1200)
    ax2.set_ylim(-3.5, 3.5)

    for k, clus in had_clusters_sel.items():
        clu = X[clus]
        # ax0.scatter(1000*clu[:,2],  np.pi*clu[:,1], marker = '^')
        ax1.scatter(1000 * clu[:, 2], np.pi * clu[:, 1], marker='^')
        ax2.scatter(1000 * clu[:, 2], np.pi * clu[:, 1], marker='^')

    plt.tight_layout()
    fig.savefig(plot_prefix + 'Z_Phi.png')
    # return fig;


# %%
# %matplotlib inline

# import matplotlib.pyplot as plt
# from matplotlib import collections  as mc

def draw_sample_signedR_Z(X, Ri, Ro, y, out,
                          cmap='bwr_r',
                          skip_false_edges=True,
                          alpha_labels=False,
                          sim_list=None):
    # let's draw only the non-noise edges
    out_mask = out > 0
    Ri = Ri[out_mask]
    Ro = Ro[out_mask]
    good_outs = out[out_mask]

    # Select the i/o node features for each segment    
    feats_o = X[Ro]
    feats_i = X[Ri]
    # Prepare the figure
    fig, (ax0, ax1, ax2) = plt.subplots(3, 1, dpi=400, figsize=(10, 30))
    cmap = plt.get_cmap(cmap)

    # if sim_list is None:    
    # Draw the hits (layer, x, y)
    #    ax0.scatter(X[:,0], X[:,2], c='k')
    #    ax1.scatter(X[:,1], X[:,2], c='k')
    # else:        

    # e_max = np.max(X[:,4])
    # e_normed = np.tanh(X[:,4]/e_max)#1. / (1. + np.exp(-X[:,4]/e_max))
    e_max = np.max(1000 * X[:, 2])
    e_normed = np.tanh(1000 * X[:, 2] / e_max)  # 1. / (1. + np.exp(-X[:,4]/e_max))

    for j in tqdm(range(X.shape[0])):
        if np.sin(X[j, 1]) > 0:
            ax0.scatter(1000 * X[j, 2], 1000 * X[j, 0], s=(e_normed), c='k')
            ax1.scatter(1000 * X[j, 2], 1000 * X[j, 0], s=(e_normed), c='k')
            ax2.scatter(1000 * X[j, 2], 1000 * X[j, 0], s=(e_normed), c='k')
        else:
            ax0.scatter(1000 * X[j, 2], -1000 * X[j, 0], s=(e_normed), c='k')
            ax1.scatter(1000 * X[j, 2], -1000 * X[j, 0], s=(e_normed), c='k')
            ax2.scatter(1000 * X[j, 2], -1000 * X[j, 0], s=(e_normed), c='k')

    lines0 = []
    lines1 = []
    lines2 = []
    colors = []
    # Draw the segments    
    if out is not None:
        # t = tqdm.tqdm()
        color_map = {0: (1, 1, 1, 1),
                     1: (0, 0, 1, 1),
                     2: (1, 0, 0, 1),
                     3: (0, 1, 0, 1)}

        for j in tqdm(range(good_outs.shape[0])):
            if np.sin(feats_o[j, 1]) > 0:
                lines0.append([(1000 * feats_o[j, 2], 1000 * feats_o[j, 0]),
                               (1000 * feats_i[j, 2], 1000 * feats_i[j, 0])])
                lines1.append([(1000 * feats_o[j, 2], 1000 * feats_o[j, 0]),
                               (1000 * feats_i[j, 2], 1000 * feats_i[j, 0])])
                lines2.append([(1000 * feats_o[j, 2], 1000 * feats_o[j, 0]),
                               (1000 * feats_i[j, 2], 1000 * feats_i[j, 0])])
            else:
                lines0.append([(1000 * feats_o[j, 2], -1000 * feats_o[j, 0]),
                               (1000 * feats_i[j, 2], -1000 * feats_i[j, 0])])
                lines1.append([(1000 * feats_o[j, 2], -1000 * feats_o[j, 0]),
                               (1000 * feats_i[j, 2], -1000 * feats_i[j, 0])])
                lines2.append([(1000 * feats_o[j, 2], -1000 * feats_o[j, 0]),
                               (1000 * feats_i[j, 2], -1000 * feats_i[j, 0])])
            colors.append(color_map[good_outs[j]])

            # _ = ax0.plot([feats_o[j,0], feats_i[j,0]],
            #             [feats_o[j,2], feats_i[j,2]], '-', **seg_args)
            # _ = ax1.plot([feats_o[j,1], feats_i[j,1]],
            #             [feats_o[j,2], feats_i[j,2]], '-', **seg_args)
    else:
        t = tqdm.tqdm(range(y.shape[0]))
        for j in t:
            if y[j]:
                seg_args = dict(c='b', alpha=0.4)
            elif not skip_false_edges:
                seg_args = dict(c='black', alpha=0.4)
            else:
                continue

            if np.sin(feats_o[j, 1]) > 0:
                ax0.plot([1000 * feats_o[j, 2], 1000 * feats_i[j, 2]],
                         [1000 * feats_o[j, 0], 1000 * feats_i[j, 0]], '-', **seg_args)
                ax1.plot([1000 * feats_o[j, 2], 1000 * feats_i[j, 2]],
                         [1000 * feats_o[j, 0], 1000 * feats_i[j, 0]], '-', **seg_args)
                ax2.plot([1000 * feats_o[j, 2], 1000 * feats_i[j, 2]],
                         [1000 * feats_o[j, 0], 1000 * feats_i[j, 0]], '-', **seg_args)
            else:
                ax0.plot([1000 * feats_o[j, 2], 1000 * feats_i[j, 2]],
                         [-1000 * feats_o[j, 0], -1000 * feats_i[j, 0]], '-', **seg_args)
                ax1.plot([1000 * feats_o[j, 2], 1000 * feats_i[j, 2]],
                         [-1000 * feats_o[j, 0], -1000 * feats_i[j, 0]], '-', **seg_args)
                ax2.plot([1000 * feats_o[j, 2], 1000 * feats_i[j, 2]],
                         [-1000 * feats_o[j, 0], -1000 * feats_i[j, 0]], '-', **seg_args)

    col_arr = np.array(colors)
    lc0 = mc.LineCollection(lines0, colors=col_arr, linewidths=1)
    lc1 = mc.LineCollection(lines1, colors=col_arr, linewidths=1)
    lc2 = mc.LineCollection(lines2, colors=col_arr, linewidths=1)

    ax0.add_collection(lc0)
    # ax1.add_collection(lc1)
    ax2.add_collection(lc2)
    # Adjust axes
    ax0.set_xlabel('$Z [mm]$')
    ax0.set_ylabel('$signed R [mm]$')
    ax0.set_xlim(-1200, 1200)
    ax0.set_ylim(-1050, 1050)

    ax1.set_xlabel('$Z [mm]$')
    ax1.set_ylabel('$signed R [mm]$')
    ax1.set_xlim(-1200, 1200)
    ax1.set_ylim(-1050, 1050)

    ax2.set_xlabel('$Z [mm]$')
    ax2.set_ylabel('$signed R [mm]$')
    ax2.set_xlim(-1200, 1200)
    ax2.set_ylim(-1050, 1050)

    for k, clus in had_clusters_sel.items():
        clu = X[clus]
        for j in tqdm(range(clu.shape[0])):
            if np.sin(clu[j, 1]) > 0:
                # ax0.scatter(1000*clu[j,2],  1000*clu[j,0], marker = '^')
                ax1.scatter(1000 * clu[j, 2], 1000 * clu[j, 0], marker='^')
                ax2.scatter(1000 * clu[j, 2], 1000 * clu[j, 0], marker='^')
            else:
                # ax0.scatter(1000*clu[j,2],  -1000*clu[j,0], marker = '^')
                ax1.scatter(1000 * clu[j, 2], -1000 * clu[j, 0], marker='^')
                ax2.scatter(1000 * clu[j, 2], -1000 * clu[j, 0], marker='^')

    plt.tight_layout()
    fig.savefig(plot_prefix + 'signedR_Z.png')
    # return fig;



def main(args):
    from models.EdgeNetWithCategories import EdgeNetWithCategories
    with mlflow.start_run() as run:
        checkpoints = glob.glob(os.path.join(os.environ['GNN_TRAINING_DATA_ROOT'], "checkpoints","*best*"))
        model_path = list(checkpoints)[0]
        plot_prefix=os.path.basename(model_path).split(".")[0]
        test_data_path = os.path.join(os.environ['GNN_TRAINING_DATA_ROOT'],'test_track', 'processed', args.test_data)
        mdl = EdgeNetWithCategories(input_dim=3, hidden_dim=64, output_dim=2, n_iters=6).\
            to('cuda:0')

        mdl.load_state_dict(torch.load(model_path)['model'])
        mdl.eval()

        data = torch.load(test_data_path).to('cuda:0')
        with torch.no_grad():
            pred_edges = mdl(data).detach()
            pred_edges_np = pred_edges.cpu().numpy()

        print(np.unique(np.argmax(pred_edges_np,axis=-1), return_counts=True))
        print(torch.unique(data.y.cpu(), return_counts=True))

        X = data.x.cpu().numpy()
        index = data.edge_index.cpu().numpy().T
        Ro = index[:,0]
        Ri = index[:,1]
        y = data.y.cpu().numpy()

        out = np.argmax(pred_edges_np,axis=-1)

        from unionfind import unionfind
        finder_had = unionfind(X.shape[0])
        finder_pho = unionfind(X.shape[0])
        finder_mip = unionfind(X.shape[0])

        for i in tqdm(range(index.shape[0])):
            if out[i] == 1:
                finder_had.unite(index[i, 0], index[i, 1])
            if out[i] == 2:
                finder_pho.unite(index[i, 0], index[i, 1])
            if out[i] == 3:
                finder_mip.unite(index[i, 0], index[i, 1])

        had_roots = np.array([finder_had.find(i) for i in range(X.shape[0])],
                             dtype=np.uint32)
        pho_roots = np.array([finder_pho.find(i) for i in range(X.shape[0])],
                             dtype=np.uint32)
        mip_roots = np.array([finder_mip.find(i) for i in range(X.shape[0])],
                             dtype=np.uint32)

        # %%
        had_clusters = np.unique(had_roots, return_inverse=True, return_counts=True)
        pho_clusters = np.unique(pho_roots, return_inverse=True, return_counts=True)
        mip_clusters = np.unique(mip_roots, return_inverse=True, return_counts=True)

        hads = had_clusters[0][np.where(had_clusters[2] > 4)]
        ems = pho_clusters[0][np.where(pho_clusters[2] > 4)]
        mips = mip_clusters[0][np.where(mip_clusters[2] > 4)]

        had_clusters_sel = {i: np.where(had_roots == had)[0] for i, had in
                            enumerate(hads)}
        em_clusters_sel = {i: np.where(pho_roots == em)[0] for i, em in enumerate(ems)}
        mip_clusters_sel = {i: np.where(mip_roots == mip)[0] for i, mip in
                            enumerate(mips)}

        mlflow.log_artifact(draw_sample(X, Ri, Ro, y, y, plot_prefix=plot_prefix))
        #%%
        mlflow.log_artifact(draw_sample(X, Ri, Ro, y, out, plot_prefix=plot_prefix+"2"))

        draw_sample_barrel(X, Ri, Ro, y, out)
        draw_sample_R_Phi(X, Ri, Ro, y, out)
        draw_sample_R_Z(X, Ri, Ro, y, out)
        draw_sample_Z_Phi(X, Ri, Ro, y, out)
        draw_sample_signedR_Z(X, Ri, Ro, y, out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--test-data', required=True)
    args = parser.parse_args()
    main(args)

