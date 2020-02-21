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


def main(args):
    from models.EdgeNetWithCategories import EdgeNetWithCategories
    model_path = os.path.join(os.environ['GNN_TRAINING_DATA_ROOT'], "checkpoints", args.model)
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
import matplotlib.pyplot as plt
from matplotlib import collections  as mc

def draw_sample(X, Ri, Ro, y, out,
                cmap='bwr_r', 
                skip_false_edges=True,
                alpha_labels=False, 
                sim_list=None): 
    
    #let's draw only the non-noise edges
    out_mask = out > 0
    Ri = Ri[out_mask]
    Ro = Ro[out_mask]
    good_outs = out[out_mask]
    
    # Select the i/o node features for each segment    
    feats_o = X[Ro]
    feats_i = X[Ri]    
    # Prepare the figure
    fig, (ax0,ax1,ax2) = plt.subplots(3, 1, dpi=400, figsize=(10, 20))
    cmap = plt.get_cmap(cmap)
    
    
    #if sim_list is None:    
        # Draw the hits (layer, x, y)
    #    ax0.scatter(X[:,0], X[:,2], c='k')
    #    ax1.scatter(X[:,1], X[:,2], c='k')
    #else:        
    
    #e_max = np.max(X[:,4])
    #e_normed = np.tanh(X[:,4]/e_max)#1. / (1. + np.exp(-X[:,4]/e_max))
    e_max = np.max(1000*X[:,2])
    e_normed = np.tanh(1000*X[:,2]/e_max)#1. / (1. + np.exp(-X[:,4]/e_max))
    
    
    ax0.scatter(np.pi*X[:,1], 1000*X[:,0], s=(e_normed), c='k')
    ax1.scatter(1000*X[:,2], 1000*X[:,0], s=(e_normed), c='k')
    ax2.scatter(1000*X[:,2], np.pi*X[:,1], s=(e_normed), c='k')
        
        
    lines0 = []
    lines1 = []
    lines2 = []
    colors = []
    # Draw the segments    
    if out is not None:
        #t = tqdm.tqdm()
        color_map = {0: (1,1,1,1),
                     1: (0,0,1,1),
                     2: (1,0,0,1),
                     3: (0,1,0,1)}
        
        for j in tqdm(range(good_outs.shape[0])):
            lines0.append([(np.pi*feats_o[j,1], 1000*feats_o[j,0]), (np.pi*feats_i[j,1], 1000*feats_i[j,0])])
            lines1.append([(1000*feats_o[j,2],  1000*feats_o[j,0]), (1000*feats_i[j,2],  1000*feats_i[j,0])])
            lines2.append([(1000*feats_o[j,2],  np.pi*feats_o[j,1]),(1000*feats_i[j,2],  np.pi*feats_i[j,1])])
            colors.append(color_map[good_outs[j]])
            
            #_ = ax0.plot([feats_o[j,0], feats_i[j,0]],
            #             [feats_o[j,2], feats_i[j,2]], '-', **seg_args)
            #_ = ax1.plot([feats_o[j,1], feats_i[j,1]],
            #             [feats_o[j,2], feats_i[j,2]], '-', **seg_args)
    else:
        t = tqdm.tqdm(range(y.shape[0]))
        for j in t:
            if y[j]:
                seg_args = dict(c='b', alpha=0.4)
            elif not skip_false_edges:
                seg_args = dict(c='black', alpha=0.4)
            else: continue
                
            ax0.plot([np.pi*feats_o[j,1], np.pi*feats_i[j,1]],
                     [1000*feats_o[j,0],  1000*feats_i[j,0]], '-', **seg_args)
            ax1.plot([1000*feats_o[j,2],  1000*feats_i[j,2]],
                     [1000*feats_o[j,0],  1000*feats_i[j,0]], '-', **seg_args)
            ax2.plot([1000*feats_o[j,2],  1000*feats_i[j,2]],
                     [np.pi*feats_o[j,1], np.pi*feats_i[j,1]], '-', **seg_args)
        
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
    fig.savefig(plot_prefix + 'edges.png')



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument('--test-data', required=True)
    args = parser.parse_args()
    main(args)

