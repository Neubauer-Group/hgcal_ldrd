import uproot
import awkward
import numpy as np
from scipy.sparse import csr_matrix, find
from scipy.spatial import cKDTree
from tqdm import tqdm_notebook as tqdm


import os
import os.path as osp

print(os.environ['GNN_TRAINING_DATA_ROOT'])

fname = osp.join(os.environ['GNN_TRAINING_DATA_ROOT'], 'single_taus/root/partGun_PDGid15_x1000_Pt3.0To100.0_NTUP_9.root')

print(type(fname))

test = uproot.open(fname)['ana']['hgc']

