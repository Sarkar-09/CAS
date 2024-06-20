# This functions adapted from https://github.com/abojchevski/sparse_smoothing
from easydict import EasyDict
import numpy as np
import gmpy2
from tqdm import tqdm
from statsmodels.stats.proportion import proportion_confint
from itertools import product
from collections import defaultdict


import os
import pathlib
import sys

import pandas as pd
import torch
import torchvision
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import seaborn as sns
import numpy as np

import torch_geometric
from torch_geometric.data import Data as GraphData
import torch_geometric.datasets as pyg_datasets

import matplotlib.pyplot as plt
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"device = {device}")

from tgnnu.data_utils.splits import SplitManager
from tgnnu.networks.node_classif_models import GCN
from tgnnu.networks.node_classif_lightner import NodeLevelGNN

import gnn_cp.cp.transformations as cp_t
import gnn_cp.cp.graph_transformations as cp_gt
from gnn_cp.cp.graph_cp import GraphCP

from graph_split import GraphSplit

# import regions_binary
import cvxpy as convex


from scipy.stats import norm

from utils import ModelManager
from utils import standard_l2_norm

# assignments
datasets_folder = "path_to_datasets"
models_direction = "path_to_model"
from discrete_data.certify_utils import *
import pickle

def save_pkl(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
def load_pkl(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def regions_binary(ra, rd, pf_plus, pf_minus, precision=1000):
    """
    Construct (px, px_tilde, px/px_tilde) regions used to find the certified radius for binary data.

    Intuitively, pf_minus controls rd and pf_plus controls ra.

    Parameters
    ----------
    ra: int
        Number of ones y has added to x
    rd : int
        Number of ones y has deleted from x
    pf_plus : float, 0 <= p_plus <= 1
        The probability to flip a zero to a one
    pf_minus: float, 0 <= p_plus <= 1
        The probability to flip a one to a zero
    precision: int
        Numerical precision for floating point calculations

    Returns
    -------
    regions: array-like, [None, 3]
        Regions of constant probability under px and px_tilde,
    """

    pf_plus, pf_minus = gmpy2.mpfr(pf_plus), gmpy2.mpfr(pf_minus)
    with gmpy2.context(precision=precision):
        if pf_plus == 0:
            px = pf_minus ** rd
            px_tilde = pf_minus ** ra

            return np.array([[1 - px, 0, float('inf')],
                             [px, px_tilde, px / px_tilde],
                             [0, 1 - px_tilde, 0]
                             ])

        if pf_minus == 0:
            px = pf_plus ** ra
            px_tilde = pf_plus ** rd
            return np.array([[1 - px, 0, float('inf')],
                             [px, px_tilde, px / px_tilde],
                             [0, 1 - px_tilde, 0],
                             ])
        max_q = ra + rd
        i_vec = np.arange(0, max_q + 1)

        T = ra * ((pf_plus / (1 - pf_plus)) ** i_vec) + \
            rd * ((pf_minus / (1 - pf_minus)) ** i_vec)

        ratio = np.zeros_like(T)
        px = np.zeros_like(T)
        px[0] = 1

        for q in range(0, max_q + 1):
            ratio[q] = (pf_plus/(1-pf_minus)) ** (q - rd) * \
                (pf_minus/(1-pf_plus)) ** (q - ra)

            if q == 0:
                continue

            for i in range(1, q + 1):
                px[q] = px[q] + ((-1) ** (i + 1)) * T[i] * px[q - i]
            px[q] = px[q] / q

        scale = ((1-pf_plus) ** ra) * ((1-pf_minus) ** rd)

        px = px * scale

        regions = np.column_stack((px, px / ratio, ratio))
        if pf_plus+pf_minus > 1:
            # reverse the order to maintain decreasing sorting
            regions = regions[::-1]
        return regions


def standard_sparse_smoothing(input, p_add=0, p_del=0, p_add_edge=0, p_del_edge=0):
    # input is a torch_geometric.data.Data object
    # p_add: probability flipping a node feature if it is 0
    # p_del: probability flipping a node feature if it is 1
    # p_add_edge: probability adding an edge
    # p_del_edge: probability deleting an edge
    # return a new torch_geometric.data.Data object
    # with the same number of nodes and edges

    # random smoothing over features
    x = input.x.clone()
    x_add_mask = (torch.rand_like(x) < p_add) & (input.x == 0)
    x_del_mask = (torch.rand_like(x) < p_del) & (input.x == 1)
    x[x_add_mask] = 1
    x[x_del_mask] = 0

    # random smoothing over edges
    edge_index = input.edge_index.clone()
    edge_index = edge_index.T[edge_index[0] <= edge_index[1]].T
    add_edges = torch_geometric.utils.erdos_renyi_graph(num_nodes=x.shape[0], edge_prob=p_add_edge).to(device)
    # computing edge removal mask
    edge_removal_mask = (torch.rand(edge_index.shape[1]) < p_del_edge).to(device)
    edge_index = torch.cat([(edge_index.T[~edge_removal_mask]).T, add_edges], dim=1)
    edge_index = torch_geometric.utils.to_undirected(edge_index)
    edge_index = torch_geometric.utils.sort_edge_index(edge_index)

    duplicate_mask = torch.cat([torch.tensor([False]).to(device), ((edge_index[:, 1:] == edge_index[:, :-1]).sum(dim=0) == 2)])
    edge_index = edge_index[:, ~duplicate_mask]

    return GraphData(x=x, edge_index=edge_index, y=input.y.clone())
    

def standard_sparse_smoothing_concat(input, n_samples=1, p_add=0, p_del=0, p_add_edge=0, p_del_edge=0):
    samples = [standard_sparse_smoothing(input, p_add, p_del, p_add_edge, p_del_edge) for _ in range(n_samples)]
    final_x = torch.cat([sample.x for sample in samples], dim=0)
    final_edge_index = torch.cat([sample.edge_index + input.x.shape[0] * i for i, sample in enumerate(samples)], dim=1)
    final_y = torch.cat([sample.y for sample in samples], dim=0)
    
    include_train_mask = False
    final_train_mask = None
    if hasattr(input, "train_mask"):
        include_train_mask = True
        final_train_mask = torch.cat([input.train_mask for sample in samples], dim=0)
    
    include_train_mask = False
    final_val_mask = None
    if hasattr(input, "val_mask"):
        include_train_mask = True
        final_val_mask = torch.cat([input.val_mask for sample in samples], dim=0)

    dataset = GraphData(x=final_x, edge_index=final_edge_index, y=final_y, train_mask=final_train_mask, val_mask=final_val_mask)
    return dataset


def np_offset(scores_smooth, node_idx, class_idx,  pf_plus_att, pf_minus_att, ra=10, rd=1, alpha=0.05):
    eps = 0 
    scores = scores_smooth[node_idx, class_idx, :].cpu().numpy()
    p_emp = scores.mean() + eps
    reg = regions_binary(ra=ra, rd=rd, pf_plus=pf_plus_att, pf_minus=pf_minus_att)
    a = 0.0
    b = 1

    h = convex.Variable(len(reg))
    result = convex.Problem(convex.Maximize(convex.sum(h)), [h>=reg[:, 1]*a, h<=reg[:, 1]*b, h@reg[:, 2] == p_emp]).solve(solver='MOSEK')
    return result



def dkw_offset(scores_smooth, node_idx, class_idx,  pf_plus_att, pf_minus_att, ra=10, rd=1, alpha=0.05, num_s=1000):
#     bon_alpha = alpha / scores_smooth.shape[1]
    eps = 0 
    scores = scores_smooth[node_idx, class_idx, :].cpu().numpy()
    p_emp = scores.mean() + eps
    reg = regions_binary(ra=ra, rd=rd, pf_plus=pf_plus_att, pf_minus=pf_minus_att)
    a = 0
    b = 1

    s = np.linspace(a, b, num_s)[1:-1]

    # CDF-based upper bound
    h = convex.Variable((len(s), len(reg)))

    result = convex.Problem(convex.Maximize(s[0] + h@reg[:, 1] @ np.diff(list(s) + [b])[::-1]),
            [h>=0, h<=1, 
                h@reg[:, 0] ==  np.minimum((scores[:, None] > s[::-1]).mean(0) + eps, 1) ]).solve(solver='MOSEK')
    return result

def singleton_hit(pred_set, y_true):
    return ((pred_set[y_true])[pred_set.sum(axis=1) == 1].sum() / (pred_set).shape[0]).item()