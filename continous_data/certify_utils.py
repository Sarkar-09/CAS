import os
import pathlib
import sys
import argparse
import pandas as pd
import torch
import torchvision
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import seaborn as sns
import numpy as np

import matplotlib.pyplot as plt

from scipy.stats import norm

from utils import ModelManager
from utils import standard_l2_norm



randoms = (torch.rand((1000,)) * (0.73 - 0.72)) + 0.72

def np_upperbound(randoms, SIGMA, radi, alpha=0.05, n_classes=1):

    bon_alpha = alpha / n_classes
    # error = np.sqrt(np.log(1 / bon_alpha) / (2 * randoms.shape[0]))
    error = 0
    p_upper = torch.minimum(randoms.mean() + error, torch.tensor(1.0).to(randoms.device))
    result = norm.cdf(
        norm.ppf(p_upper.cpu(), scale=SIGMA) + radi,
        scale=SIGMA)
    return torch.tensor(result)
# print("NP bound: ", np_upperbound(randoms, pert_radi, sigma))

def dkw_upperbound(randoms, SIGMA, radi, alpha=0.05, num_s=1000, n_classes=1, evasion=True):
    bon_alpha = alpha / n_classes
    # error = np.sqrt(np.log((1 if evasion else 2) / bon_alpha) / (2 * randoms.shape[0]))
    error = 0
    s_min = 0
    s_max = 1
    s_seg = torch.linspace(s_min, s_max, num_s + 1)

    empi_cdf = torch.minimum(
        ((randoms.view(-1, 1) > s_seg.to(randoms.device)).sum(dim=0) / randoms.shape[0]) + error,
        torch.tensor([1.0]).to(randoms.device))

    result = (norm.cdf(norm.ppf(empi_cdf.cpu(), scale=SIGMA) + radi, scale=SIGMA) * (1 / (num_s))).sum() + (1/num_s)
    return torch.tensor(result)
# print("DKW bound: ", dkw_upperbound(randoms, pert_radi, sigma))


def dkw_lowerbound(randoms, SIGMA, radi, alpha=0.05, num_s=1000, n_classes=1, evasion=True):
    bon_alpha = alpha / n_classes
    # error = np.sqrt(np.log((1 if evasion else 2) / bon_alpha) / (2 * randoms.shape[0]))
    error = 0
    s_min = 0
    s_max = 1
    s_seg = torch.linspace(s_min, s_max, num_s + 1)

    empi_cdf = torch.maximum(
        ((randoms.view(-1, 1) > s_seg.to(randoms.device)).sum(dim=0) / randoms.shape[0]) - error,
        torch.tensor([0.0]).to(randoms.device))

    result = (norm.cdf(norm.ppf(empi_cdf.cpu(), scale=SIGMA) - radi, scale=SIGMA) * (1 / (num_s))).sum()
    return torch.tensor(result)


def np_upperbound_tensor(scores_samplings, SIGMA, radi, alpha=0.05, n_classes=1):
    bon_alpha = alpha / n_classes
    # error = np.sqrt(np.log(1 / bon_alpha) / (2 * scores_samplings.shape[-1]))
    error = 0
    p_uppers = torch.minimum(scores_samplings.mean(dim=-1) + error, torch.tensor(1.0).to(scores_samplings.device))
    result = norm.cdf(
        norm.ppf(p_uppers.cpu(), scale=SIGMA) + radi,
        scale=SIGMA)
    return torch.tensor(result).to(scores_samplings.device)

def dkw_upperbound_tensor(scores_sampling, SIGMA, radi, alpha=0.05, num_s=10000, n_classes=1):
    return torch.stack([
        torch.stack([
            dkw_upperbound(scores_sampling[d, c, :], SIGMA=SIGMA, radi=radi, alpha=alpha, num_s=num_s, n_classes=n_classes)
            for c in range(scores_sampling.shape[1])
        ])
        for d in range(scores_sampling.shape[0])
    ]).to(scores_sampling.device)

def dkw_lowerbound_tensor(scores_sampling, SIGMA, radi, alpha=0.05, num_s=10000, n_classes=1):
    return torch.stack([
        torch.stack([
            dkw_lowerbound(scores_sampling[d, c, :], SIGMA=SIGMA, radi=radi, alpha=alpha, num_s=num_s, n_classes=n_classes)
            for c in range(scores_sampling.shape[1])
        ])
        for d in range(scores_sampling.shape[0])
    ]).to(scores_sampling.device)


def np_bounds_tensor(scores_samplings, SIGMA, radi, alpha=0.05, n_classes=1):
    bon_alpha = alpha / n_classes
    # error = np.sqrt(np.log(1 / bon_alpha) / (2 * scores_samplings.shape[-1]))
    error = 0
    p_uppers = torch.minimum(scores_samplings.mean(dim=-1) + error, torch.tensor(1.0).to(scores_samplings.device))
    p_lowers = torch.maximum(scores_samplings.mean(dim=-1) - error, torch.tensor(0.0).to(scores_samplings.device))

    upper_result = norm.cdf(
        norm.ppf(p_uppers.cpu(), scale=SIGMA) + radi,
        scale=SIGMA)
    lower_result = norm.cdf(
        norm.ppf(p_lowers.cpu(), scale=SIGMA) - radi,
        scale=SIGMA)

    return torch.tensor(lower_result).to(scores_samplings.device), torch.tensor(upper_result).to(scores_samplings.device)


def get_cal_mask(vals_tensor, fraction=0.1):
    perm = torch.randperm(vals_tensor.shape[0])
    mask = torch.zeros((vals_tensor.shape[0]), dtype=bool)
    cutoff_index = int(vals_tensor.shape[0] * fraction)
    mask[perm[:cutoff_index]] = True
    return mask

def singleton_hit(pred_set, y_true):
    return ((pred_set[y_true])[pred_set.sum(axis=1) == 1].sum() / (pred_set).shape[0]).item()

