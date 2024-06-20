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

from datasets import  DATASETS, get_dataset
from architectures import get_architecture

import matplotlib.pyplot as plt

from scipy.stats import norm

from utils import ModelManager
from utils import standard_l2_norm
from tqdm import tqdm
from discrete_data.certify_utils import *
import wandb

import cp.transformations as cp_t
import cp.graph_transformations as cp_gt
from cp.graph_cp import GraphCP
import pickle

def save_pkl(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
def load_pkl(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def adversarial_dataset(sigma, path):
    imgs = torch.load(f"{path}/radi_{sigma}/pgd_images.pth")
    labels = torch.load(f"{path}/labels.pth")
    return torch.utils.data.TensorDataset(imgs, labels)
def get_parser():
    parser = argparse.ArgumentParser(description='Efficient yet Robust CPS')
    
    # dataset
    parser.add_argument("--dataset", choices=DATASETS, help="which dataset")
    parser.add_argument("--split", choices=["train", "test"], default="test", help="train or test set")

    parser.add_argument("--skip", type=int, default=None, help="how many examples to skip")
    parser.add_argument("--batch_size", type=int, default=1000, help="batch size")

    # model
    parser.add_argument("--checkpoint", type=str, help="path to saved pytorch model of base classifier")

    
    
    #CP & smoohting
    parser.add_argument("--n_samples", type=int, default=10000, help="number of samples to use")    
    parser.add_argument("--alpha", type=float, default=0.05, help="failure probability")
    # parser.add_argument("--converage_guarantee", type=float, default=0.9, help="coverage guarantee")
    parser.add_argument("--smoothing_sigma", type=float, default=0.05, help="smoothing sigma")

    parser.add_argument("--n_iters", type=int, default=100, help="number of iterations")

    parser.add_argument("--fraction", type=float, default=0.2, help="fraction of data to use for calibration")
    
    # wandb
    parser.add_argument("--wandb", action="store_true", help="use wandb")
    parser.add_argument("--wandb_project", type=str, default="efficient-robustness", help="wandb project name")

    parser.add_argument("--debug", action="store_true", help="debug mode")

    parser.add_argument("--adversarial", action="store_true", help="use adversarial examples")
    parser.add_argument("--adversarial_sigma", type=float, default= 0.125, choices=[0.0625, 0.125, 0.1875, 0.25], help="adversarial sigma")

    parser.add_argument("--log_path", type=str, default="logs", help="path to save logs")

    args = parser.parse_args()
    return args

path_to_adv_example = ""


if __name__ == "__main__":

    args = get_parser()
    num_classes = 10 if args.dataset == "cifar10" else 1000

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"device = {device}")

    if args.wandb:
        wandb.init(project=args.wandb_project, config=args)
        wandb.config.update(args)


    # load model
    try:
        checkpoint = torch.load(args.checkpoint)
    except:
        raise('No checkpoint found')
    args.model = checkpoint["arch"]
    if args.wandb:
        wandb.config.update(args)
        
    model = get_architecture(checkpoint["arch"], args.dataset)
    model.load_state_dict(checkpoint['state_dict'])
    
    # create smoothed classifier
    smooth_model = ModelManager(model=model, device=device)

    # load dataset
    if args.adversarial:
        print("using adversarial examples")
        if args.dataset == "cifar10":
            dataset = adversarial_dataset(args.adversarial_sigma, path= path_to_adv_example)
        else:
            raise NotImplementedError
    else:
        dataset = get_dataset(args.dataset, args.split)

    print(f"dataset size = {len(dataset)}")
    # subset_indices = list(range(len(dataset)))
    if args.skip is not None:
        subset_indices = list(range(0, len(dataset), args.skip))
        dataset = Subset(dataset, subset_indices)
    print(f"dataset size = {len(dataset)}")

    dataset = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True)


    y_pred, logits, y_true = smooth_model.smooth_predict(dataset, 
                                                         n_samples=args.n_samples, 
                                                         smoothing_function=lambda inputs: standard_l2_norm(inputs, args.smoothing_sigma))
    

    if args.wandb:
        if not os.path.exists(f"{args.log_path}/{args.wandb_project}/{wandb.run.name}"):
            os.makedirs(f"{args.log_path}/{args.wandb_project}/{wandb.run.name}")
        print('save')
        save_pkl((y_pred, logits, y_true), f"{args.log_path}/{args.wandb_project}/{wandb.run.name}/y_pred_logits_y_true.pkl")
   
   
    # compute accuracy
    acc = ((y_pred == y_true).sum() / y_true.shape[0]).item()
    print(f"acc = {acc}")
    if args.wandb:
        wandb.log({"acc": acc})
