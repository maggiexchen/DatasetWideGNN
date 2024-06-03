import pandas as pd
# import uproot
import numpy
# import h5py
import json
# import math
# import random
# import yaml
import glob
import re
import os
os.environ["NUMEXPR_MAX_THREADS"] = "16"
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
import mplhep as hep
import argparse
import gc

import utils.normalisation as norm
import utils.torch_distances as dis
import utils.adj_mat as adj
import utils.misc as misc
import utils.performance as perf
import utils.plotting as plotting

import numpy as np
import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch.utils.data import DataLoader, TensorDataset
# from torchinfo import summary

import time
st = time.time()
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, auc
from sklearn.utils import shuffle

import shap
import logging
logging.getLogger().setLevel(logging.INFO)

torch.cuda.empty_cache()

def GetParser():
    """Argument parser for reading Ntuples script."""
    parser = argparse.ArgumentParser(
        description="Reading Ntuples command line options."
    )
    parser.add_argument(
        "--MLconfig",
        "-c",
        type=str,
        required=True,
        help="Specify the config file for training",
    )

    parser.add_argument(
        "--userconfig",
        "-u",
        type=str,
        required=True,
        help="Specify the config for the user e.g. paths to store all the input/output data and results, signal model to look at",
    )

    return parser

parser = GetParser()
args = parser.parse_args()

print("CUDA is available? ", torch.cuda.is_available())  # Outputs True if GPU is available
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(torch.cuda.mem_get_info())

train_config_path = args.MLconfig
train_config = misc.load_config(train_config_path)

user_config_path = args.userconfig
user_config = misc.load_config(user_config_path)
print(user_config)
h5_path = user_config["h5_path"]
plot_path = user_config["plot_path"]
ll_path = user_config["ll_path"]
adj_path = user_config["adj_path"]
dist_path = user_config["dist_path"]

# TODO: assert. This should be "hhh" "LQ" or "stau"
signal = user_config["signal"]

training_name = train_config["name"]

variable = train_config["variable"]
if variable is None:
    print("Need to specify a type of kinematic variable in the config")
distance = train_config["distance"]
if distance is None:
    print("Need to specify a type of distance metric for the adjacency matrix in the config")
eff = train_config["sigsig_eff"]
if eff is None:
    print("Need to specify a sig-sig efficiency for the adjacency matrix when training a gcn in the config")
elif eff not in [0.6, 0.7, 0.8, 0.9]:
    raise Exception("not given a supported efficiency, (0.6, 0.7, 0.8, 0.9)")

kinematics = misc.get_kinematics(variable)
input_size = len(kinematics)

logging.info("signal: "+signal)
logging.info("variable set: "+variable)
logging.info("input data path: "+h5_path)
logging.info("input ll json path: "+ll_path)
logging.info("input distances path: "+dist_path)
logging.info("output plot path: "+plot_path)
logging.info("adj matrix storage path: "+adj_path)

logging.info("distance metric: "+distance)
logging.info("desired efficieny: "+str(eff))

# load training data file and kinematics
logging.info('Importing signal and background files...')

# un-normalised signal and background kinematics (for checking and plotting)
# raw_train_sig, raw_train_bkg, _, _, _, _ = adj.data_loader(h5_path, plot_path, "train", kinematics, norm_kin=False, signal=signal)
# raw_val_sig, raw_val_bkg, _, _, _, _ = adj.data_loader(h5_path, plot_path, "val", kinematics, norm_kin=False, signal=signal)

# normalised signal and background kinematics
train_sig, train_bkg, train_x, train_sig_wgts, train_bkg_wgts, _, _  = adj.data_loader(h5_path, plot_path, "train", kinematics, norm_kin=True, signal=signal)
val_sig, val_bkg, val_x, val_sig_wgts, val_bkg_wgts, _, _ = adj.data_loader(h5_path, plot_path, "val", kinematics, norm_kin=True, signal=signal)

full_sig = torch.cat((train_sig, val_sig), dim=0)
full_bkg = torch.cat((train_bkg, val_bkg), dim=0)

# raw_full_sig = torch.cat((raw_train_sig, raw_val_sig), dim=0)
# raw_full_bkg = torch.cat((raw_train_bkg, raw_val_bkg), dim=0)

full_x = torch.cat((full_sig, full_bkg), dim=0).to(device)
full_wgts = torch.cat((torch.cat((train_sig_wgts, val_sig_wgts), dim=0), torch.cat((train_bkg_wgts, val_bkg_wgts), dim=0)), dim=0)#.cuda()

print("numevents: ",full_x.size(0))

# read in linking length calculated from sampled training data
sigsig_eff = eff
ll_path = ll_path+str(variable)+"_"+str(distance)+"_linking_length.json"
print(ll_path)
with open(ll_path, 'r') as lfile:
    length_dict = json.load(lfile)
    lengths = length_dict["length"]
    linking_length = lengths[length_dict["sigsig_eff"].index(sigsig_eff)]
    logging.info("linking length ="+str(linking_length))
# linking_length = 0.1

# TODO: batch load in event distances to apply linking length to
# If the distances were to be calculated and stored in advance, then loaded here, the ordering of the events need to be the same!
logging.info("Batch applying the linking length and getting non-zero indices ...")
logging.info("For sigsig ...")
sigsig_ind = adj.generate_batched_nonzero_ind(dist_path, variable, distance, "sigsig", linking_length, flip=True)
print("sigsig: ",sigsig_ind.shape, sigsig_ind)
logging.info("For sigbkg ...")
sigbkg_ind = adj.generate_batched_nonzero_ind(dist_path, variable, distance, "sigbkg", linking_length, flip=True)
print("sigbg: ", sigbkg_ind.shape, sigbkg_ind)
logging.info("For bkgsig ...")
bkgsig_ind = torch.clone(sigbkg_ind)
# pdb.set_trace()
bkgsig_ind = bkgsig_ind[:, [1, 0]]
print("bgsig: ", bkgsig_ind.shape, bkgsig_ind)
logging.info("For bkgbkg ...")
bkgbkg_ind = adj.generate_batched_nonzero_ind(dist_path, variable, distance, "bkgbkg", linking_length, flip=True)
print("bgbg: ", bkgbkg_ind.shape, bkgbkg_ind)

# adding to the indices to form the full matrix indices
logging.info("Stitching together the non-zero indices ...")
sigbkg_ind[:,1]+=len(full_sig)
bkgsig_ind[:,0]+=len(full_sig)
bkgbkg_ind += len(full_sig)
logging.info("Generating sparse adjacency matrix ...")
sparse_adj_mat, edge_ind, crow_ind, col_ind, values = adj.generate_sparse_adj_mat(sigsig_ind, sigbkg_ind, bkgsig_ind, bkgbkg_ind, len(full_sig)+len(full_bkg))

print("sparse adj mat: ", sparse_adj_mat)
total_edges = sigsig_ind.shape[0]+sigbkg_ind.shape[0]+bkgbkg_ind.shape[0]
total_pairs = (len(full_sig)+len(full_bkg))**2
print("The fraction of edges in graph is ", total_edges / total_pairs)

misc.print_mem_info()

# saving adjacency matrix
# Save the sparse tensor to a .pt file
logging.info("Saving sparse adjacency matrix ...")
misc.create_dirs(adj_path)
torch.save(sparse_adj_mat, adj_path+'sparse_adjacency_matrix.pt')
torch.save(edge_ind, adj_path+'coo_row.pt')
torch.save(crow_ind, adj_path+'csr_row.pt')
torch.save(col_ind, adj_path+'csr_col.pt')
torch.save(values, adj_path+'csr_values.pt')

