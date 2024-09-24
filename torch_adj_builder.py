import pandas as pd
import numpy
import json

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
distance_h5_path = user_config["distance_h5_path"]
kinematic_h5_path = user_config["kinematic_h5_path"]
plot_path = user_config["plot_path"]
ll_path = user_config["ll_path"]
adj_path = user_config["adj_path"]
dist_path = user_config["dist_path"]

# TODO: assert. This should be "hhh" "LQ" or "stau"
signal = user_config["signal"]
feature_dim = user_config["feature_dim"]
assert signal in ["hhh", "LQ", "stau"], f"Invalid signal type: {signal}"

kinematic_variable = train_config["kinematic_variable"]
embedding_variable = train_config["embedding_variable"]
if kinematic_variable is None:
    print("Need to specify a type of kinematic variable in the config")

if embedding_variable is None:
    embedding_variable = kinematic_variable

distance = train_config["distance"]
if distance is None:
    print("Need to specify a type of distance metric for the adjacency matrix in the config")

eff = train_config["sigsig_eff"]
if eff is None:
    print("Need to specify a sig-sig efficiency for the adjacency matrix when training a gcn in the config")
elif eff not in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    raise Exception("not given a supported efficiency, (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)")

linking_length = train_config["linking_length"]
# read in linking length calculated from sampled training data, if not given in config
sigsig_eff = None
if linking_length is None:
    sigsig_eff = eff
    ll_path = ll_path+str(embedding_variable)+"_"+str(distance)+"_linking_length.json"
    print(ll_path)
    with open(ll_path, 'r') as lfile:
        length_dict = json.load(lfile)
        lengths = length_dict["length"]
        linking_length = lengths[length_dict["sigsig_eff"].index(sigsig_eff)]
        logging.info("linking length ="+str(linking_length))
else:
    print("linking length is given in config, IGNORING the sigsig_eff in the config!")


kinematics = misc.get_kinematics(kinematic_variable, feature_dim)
input_size = len(kinematics)

logging.info("signal: "+signal)
logging.info("kinematic variable set: "+kinematic_variable)
logging.info("embedding variable set: "+embedding_variable)
logging.info("input data distance path: "+distance_h5_path)
logging.info("input data kinematic path: "+kinematic_h5_path)
logging.info("input ll json path: "+ll_path)
logging.info("input distances path: "+dist_path)
logging.info("output plot path: "+plot_path)
logging.info("adj matrix storage path: "+adj_path)
logging.info("distance metric: "+distance)
if sigsig_eff is not None:
    logging.info("desired efficieny: "+str(sigsig_eff))
elif linking_length is not None:
    logging.info("linking length: "+str(linking_length))

# load training data file and kinematics
logging.info('Importing signal and background files...')

# normalised signal and background kinematics
train_sig, train_bkg, train_x, train_sig_wgts, train_bkg_wgts, _, _  = adj.data_loader(kinematic_h5_path, plot_path, "train", kinematics, plot=True, signal=signal)
val_sig, val_bkg, val_x, val_sig_wgts, val_bkg_wgts, _, _ = adj.data_loader(kinematic_h5_path, plot_path, "val", kinematics, plot=False, signal=signal)
test_sig, test_bkg, test_x, test_sig_wgts, test_bkg_wgts, _, _ = adj.data_loader(kinematic_h5_path, plot_path, "test", kinematics, plot=False, signal=signal)

full_sig = torch.cat((train_sig, val_sig, test_sig), dim=0)
full_bkg = torch.cat((train_bkg, val_bkg, test_bkg), dim=0)

full_x = torch.cat((full_sig, full_bkg), dim=0).to(device)
full_wgts = torch.cat((torch.cat((train_sig_wgts, val_sig_wgts, test_sig_wgts), dim=0), torch.cat((train_bkg_wgts, val_bkg_wgts, test_bkg_wgts), dim=0)), dim=0)#.cuda()

print("numevents: ",full_x.size(0))

### load distances and apply linking length to receieve indices
logging.info("Batch applying the linking length and getting non-zero indices ...")
logging.info("For sigsig ...")
sigsig_ind = adj.generate_batched_nonzero_ind(dist_path, embedding_variable, distance, "sigsig", linking_length, flip=True)
print("sigsig: ",sigsig_ind.shape)

logging.info("For sigbkg ...")
sigbkg_ind = adj.generate_batched_nonzero_ind(dist_path, embedding_variable, distance, "sigbkg", linking_length, flip=True)
print("sigbg: ", sigbkg_ind.shape)

logging.info("For bkgsig ...")
bkgsig_ind = torch.clone(sigbkg_ind)
bkgsig_ind = bkgsig_ind[:, [1, 0]]
print("bgsig: ", bkgsig_ind.shape)

logging.info("For bkgbkg ...")
bkgbkg_ind = adj.generate_batched_nonzero_ind(dist_path, embedding_variable, distance, "bkgbkg", linking_length, flip=True)
print("bgbg: ", bkgbkg_ind.shape)

# adding to the indices to form the full matrix indices
logging.info("Stitching together the non-zero indices ...")
sigbkg_ind[:,1]+=len(full_sig)
bkgsig_ind[:,0]+=len(full_sig)
bkgbkg_ind += len(full_sig)

logging.info("Concatenating the indices ...")
full_ind = torch.cat((sigsig_ind, sigbkg_ind, bkgsig_ind, bkgbkg_ind)).round().to(torch.int32)

#### generate the adjacency matrix as a sparse tensor object (currently not needed, using edge index instead)
# logging.info("Rounding the indices to int32 ...")
# full_ind = full_ind.to(torch.int32)
# logging.info("Generating sparse adjacency matrix ...")
# sparse_adj_mat, edge_ind, crow_ind, col_ind, values = adj.generate_sparse_adj_mat(sigsig_ind, sigbkg_ind, bkgsig_ind, bkgbkg_ind, len(full_sig)+len(full_bkg))
# print("sparse adj mat: ", sparse_adj_mat.shape)


total_edges = sigsig_ind.shape[0]+sigbkg_ind.shape[0]+bkgbkg_ind.shape[0]
total_pairs = (len(full_sig)+len(full_bkg))**2
print("Linking length at sig-sig efficiency ", sigsig_eff)
print("The fraction of edges in graph is ", total_edges / total_pairs)
del sigsig_ind, sigbkg_ind, bkgsig_ind, bkgbkg_ind

misc.print_mem_info()
logging.info("Saving sparse adjacency matrix ...")

if sigsig_eff is None:
    adj_path = adj_path + "/" + f"linking_length_{linking_length}/"
else:
    adj_path = adj_path + "/" + f"sigsig_eff_{sigsig_eff}/"
misc.create_dirs(adj_path)

#### saving the indices and sparse tensor object
# torch.save(sparse_adj_mat, adj_path+'sparse_adjacency_matrix.pt')
# torch.save(edge_ind, adj_path+'coo_row.pt')
# torch.save(crow_ind, adj_path+'csr_row.pt')
# torch.save(col_ind, adj_path+'csr_col.pt')
# torch.save(values, adj_path+'csr_values.pt')

### saving the adjaceny matrix indices as edge indices
torch.save(full_ind[:,0], adj_path+'row_ind.pt')
torch.save(full_ind[:,1], adj_path+'col_ind.pt')


