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
feature_h5_path = user_config["feature_h5_path"]
kinematic_h5_path = user_config["kinematic_h5_path"]
plot_path = user_config["plot_path"]
ll_path = user_config["ll_path"]
adj_path = user_config["adj_path"]
dist_path = user_config["dist_path"]
flip = train_config["flip"]
bool_edge_wgt = train_config["edge_weights"]
os.makedirs(adj_path, exist_ok=True)
# TODO: assert. This should be "hhh" "LQ" or "stau"
signal = user_config["signal"]
signal_mass = user_config["signal_mass"]
feature_dim = user_config["feature_dim"]
assert signal in ["hhh", "LQ", "stau"], f"Invalid signal type: {signal}"
### rename signal to include mass
signal = signal + "_" + str(signal_mass)


kinematic_variable = train_config["kinematic_variable"]
embedding_variable = train_config["embedding_variable"]
if kinematic_variable is None:
    print("Need to specify a type of kinematic variable in the config")

if embedding_variable is None:
    embedding_variable = kinematic_variable

distance = train_config["distance"]
if distance is None:
    print("Need to specify a type of distance metric for the adjacency matrix in the config")

edge_frac = train_config["edge_frac"]
if edge_frac is None:
    print("Need to specify a desired edge_fraction for the adjacency matrix when training a gcn in the config")
elif edge_frac not in [0.1, 0.2, 0.3, 0.4, 0.5]:
    raise Exception("not given a supported edge fraction, (0.1, 0.2, 0.3, 0.4, 0.5)")

linking_length = train_config["linking_length"]
os.makedirs(ll_path, exist_ok=True)
if linking_length is None:
    ll_path = ll_path+str(embedding_variable)+"_"+str(distance)+"_linking_length.json"
    print("Saving linking length to ", ll_path)
    with open(ll_path, 'r') as lfile:
        length_dict = json.load(lfile)
        lengths = length_dict["length"]
        linking_length = lengths[length_dict["edge_frac"].index(edge_frac)]
        logging.info("linking length ="+str(linking_length))
else:
    print("linking length is given in config, IGNORING the edge fraction in the config!")

if edge_frac is None:
    adj_path = adj_path + "/" + str(distance) + "_" + f"linking_length_{linking_length}/"
else:
    adj_path = adj_path + "/" + str(distance) + "_" + f"edge_frac_{edge_frac}/"
misc.create_dirs(adj_path)

kinematics, kinematic_labels = misc.get_kinematics(kinematic_variable, feature_dim)
input_size = len(kinematics)

logging.info("signal: "+signal)
logging.info("kinematic variable set: "+kinematic_variable)
logging.info("embedding variable set: "+embedding_variable)
logging.info("input data distance path: "+feature_h5_path)
logging.info("input data kinematic path: "+kinematic_h5_path)
logging.info("input ll json path: "+ll_path)
logging.info("input distances path: "+dist_path)
logging.info("output plot path: "+plot_path)
logging.info("adj matrix storage path: "+adj_path)
logging.info("distance metric: "+distance)
if edge_frac is not None:
    logging.info("desired edge fraction: "+str(edge_frac))
elif linking_length is not None:
    logging.info("linking length: "+str(linking_length))

# load training data file and kinematics
logging.info('Importing signal and background files...')

# normalised signal and background kinematics
logging.info('Importing signal and background files...')
full_sig, full_bkg, full_x, sig_wgt, bkg_wgt, sig_labels, bkg_labels = adj.data_loader(feature_h5_path, plot_path, kinematics, kinematic_labels, ex="", plot=False, signal=signal, signal_mass=signal_mass, standardisation=True)
full_event_weights = torch.cat((sig_wgt, bkg_wgt))
print("Number of total events: ",full_x.size(0))
del sig_wgt, bkg_wgt, full_x

### load distances and apply linking length to receieve indices
logging.info("Batch applying the linking length and getting non-zero indices ...")
logging.info("For sigsig ...")
sigsig_result = adj.generate_batched_nonzero_ind(dist_path, embedding_variable, distance, "sigsig", linking_length, flip=flip, edge_wgt=bool_edge_wgt)
if bool_edge_wgt:
    sigsig_ind, sigsig_edge_wgts = sigsig_result
else:
    sigsig_ind = sigsig_result
print("sigsig: ",sigsig_ind.shape)
print("fraction of egdes in sigsig: ", sigsig_ind.shape[0]/(len(full_sig)**2))

logging.info("For sigbkg ...")
sigbkg_result = adj.generate_batched_nonzero_ind(dist_path, embedding_variable, distance, "sigbkg", linking_length, flip=flip, edge_wgt=bool_edge_wgt)
if bool_edge_wgt:
    sigbkg_ind, sigbkg_edge_wgts = sigbkg_result
else:
    sigbkg_ind = sigbkg_result

print("sigbg: ", sigbkg_ind.shape)
print("fraction of egdes in sigbkg: ", sigbkg_ind.shape[0]/(len(full_sig)*len(full_bkg)))

logging.info("For bkgsig ...")
bkgsig_ind = torch.clone(sigbkg_ind)
bkgsig_ind = bkgsig_ind[:, [1, 0]]
if bool_edge_wgt:
    bkgsig_edge_wgts = torch.clone(sigbkg_edge_wgts)
print("bgsig: ", bkgsig_ind.shape)
print("fraction of egdes in bkgsig: ", bkgsig_ind.shape[0]/(len(full_bkg)*len(full_sig)))

logging.info("For bkgbkg ...")
bkgbkg_result = adj.generate_batched_nonzero_ind(dist_path, embedding_variable, distance, "bkgbkg", linking_length, flip=flip, edge_wgt=bool_edge_wgt)
if bool_edge_wgt:
    bkgbkg_ind, bkgbkg_edge_wgts = bkgbkg_result
else:
    bkgbkg_ind = bkgbkg_result
print("bgbg: ", bkgbkg_ind.shape)

# adding to the indices to form the full matrix indices
logging.info("Stitching together the non-zero indices ...")
sigbkg_ind[:,1]+=len(full_sig)
bkgsig_ind[:,0]+=len(full_sig)
bkgbkg_ind += len(full_sig)

logging.info("Concatenating the indices ...")
full_ind = torch.cat((sigsig_ind, sigbkg_ind, bkgsig_ind, bkgbkg_ind)).round().to(torch.int32)
if bool_edge_wgt:
    full_edge_wgts = torch.cat((sigsig_edge_wgts, sigbkg_edge_wgts, bkgsig_edge_wgts, bkgbkg_edge_wgts)).to(torch.float32)

    ### define indices for plotting random subsets of the edge weights
    sigsig_plot_ind = torch.randperm(len(sigsig_edge_wgts))[:2000]
    sigbkg_plot_ind = torch.randperm(len(sigbkg_edge_wgts))[:1000]
    bkgsig_plot_ind = torch.randperm(len(bkgsig_edge_wgts))[:1000]
    bkgbkg_plot_ind = torch.randperm(len(bkgbkg_edge_wgts))[:2000]

    ### plot the edge weights before minmax normalisation (1/d)
    fig, ax = plt.subplots()
    _, binning, _ = ax.hist(torch.cat((sigbkg_edge_wgts[sigbkg_plot_ind], bkgsig_edge_wgts[bkgsig_plot_ind])) , bins=70, color="darkorange", alpha=0.5, label="sig-bkg")
    ax.hist(sigsig_edge_wgts[sigsig_plot_ind], bins=binning, color="steelblue", alpha=0.5, label="sig-sig")
    ax.hist(torch.cat((sigbkg_edge_wgts[sigbkg_plot_ind], bkgsig_edge_wgts[bkgsig_plot_ind])) , bins=binning, color="darkorange", alpha=0.5, label="sig-bkg")
    ax.hist(bkgbkg_edge_wgts[bkgbkg_plot_ind], bins=binning, color="forestgreen", alpha=0.5, label="bkg-bkg")
    ax.set_xlabel("Edge weights (excluding event weights)", loc="right")
    ax.set_ylabel("Edges / Bin", loc="top")
    ax.legend(loc="upper right")
    ax.set_yscale("log")
    fig.tight_layout()
    fig.savefig(adj_path+"Unnormed_edge_wgts_no_EventWeights.pdf", transparent=True)

    ### min_max normalise the edge weights
    inf_mask = torch.isinf(full_edge_wgts)
    max_wgt = torch.max(full_edge_wgts[~inf_mask])
    min_wgt = torch.min(full_edge_wgts[~inf_mask])
    full_edge_wgts = norm.minmax(full_edge_wgts, min_wgt, max_wgt)
    full_edge_wgts[inf_mask] = 1

    ### plot the edge weights after minmax normalisation (1/d)
    fig, ax = plt.subplots()
    binning = numpy.linspace(0, 1, 70)
    ax.hist(full_edge_wgts[sigsig_plot_ind], bins=binning, color="steelblue", alpha=0.5, label="sig-sig")
    ax.hist(torch.cat((full_edge_wgts[sigbkg_plot_ind+len(sigsig_edge_wgts)], full_edge_wgts[bkgsig_plot_ind+len(sigsig_edge_wgts)+len(sigbkg_edge_wgts)])) , bins=binning, color="darkorange", alpha=0.5, label="sig-bkg")
    ax.hist(full_edge_wgts[bkgbkg_plot_ind+len(sigsig_edge_wgts)+len(sigbkg_edge_wgts)+len(bkgsig_edge_wgts)], bins=binning, color="forestgreen", alpha=0.5, label="bkg-bkg")
    ax.set_xlabel("Edge weights (including event weights)", loc="right")
    ax.set_ylabel("Edges / Bin", loc="top")
    ax.legend(loc="upper right")
    ax.set_yscale("log")
    fig.tight_layout()
    fig.savefig(adj_path+"Normed_edge_wgts_no_EventWeights.pdf", transparent=True)

    ### we are only combining the MC events later, in torch_train.py, depending on the gnn model type
    # fig, ax = plt.subplots()
    # _, binning, _ = ax.hist(normed_full_edge_wgts[bkgbkg_plot_ind+len(sigsig_edge_wgts)+len(sigbkg_edge_wgts)+len(bkgsig_edge_wgts)], bins=70, color="forestgreen", alpha=0.5, label="bkg-bkg")
    # ax.hist(torch.cat((normed_full_edge_wgts[sigbkg_plot_ind+len(sigsig_edge_wgts)], normed_full_edge_wgts[bkgsig_plot_ind+len(sigsig_edge_wgts)+len(sigbkg_edge_wgts)])) , bins=binning, color="darkorange", alpha=0.5, label="sig-bkg")
    # ax.hist(normed_full_edge_wgts[sigsig_plot_ind], bins=binning, color="steelblue", alpha=0.5, label="sig-sig")
    # ax.set_xlabel("Edge weights (including event weights)", loc="right")
    # ax.set_ylabel("Edges / Bin", loc="top")
    # ax.legend(loc="upper right")
    # ax.set_yscale("log")
    # fig.tight_layout()
    # fig.savefig(adj_path+"Normed_edge_wgts_with_EventWeights.pdf", transparent=True)

    del sigsig_edge_wgts, sigbkg_edge_wgts, bkgsig_edge_wgts, bkgbkg_edge_wgts

total_edges = sigsig_ind.shape[0]+sigbkg_ind.shape[0]+bkgbkg_ind.shape[0]
total_pairs = (len(full_sig)+len(full_bkg))**2
print("Linking length at edge fraction ", edge_frac)
print("The fraction of edges in graph is ", total_edges / total_pairs)
del sigsig_ind, sigbkg_ind, bkgsig_ind, bkgbkg_ind

misc.print_mem_info()
logging.info("Saving sparse adjacency matrix ...")

### saving the adjaceny matrix indices as edge indices
torch.save(full_ind[:,0], adj_path+'row_ind.pt')
torch.save(full_ind[:,1], adj_path+'col_ind.pt')
if bool_edge_wgt:
    torch.save(full_edge_wgts, adj_path+'edge_wgts.pt')



