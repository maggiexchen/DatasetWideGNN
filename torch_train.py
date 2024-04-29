import pandas as pd
# import uproot
import numpy
# import h5py
import json
# import math
# import random
# import yaml
import os
os.environ["NUMEXPR_MAX_THREADS"] = "16"
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
import mplhep as hep
import argparse

import utils.normalisation as norm
import utils.torch_distances as dis
import utils.adj_mat as adj
import utils.misc as misc
import utils.performance as perf
import utils.plotting as plotting
from utils.gcn_model import GCNClassifier
from utils.dnn_model import DNNClassifier

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import time
st = time.time()
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, auc
from sklearn.utils import shuffle

import shap
import logging
logging.getLogger().setLevel(logging.INFO)

def GetParser():
    """Argument parser for reading Ntuples script."""
    parser = argparse.ArgumentParser(
        description="Reading Ntuples command line options."
    )
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        required=True,
        help="Specify the config file for training",
    )

    parser.add_argument(
        "--path",
        "-p",
        type=str,
        required=False,
        help="Specify the path to store all the input/output data and results",
    )

    return parser

parser = GetParser()
args = parser.parse_args()

if args.path:
    path = args.path
    if path[-1]!="/": path += "/"
else:
    path = "/data/atlas/atlasdata3/maggiechen/gnn_project/" #maggies path
    # path = "/home/srutherford/GNN_shared/hhhgraph/data/" # sebs path

print("CUDA is available? ", torch.cuda.is_available())  # Outputs True if GPU is available

config_path = args.config
train_config = misc.load_config(config_path)
training_name = train_config["name"]
hidden_sizes_gcn = train_config["hidden_sizes_gcn"]
hidden_sizes_mlp = train_config["hidden_sizes_mlp"]
LR = train_config["LR"]
dropout_rates = train_config["dropout_rates"]
epochs = train_config["epochs"]

if len(hidden_sizes_gcn) == 0:
    model_label = "DNN"
    plot_path = "plots/DNN/"
else:
    model_label = "GCN"
    plot_path = "plots/GCN/"

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

train_loss = []
val_loss = []

logging.info("chosen model: "+model_label)
logging.info("variable set: "+variable)
logging.info("input/output path: "+path)

model = GCNClassifier(input_size=input_size, hidden_sizes_gcn=hidden_sizes_gcn, hidden_sizes_mlp = hidden_sizes_mlp, output_size=1, dropout_rates=dropout_rates)
logging.info("distance metric: "+distance)
logging.info("desired efficieny: "+str(eff))

# load training data file and kinematics
logging.info('Importing signal and background files...')

# un-normalised signal and background kinematics (for checking and plotting)
raw_train_sig, raw_train_bkg, _, _, _, _ = adj.data_loader("data", "train", kinematics, norm_kin=False)
raw_val_sig, raw_val_bkg, _, _, _, _ = adj.data_loader("data", "val", kinematics, norm_kin=False)

# normalised signal and background kinematics
train_sig, train_bkg, train_x, train_sig_wgts, train_bkg_wgts, train_truth_labels = adj.data_loader("data", "train", kinematics, norm_kin=True)
val_sig, val_bkg, val_x, val_sig_wgts, val_bkg_wgts, val_truth_labels = adj.data_loader("data", "val", kinematics, norm_kin=True)

full_sig = torch.cat((train_sig, val_sig), dim=0)
full_bkg = torch.cat((train_bkg, val_bkg), dim=0)

raw_full_sig = torch.cat((raw_train_sig, raw_val_sig), dim=0)
raw_full_bkg = torch.cat((raw_train_bkg, raw_val_bkg), dim=0)

full_x = torch.cat((full_sig, full_bkg), dim=0)
full_wgts = torch.cat((torch.cat((train_sig_wgts, val_sig_wgts), dim=0), torch.cat((train_bkg_wgts, val_bkg_wgts), dim=0)), dim=0)

if len(hidden_sizes_gcn) > 0:
    # read in linking length calculated from sampled training data
    sigsig_eff = eff
    ll_path = path+"linking_lengths/"+str(variable)+"_"+str(distance)+"_linking_length.json"
    misc.create_dirs(ll_path)

    with open(ll_path, 'r') as lfile:
        length_dict = json.load(lfile)
        lengths = length_dict["length"]
        linking_length = lengths[length_dict["sigsig_eff"].index(sigsig_eff)]
        logging.info("linking length ="+str(linking_length))
    
    # TODO: batch load in event distances to apply linking length to
    # If the distances were to be calculated and stored in advance, then loaded here, the ordering of the events need to be the same!
    logging.info("Batch loading in the distances ...")
    sigsig_distance, sigsig_wgt = misc.get_batched_distances(path, variable, distance, "sigsig", sample=False)
    sigbkg_distance, sigbkg_wgt = misc.get_batched_distances(path, variable, distance, "sigbkg", sample=False)
    bkgbkg_distance, bkgbkg_wgt = misc.get_batched_distances(path, variable, distance, "bkgbkg", sample=False)
    logging.info("Reshaping the flattened distances ...")
    print("full sig",len(full_sig))
    print("full bkg",len(full_bkg))
    print("sigsig", len(sigsig_distance))
    print("sigbkg", len(sigbkg_distance))
    print("bkgbkg", len(bkgbkg_distance))
    sigsig_distance = torch.reshape(sigsig_distance, (len(full_sig), len(full_sig)))
    sigbkg_distance = torch.reshape(sigbkg_distance, (len(full_sig), len(full_bkg)))
    bkgbkg_distance = torch.reshape(bkgbkg_distance, (len(full_bkg), len(full_bkg)))
    logging.info("Applying the linking length to get non-zero indices")
    sigsig_ind = (sigsig_distance <= linking_length).nonzero()
    sigbkg_ind = (sigbkg_distance <= linking_length).nonzero()
    bkgbkg_ind = (bkgbkg_distance <= linking_length).nonzero()
    bkgsig_ind = torch.clone(sigbkg_ind)
    sigbkg_ind[:, 1] += len(full_sig)
    bkgsig_ind[:, 0] += len(full_sig)
    bkgbkg_ind += len(full_sig)
    logging.info("Patching the indices of sigsig, sigbkg and bkgbkg distances together ...")
    ones_indices = torch.cat((sigsig_ind, sigbkg_ind, bkgsig_ind, bkgbkg_ind), dim=0)
    logging.info("Generating sparse adjacency matrix using indices ...")
    print(ones_indices.size())
    print(len(full_sig)+len(full_bkg))
    sparse_adj_mat = torch.sparse_coo_tensor(ones_indices.t(), torch.ones(ones_indices.size(0)), len(full_sig)+len(full_bkg))
    """
    logging.info("Generating adjacency matrix ...")
    full_adj_mat = adj.create_adj_mat(a, length)
    logging.info("Calculating distances and adjacency matrix ...")
    # full_adj_mat = adj.generate_adj_mat(full_x, full_wgts, distance, linking_length)

    edge_frac = torch.sum(full_adj_mat == 1).item() / len(full_adj_mat)**2
    print("Fraction of edges: ", edge_frac)

    # calculate centrality
    logging.info("Calculating and plotting centrality ...")
    deg_cent = torch.sum(full_adj_mat, dim=1)
    plotting.plot_centrality(deg_cent, full_sig, full_bkg, path+"plots", eff)

    adj_mat = full_adj_mat.to_sparse_csr() ### densor tensor to csr tensor
    norm_label = training_name ### pyg layer uses D_half_inv normalisation

    logging.info("Plotting convoluted kinematics ... ")
    plotting.plot_conv_kinematics(adj_mat, deg_cent, raw_full_sig, raw_full_bkg, kinematics, eff, path+"/training_kinematics/"+norm_label, normalisation="D_half_inv", standardise=False)
    plotting.plot_conv_conv_kinematics(adj_mat, deg_cent, raw_full_sig, raw_full_bkg, kinematics, eff, path+"/training_kinematics/"+norm_label, normalisation="D_half_inv", standardise=False)

    print("Normalised adjacency matrix\n", adj_mat)

else:
    adj_mat = None
    norm_label = ""


logging.info("Training ...")
# Define loss function for binary classification and ADAM optimiser
loss_function = nn.BCELoss()
optimiser = torch.optim.Adam(model.parameters(), lr=LR)

for epoch in range(epochs):
    model.train()
    optimiser.zero_grad()
    full_outputs = model(full_x, adj_mat)
    
    # splitting outputs into training/validation set
    # full x is concatenated as [train_sig : val_sig: train_bkg : val_bkg], so outputs need to be selected accordingly
    train_outputs = torch.cat((full_outputs[:len(train_sig)], full_outputs[(len(train_sig)+len(val_sig)):(len(train_sig)+len(val_sig)+len(train_bkg))]), dim=0)
    val_outputs = torch.cat((full_outputs[(len(train_sig)):(len(train_sig)+len(val_sig))], full_outputs[-len(val_bkg):]), dim=0)
    loss = loss_function(train_outputs.squeeze(), train_truth_labels.squeeze())
    loss.backward()
    train_loss.append(loss.item())
    optimiser.step()

    model.eval()
    validation_loss = loss_function(val_outputs.squeeze(), val_truth_labels.squeeze())
    val_loss.append(validation_loss.item())

    print(f'Epoch {epoch + 1}/{epochs}, Train Loss: {loss.item()}, Validation Loss: {validation_loss.item()}')

# save trained model
logging.info("Saving trained model and performance...")
model_path = path+"models/"+model_label+norm_label+"/"
misc.create_dirs(model_path)
model_file_name = "model.pth"
torch.save({
    'model_state': model.state_dict(),
    'optimiser_state': optimiser.state_dict(),
}, model_path+model_file_name)

train_outputs = train_outputs.view(-1)
train_label_bool = train_truth_labels.bool()
train_sig_pred = train_outputs[train_label_bool]
train_bkg_pred = train_outputs[torch.logical_not(train_label_bool)]

train_fpr, train_tpr, train_cut = roc_curve(train_truth_labels.detach().numpy(), train_outputs.detach().numpy())
train_auc = roc_auc_score(train_truth_labels.detach().numpy(), train_outputs.detach().numpy())
print("Training AUC", train_auc)

val_outputs = val_outputs.view(-1)
val_label_bool = val_truth_labels.bool()
val_sig_pred = val_outputs[val_label_bool]
val_bkg_pred = val_outputs[torch.logical_not(val_label_bool)]
fig, ax = plt.subplots()

val_fpr, val_tpr, val_cut = roc_curve(val_truth_labels.detach().numpy(), val_outputs.detach().numpy())
val_auc = roc_auc_score(val_truth_labels.detach().numpy(), val_outputs.detach().numpy())
print("Validation AUC", val_auc)

# save performance to json
perf.save_performance(train_loss, train_fpr, train_tpr, train_cut, train_auc, val_loss, val_fpr, val_tpr, val_cut, val_auc, model_path)
perf.save_metadata(len(train_sig), len(train_bkg), len(val_sig), len(val_bkg), hidden_sizes_gcn, hidden_sizes_mlp, LR, dropout_rates, epochs, model_path)

logging.info("Plotting training/validation losses ...")
fig, ax = plt.subplots()
x_epoch = numpy.arange(1,epochs+1,1)
ax.plot(x_epoch, train_loss, label="Training loss")
ax.plot(x_epoch, val_loss, label="Validation loss")
ax.legend(loc='upper right', fontsize=9)
ax.text(0.02, 0.95, model_label, verticalalignment="bottom", size=9, transform=ax.transAxes)
ax.set_xlabel("Epoch", loc="right")
ax.set_ylabel("Loss", loc="top")
fig_path = path + plot_path
misc.create_dirs(fig_path)
fig.savefig(fig_path+variable+"_"+model_label+norm_label+"_training_validation_loss.pdf", transparent=True)

logging.info("Plotting model outputs ...")
fig, ax = plt.subplots()
binning = numpy.linspace(0,1,50)
ax.hist(train_sig_pred.detach().numpy(), bins=binning, label="Signal (training)", histtype='step', linestyle='--', density=True, color="darkorange")
ax.hist(train_bkg_pred.detach().numpy(), bins=binning, label="Background (training)", histtype='step', linestyle='--', density=True, color="steelblue")
ax.hist(val_sig_pred.detach().numpy(), bins=binning, label="Signal (validation)", alpha=0.5, density=True, color="darkorange")
ax.hist(val_bkg_pred.detach().numpy(), bins=binning, label="Background (validation)", alpha=0.5, density=True, color="steelblue")
ax.text(0.02, 0.95, "Training AUC = {:.3f}".format(train_auc), verticalalignment="bottom", size=9, transform=ax.transAxes)
ax.text(0.02, 0.91, "Validation AUC = {:.3f}".format(val_auc), verticalalignment="bottom", size=9, transform=ax.transAxes)
ax.text(0.02, 0.87, "6b Resonant TRSM signal, 5b Data", verticalalignment="bottom", size=9, transform=ax.transAxes)
ax.text(0.02, 0.83, "Linking length at sig-sig efficiency "+str(eff), verticalalignment="bottom", size=9, transform=ax.transAxes)
ax.legend(loc='upper right', fontsize=9)
ax.set_xlabel("Output score", loc="right")
ax.set_ylabel("Normalised No. Events", loc="top")
ymin, ymax = ax.get_ylim()
ax.set_ylim((ymin, ymax*1.2))
fig_path = path + plot_path
misc.create_dirs(fig_path)
fig.savefig(fig_path+variable+"_"+model_label+norm_label+"_training_validation_pred.pdf", transparent=True)

logging.info("Plotting ROC curves ...")
fig, ax = plt.subplots()
plt.plot(train_fpr, train_tpr, label='Training ROC curve (AUC = {:.3f})'.format(train_auc))
plt.plot(val_fpr, val_tpr, label='Validation ROC curve (AUC = {:.3f})'.format(val_auc))
plt.legend(loc="upper left", fontsize=9)
plt.xlim(0,1)
plt.xlabel("Background Efficiency", loc="right")
plt.ylabel("Signal Efficiency", loc="top")
fig_path = path + plot_path
misc.create_dirs(fig_path)
fig.savefig(fig_path+variable+"_"+model_label+norm_label+"_training_validation_ROC.pdf", transparent=True)
"""