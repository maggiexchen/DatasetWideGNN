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
from utils.gcn_model import GCNClassifier
from utils.dnn_model import DNNClassifier

import numpy as np
import pdb
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

torch.cuda.empty_cache()

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
CUDA_LAUNCH_BLOCKING=1
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(torch.cuda.mem_get_info())
t = torch.cuda.get_device_properties(0).total_memory/(1024*1024*1024)
r = torch.cuda.memory_reserved(0)/(1024*1024*1024)
a = torch.cuda.memory_allocated(0)/(1024*1024*1024)
f = r-a  # free inside reserved
print("total: ", t, "reserved: ", r, "allocated:", a, "free: ", f)

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
model.to(device)
logging.info("distance metric: "+distance)
logging.info("desired efficieny: "+str(eff))

# load training data file and kinematics
logging.info('Importing signal and background files...')

# un-normalised signal and background kinematics (for checking and plotting)
raw_train_sig, raw_train_bkg, _, _, _, _ = adj.data_loader("data", "train", kinematics, norm_kin=False)
raw_val_sig, raw_val_bkg, _, _, _, _ = adj.data_loader("data", "val", kinematics, norm_kin=False)

# normalised signal and background kinematics
train_sig, train_bkg, train_x, train_sig_wgts, train_bkg_wgts, train_truth_labels  = adj.data_loader("data", "train", kinematics, norm_kin=True)
val_sig, val_bkg, val_x, val_sig_wgts, val_bkg_wgts, val_truth_labels = adj.data_loader("data", "val", kinematics, norm_kin=True)

full_sig = torch.cat((train_sig, val_sig), dim=0)
full_bkg = torch.cat((train_bkg, val_bkg), dim=0)[:30000]

raw_full_sig = torch.cat((raw_train_sig, raw_val_sig), dim=0)
raw_full_bkg = torch.cat((raw_train_bkg, raw_val_bkg), dim=0)

# full_x = torch.cat((full_sig, full_bkg), dim=0).cuda()
# full_wgts = torch.cat((torch.cat((train_sig_wgts, val_sig_wgts), dim=0), torch.cat((train_bkg_wgts, val_bkg_wgts), dim=0)), dim=0).cuda()
# train_truth_labels = torch.cat((train_truth_sig_labels, train_truth_bkg_labels))
# full_x = torch.cat((full_sig, full_bkg), dim=0)[:(len(full_sig)+30000)].cuda()
full_x = torch.cat((full_sig, full_bkg), dim=0).to(device)
#full_x = torch.cat((full_sig, full_bkg), dim=0)
full_wgts = torch.cat((torch.cat((train_sig_wgts, val_sig_wgts), dim=0), torch.cat((train_bkg_wgts, val_bkg_wgts), dim=0)), dim=0)#.cuda()
# full_wgts = torch.cat((torch.cat((train_sig_wgts, val_sig_wgts), dim=0), torch.cat((train_bkg_wgts, val_bkg_wgts), dim=0)), dim=0)[:(len(full_sig)+30000)].cuda()
# train_truth_labels = train_truth_labels[:20000]
# val_truth_labels = train_truth_labels[:(len(full_x)-20000)]

if len(hidden_sizes_gcn) > 0:
    # read in linking length calculated from sampled training data
    sigsig_eff = eff
    ll_path = path+"linking_lengths/"+str(variable)+"_"+str(distance)+"_linking_length.json"
    misc.create_dirs(ll_path)
    print(ll_path)
    with open(ll_path, 'r') as lfile:
        length_dict = json.load(lfile)
        lengths = length_dict["length"]
        linking_length = lengths[length_dict["sigsig_eff"].index(sigsig_eff)]
        logging.info("linking length ="+str(linking_length))
    linking_length = 0.5
    
    # TODO: batch load in event distances to apply linking length to
    # If the distances were to be calculated and stored in advance, then loaded here, the ordering of the events need to be the same!
    logging.info("Batch applying the linking length and getting non-zero indices ...")
    logging.info("For sigsig ...")
    sigsig_ind = adj.generate_batched_nonzero_ind(path, variable, distance, "sigsig", linking_length, flip=True)
    print(sigsig_ind.shape)
    logging.info("For sigbkg ...")
    sigbkg_ind = adj.generate_batched_nonzero_ind(path, variable, distance, "sigbkg", linking_length, flip=True)
    print(sigbkg_ind.shape)
    logging.info("For bkgsig ...")
    bkgsig_ind = torch.clone(sigbkg_ind)
    bkgsig_ind = bkgsig_ind[:, [1, 0]]
    print(bkgsig_ind.shape)
    logging.info("For bkgbkg ...")
    bkgbkg_ind = adj.generate_batched_nonzero_ind(path, variable, distance, "bkgbkg", linking_length, flip=True)
    print(bkgbkg_ind.shape)

    # adding to the indices to form the full matrix indices
    logging.info("Stitching together the non-zero indices ...")
    sigbkg_ind[:,1]+=len(full_sig)
    bkgsig_ind[:,0]+=len(full_sig)
    bkgbkg_ind += len(full_sig)
    logging.info("Generating sparse adjacency matrix ...")
    sparse_adj_mat, edge_ind, crow_ind, col_ind, values = adj.generate_sparse_adj_mat(sigsig_ind, sigbkg_ind, bkgsig_ind, bkgbkg_ind, len(full_sig)+30000)

    t = torch.cuda.get_device_properties(0).total_memory/(1024*1024*1024)
    r = torch.cuda.memory_reserved(0)/(1024*1024*1024)
    a = torch.cuda.memory_allocated(0)/(1024*1024*1024)
    f = r-a  # free inside reserved
    print("total: ", t, "reserved: ", r, "allocated:", a, "free: ", f)

#    pdb.set_trace()
    # print("Shape of sparse adj mat", sparse_adj_mat.shape)

    # saving adjacency matrix
    # Save the sparse tensor to a .pt file
    logging.info("Saving sparse adjacency matrix ...")
#    model_path = path+"models/"+model_label+"/"
    model_path = "/home/pacey/GNN/hhgraph_sparse/hhhgraph/models"+model_label+"/"
    misc.create_dirs(model_path)
    #torch.save(sparse_adj_mat, model_path+'sparse_adjacency_matrix.pt')
    torch.save(edge_ind, model_path+'coo_row.pt')
    torch.save(crow_ind, model_path+'csr_row.pt')
    torch.save(col_ind, model_path+'csr_col.pt')
    torch.save(values, model_path+'csr_values.pt')
    # # using the ROW indices of the sparse adj mat to read out the edge weights used for training

    t = torch.cuda.get_device_properties(0).total_memory/(1024*1024*1024)
    r = torch.cuda.memory_reserved(0)/(1024*1024*1024)
    a = torch.cuda.memory_allocated(0)/(1024*1024*1024)
    f = r-a  # free inside reserved
    print("total: ", t, "reserved: ", r, "allocated:", a, "free: ", f)
#    del crow_ind, col_ind, values
 #   torch.cuda.empty_cache() 
#    pdb.set_trace()
#    edge_wgts = full_wgts[edge_ind]#.cuda()
#    edge_wgts = full_wgts[edge_ind]
 #   print("edge weights", edge_wgts)

    # Still keeping the manually computed adjacency matrix
    # full_adj_mat = adj.generate_adj_mat(full_x, full_wgts, distance, linking_length).cuda()

# TODO: Adding plotting functionalities back that work with sparse adjacency matrix
#     # calculate centrality
#     logging.info("Calculating and plotting centrality ...")
#     deg_cent = torch.sum(full_adj_mat, dim=1).cuda()
#     plotting.plot_centrality(deg_cent, full_sig, full_bkg, path+"plots", eff)

#     adj_mat = full_adj_mat.to_sparse_csr() ### densor tensor to csr tensor
#     norm_label = training_name ### pyg layer uses D_half_inv normalisation

#     logging.info("Plotting convoluted kinematics ... ")
#     plotting.plot_conv_kinematics(adj_mat, deg_cent, raw_full_sig, raw_full_bkg, kinematics, eff, path+"/training_kinematics/"+norm_label, normalisation="D_half_inv", standardise=False)
#     plotting.plot_conv_conv_kinematics(adj_mat, deg_cent, raw_full_sig, raw_full_bkg, kinematics, eff, path+"/training_kinematics/"+norm_label, normalisation="D_half_inv", standardise=False)

#     print("Normalised adjacency matrix\n", adj_mat)

# else:
#     adj_mat = None
#     norm_label = ""


logging.info("Training ...")
# Define loss function for binary classification and ADAM optimiser
loss_function = nn.BCELoss()
optimiser = torch.optim.Adam(model.parameters(), lr=LR)
scaler = torch.cuda.amp.GradScaler()

print("full x", len(full_x))
print("train truth labels", len(train_truth_labels))
print("val truth labels", len(val_truth_labels))

gc.collect()
torch.cuda.empty_cache()
torch.cuda.reset_max_memory_allocated()
torch.cuda.synchronize()

for epoch in range(epochs):
    print(epoch)
    model.train()
    optimiser.zero_grad()

#    t = torch.cuda.get_device_properties(0).total_memory/(1024*1024*1024)
#    r = torch.cuda.memory_reserved(0)/(1024*1024*1024)
#    a = torch.cuda.memory_allocated(0)/(1024*1024*1024)
#    f = r-a  # free inside reserved
#    print("total: ", t, "reserved: ", r, "allocated:", a, "free: ", f)
#    torch.cuda.empty_cache()

#    full_outputs = model(full_x, sparse_adj_mat).cuda()
    with torch.cuda.amp.autocast():
        full_outputs = model(full_x, sparse_adj_mat).to("cpu")
        # splitting outputs into training/validation set
        # full x is concatenated as [train_sig : val_sig: train_bkg : val_bkg], so outputs need to be selected accordingly
        train_outputs = torch.cat((full_outputs[:len(train_sig)], full_outputs[(len(train_sig)+len(val_sig)):(len(train_sig)+len(val_sig)+len(train_bkg))]), dim=0)#.to(device)
        # train_outputs = full_outputs[:20000]
        # print("train outputs", len(train_outputs))
        val_outputs = torch.cat((full_outputs[(len(train_sig)):(len(train_sig)+len(val_sig))], full_outputs[-len(val_bkg):]), dim=0)#.cuda()
        # val_outputs = full_outputs[20000:33532]
        # print("val outputs", len(val_outputs))
        loss = loss_function(train_outputs.squeeze(), train_truth_labels.squeeze().cuda())#.cuda()

#    loss.backward()
#    train_loss.append(loss.item())
#    optimiser.step()
    scaler.scale(loss).backward()
    train_loss.append(loss.item())
    scaler.step(optimiser)

    model.eval()
    validation_loss = loss_function(val_outputs.squeeze(), val_truth_labels.squeeze().cuda())#.cuda()
    val_loss.append(validation_loss.item())

    print(f'Epoch {epoch + 1}/{epochs}, Train Loss: {loss.item()}, Validation Loss: {validation_loss.item()}')

# save trained model
logging.info("Saving trained model and performance...")
model_file_name = "model.pth"
torch.save({
    'model_state': model.state_dict(),
    'optimiser_state': optimiser.state_dict(),
}, model_path+model_file_name)

train_outputs = train_outputs.view(-1)
train_label_bool = train_truth_labels.bool()
train_sig_pred = train_outputs[train_label_bool]
train_bkg_pred = train_outputs[torch.logical_not(train_label_bool)]

train_fpr, train_tpr, train_cut = roc_curve(train_truth_labels.detach().cpu().numpy(), train_outputs.detach().cpu().numpy())
train_auc = roc_auc_score(train_truth_labels.detach().cpu().numpy(), train_outputs.detach().cpu().numpy())
print("Training AUC", train_auc)

val_outputs = val_outputs.view(-1)
val_label_bool = val_truth_labels.bool()
val_sig_pred = val_outputs[val_label_bool]
val_bkg_pred = val_outputs[torch.logical_not(val_label_bool)]
fig, ax = plt.subplots()

val_fpr, val_tpr, val_cut = roc_curve(val_truth_labels.detach().cpu().numpy(), val_outputs.detach().cpu().numpy())
val_auc = roc_auc_score(val_truth_labels.detach().cpu().numpy(), val_outputs.detach().cpu().numpy())
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
fig.savefig(fig_path+variable+"_"+model_label+"_training_validation_loss.pdf", transparent=True)

logging.info("Plotting model outputs ...")
fig, ax = plt.subplots()
binning = numpy.linspace(0,1,50)
ax.hist(train_sig_pred.detach().cpu().numpy(), bins=binning, label="Signal (training)", histtype='step', linestyle='--', density=True, color="darkorange")
ax.hist(train_bkg_pred.detach().cpu().numpy(), bins=binning, label="Background (training)", histtype='step', linestyle='--', density=True, color="steelblue")
ax.hist(val_sig_pred.detach().cpu().numpy(), bins=binning, label="Signal (validation)", alpha=0.5, density=True, color="darkorange")
ax.hist(val_bkg_pred.detach().cpu().numpy(), bins=binning, label="Background (validation)", alpha=0.5, density=True, color="steelblue")
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
fig.savefig(fig_path+variable+"_"+model_label+"_training_validation_pred.pdf", transparent=True)

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
fig.savefig(fig_path+variable+"_"+model_label+"_training_validation_ROC.pdf", transparent=True)
