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
# from torchinfo import summary

from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader

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
CUDA_LAUNCH_BLOCKING=1
# device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# print(torch.cuda.mem_get_info())
device = torch.device('cpu')


train_config_path = args.MLconfig
train_config = misc.load_config(train_config_path)

user_config_path = args.userconfig
user_config = misc.load_config(user_config_path)
print("Using config ",user_config)
h5_path = user_config["h5_path"]
plot_path = user_config["plot_path"]
ll_path = user_config["ll_path"]
adj_path = user_config["adj_path"]
dist_path = user_config["dist_path"]
model_path = user_config["model_path"]

# TODO: assert. This should be "hhh" "LQ" or "stau"
signal = user_config["signal"]

# training_name = train_config["name"]
hidden_sizes_gcn = train_config["hidden_sizes_gcn"]
hidden_sizes_mlp = train_config["hidden_sizes_mlp"]
LR = train_config["LR"]
dropout_rates = train_config["dropout_rates"]
epochs = train_config["epochs"]
num_nb_list = train_config["num_nb_list"]
batch_size = train_config["batch_size"]
gnn_type = train_config["gnn_type"]
linking_length = train_config["linking_length"]

variable = train_config["variable"]
if variable is None:
    print("Need to specify a type of kinematic variable in the config")
distance = train_config["distance"]
if distance is None:
    print("Need to specify a type of distance metric for the adjacency matrix in the config")
eff = train_config["sigsig_eff"]

if linking_length is None:
    if eff is None:
        raise Exception("Need to specify a sig-sig efficiency for the adjacency matrix when training a gcn in the config")
    elif eff not in [0.6, 0.7, 0.8, 0.9]:
        raise Exception("not given a supported efficiency, (0.6, 0.7, 0.8, 0.9)")
    else:
        ll_str = "_LLEff" + str(eff).replace(".", "p")
        adj_path = adj_path + "/" + f"sigsig_eff_{eff}/"
else:
    eff = None
    print("linking length is given in config, IGNORING the sigsig_eff in the config!")
    ll_str = "_LL" + str(linking_length).replace(".", "p")
    adj_path = adj_path + "/" + f"linking_length_{linking_length}/"

if len(hidden_sizes_gcn) == 0:
    model_label = signal\
          + "_MLP" + "-".join(map(str, hidden_sizes_mlp)).replace(".", "p")\
          + "_lr" + str(LR).replace(".", "p")\
          + "_dr" + "-".join(map(str, dropout_rates)).replace(".", "p")\
          + "_bs" + str(batch_size)\
          + "_e" + str(epochs)
else:
    model_label = signal\
            + f"_{gnn_type}" + "-".join(map(str, hidden_sizes_gcn)).replace(".", "p")\
            + "_MLP" + "-".join(map(str, hidden_sizes_mlp)).replace(".", "p")\
            + "_nb" + "-".join(map(str, num_nb_list))\
            + "_lr" + str(LR).replace(".", "p")\
            + ll_str\
            + "_dr" + "-".join(map(str, dropout_rates)).replace(".", "p")\
            + "_bs" + str(batch_size)\
            + "_e" + str(epochs)


#if len(hidden_sizes_gcn) == 0:
#    model_label = "DNN"
#    plot_path = "plots/DNN/"
#else:
#    model_label = "GCN"
#    plot_path = "plots/GCN/"

#plot_path_label = "DNN" if len(hidden_sizes_gcn) == 0 else "GCN"
#plot_path += plot_path_label 
plot_path = plot_path + model_label + "/"
misc.create_dirs(plot_path)


kinematics = misc.get_kinematics(variable)
input_size = len(kinematics)

train_loss = []
val_loss = []

logging.info("signal: "+signal)
logging.info("chosen model: "+model_label)
logging.info("variable set: "+variable)
logging.info("input data path: "+h5_path)
logging.info("input ll json path: "+ll_path)
logging.info("input distances path: "+dist_path)
logging.info("output plot path: "+plot_path)
logging.info("adj matrix storage path: "+adj_path)
logging.info("model storage path: "+model_path)

model = GCNClassifier(input_size=input_size, hidden_sizes_gcn=hidden_sizes_gcn, hidden_sizes_mlp = hidden_sizes_mlp, output_size=1, dropout_rates=dropout_rates, gnn_type=gnn_type)
model.to(device)
# summary(model)
logging.info("distance metric: "+distance)
if eff is not None:
    logging.info("desired efficieny: "+str(eff))
elif linking_length is not None:
    logging.info("linking length: "+str(linking_length))

# load training data file and kinematics
logging.info('Importing signal and background files...')

# un-normalised signal and background kinematics (for checking and plotting)
# raw_train_sig, raw_train_bkg, _, _, _, _ = adj.data_loader(h5_path, plot_path, "train", kinematics, norm_kin=False, signal=signal)
# raw_val_sig, raw_val_bkg, _, _, _, _ = adj.data_loader(h5_path, plot_path, "val", kinematics, norm_kin=False, signal=signal)

# normalised signal and background kinematics
train_sig, train_bkg, train_x, train_sig_wgts, train_bkg_wgts, train_sig_labels, train_bkg_labels = adj.data_loader(h5_path, plot_path, "train", kinematics, norm_kin=True, signal=signal)
val_sig, val_bkg, val_x, val_sig_wgts, val_bkg_wgts, val_sig_labels, val_bkg_labels = adj.data_loader(h5_path, plot_path, "val", kinematics, norm_kin=True, signal=signal)

print("train sig", len(train_sig))
print("train bkg", len(train_bkg))
print("val sig", len(val_sig))
print("val bkg", len(val_bkg))


full_sig = torch.cat((train_sig, val_sig), dim=0)
full_sig_labels = torch.cat((train_sig_labels, val_sig_labels))
#full_bkg = torch.cat((train_bkg, val_bkg), dim=0)[:30000]
full_bkg = torch.cat((train_bkg, val_bkg), dim=0)
full_bkg_labels = torch.cat((train_bkg_labels, val_bkg_labels))

# raw_full_sig = torch.cat((raw_train_sig, raw_val_sig), dim=0)
# raw_full_bkg = torch.cat((raw_train_bkg, raw_val_bkg), dim=0)

full_x = torch.cat((full_sig, full_bkg), dim=0).to(device)
del full_sig
del full_bkg
full_y = torch.cat((full_sig_labels, full_bkg_labels), dim=0).to(device)
full_y = full_y.float()
#full_x = torch.cat((full_sig, full_bkg), dim=0)
full_wgts = torch.cat((torch.cat((train_sig_wgts, val_sig_wgts), dim=0), torch.cat((train_bkg_wgts, val_bkg_wgts), dim=0)), dim=0)#.cuda()

edge_ind = None
if len(hidden_sizes_gcn) > 0:
    print("constructing sparse adjacency matrix ...")
    row_ind = torch.load(adj_path+'coo_row.pt')
    col_ind = torch.load(adj_path+'csr_col.pt')
    # crow_ind = torch.load(adj_path+'csr_row.pt')
    # values = torch.load(adj_path+'csr_values.pt')
    edge_ind = torch.stack((row_ind, col_ind)).type(torch.int64).to(device)
    edge_wgts = full_wgts[row_ind]
    del row_ind
    del col_ind
    # del crow_ind

misc.print_mem_info()

def compute_class_weights(labels):
    labels = labels.astype(int)  # Ensure labels are integers
    class_counts = np.bincount(labels)
    class_weights = 1.0 / class_counts
    class_weights = class_weights / class_weights.sum()  # Normalize to sum to 1
    return torch.tensor(class_weights, dtype=torch.float)

logging.info("Training ...")
# Define loss function for binary classification and ADAM optimiser
# loss_function = nn.BCELoss()

def weighted_bce_loss(output, target, weights):
    loss = weights[1] * target * torch.log(output) + weights[0] * (1 - target) * torch.log(1 - output)
    return -loss.mean()

optimiser = torch.optim.Adam(model.parameters(), lr=LR)
# scaler = torch.cuda.amp.GradScaler()

print("full x", len(full_x))
print("full y", len(full_y))
print("edge ind", len(edge_ind))

gc.collect()
torch.cuda.empty_cache()
#torch.cuda.reset_max_memory_allocated()
#torch.cuda.synchronize()

data = Data(x = full_x, y = full_y, edge_index = edge_ind)#, edge_weight = edge_wgts)
del edge_ind

train_idx = torch.cat((torch.arange(len(train_sig)), torch.arange(len(train_sig)+len(val_sig), len(train_sig)+len(val_sig)+len(train_bkg))), dim=0).tolist()
val_idx = torch.cat((torch.arange(len(train_sig), len(train_sig)+len(val_sig)), torch.arange(len(full_x)-len(val_bkg), len(full_x))), dim=0).tolist()
print("train idx", len(train_idx))
print("val idx", len(val_idx))
# pdb.set_trace()

all_labels = data.y[train_idx].cpu().numpy()
class_weights = compute_class_weights(all_labels).to(device)
print("class weights", class_weights)

logging.info("Graph sub-sampling for training ...")
train_loader = NeighborLoader(
    data,
    input_nodes = train_idx,
    num_neighbors = num_nb_list,
    shuffle = True,
    batch_size = batch_size, 
    # num_workers = 6, 
    # persistent_workers = True
)

logging.info("Graph sub-sampling for validation ...")
val_loader = NeighborLoader(
    data,
    input_nodes = val_idx,
    num_neighbors = num_nb_list,
    # shuffle = False,
    batch_size = batch_size,
    # num_workers = 6,
    # persistent_workers = True
)

logging.info("Starting training ...")
for epoch in range(epochs):
    print(epoch)
    ### start training loop in the epoch
    model.train()
    total_examples = total_loss = 0
    train_outputs = torch.tensor([])
    train_truth_labels = torch.tensor([])
    i = 0
    for batch in train_loader:
        i += 1
        # print("batch: ", i)
        optimiser.zero_grad()
        batch = batch.to(device)
        batch_size = batch.batch_size
        # print("==>batch size: ", batch_size)
        # with torch.cuda.amp.autocast():
        # pdb.set_trace()
        outputs = model(batch.x, batch.edge_index) #, batch.edge_weight)
        
        ### NOTE only consider predictions and labels of seed nodes
        y = batch.y[:batch_size]
        outputs = outputs[:batch_size]
        
        # pdb.set_trace()
        # loss = loss_function(outputs.squeeze(), y.squeeze().cuda())#.cuda()
        loss = weighted_bce_loss(outputs.squeeze(), y.squeeze().float(), class_weights)

        loss.backward()
#    train_loss.append(loss.item())
        optimiser.step() 
        # scaler.scale(loss).backward()
        # scaler.step(optimiser)
        #misc.print_mem_info()
        torch.cuda.empty_cache()

        total_examples += batch_size
        total_loss += float(loss) * batch_size
        train_outputs = torch.cat((train_outputs, outputs.detach()))
        train_truth_labels = torch.cat((train_truth_labels, y.detach()))
        
        # scaler.update()

    avg_tr_loss = total_loss / total_examples
    train_loss.append(avg_tr_loss)
        
    ### start validation loop in the epoch
    model.eval()
    total_examples = total_loss = 0
    val_outputs = torch.tensor([])
    val_truth_labels = torch.tensor([])
    for batch in val_loader:
        
        batch = batch.to(device)
        batch_size = batch.batch_size
        outputs = model(batch.x, batch.edge_index) #, batch.edge_weight)

        ### NOTE only consider predictions and labels of seed nodes
        y = batch.y[:batch_size]
        outputs = outputs[:batch_size]

        # loss = loss_function(outputs.squeeze(), y.squeeze().cuda())#.cuda()
        loss = weighted_bce_loss(outputs.squeeze(), y.squeeze().float(), class_weights)

        total_examples += batch_size
        total_loss += float(loss) * batch_size
        val_outputs = torch.cat((val_outputs, outputs.detach()))
        val_truth_labels = torch.cat((val_truth_labels, y.detach()))

    avg_vl_loss = total_loss / total_examples
    val_loss.append(avg_vl_loss)
        
    print(f'Epoch {epoch + 1}/{epochs}, Train Loss: {avg_tr_loss}, Validation Loss: {avg_vl_loss}')


# train_truth_labels = full_y[train_idx]
# val_truth_labels = full_y[val_idx]
print("train truth labels", len(train_truth_labels))
print("val truth labels", len(val_truth_labels))

# model_path = path + "models/" + model_label + "/"
# save trained model
logging.info("Saving trained model and performance...")
model_file_name = "model.pth"
model_path = model_path+model_label+"/"
misc.create_dirs(model_path)
torch.save({
    'model_state': model.state_dict(),
    'optimiser_state': optimiser.state_dict(),
}, model_path+model_file_name)

# pdb.set_trace()
train_outputs = train_outputs.view(-1).to(device)
train_label_bool = train_truth_labels.bool()
train_sig_pred = train_outputs[train_label_bool]
train_bkg_pred = train_outputs[torch.logical_not(train_label_bool)]

train_fpr, train_tpr, train_cut = roc_curve(train_truth_labels.detach().cpu().numpy(), train_outputs.detach().cpu().numpy())
train_auc = roc_auc_score(train_truth_labels.detach().cpu().numpy(), train_outputs.detach().cpu().numpy())
print("Training AUC", train_auc)

val_outputs = val_outputs.view(-1).to(device)
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
misc.create_dirs(plot_path)
logging.info("Saving plots to "+plot_path)
fig.savefig(plot_path+variable+"_"+model_label+"_training_validation_loss.pdf", transparent=True)

logging.info("Plotting model outputs ...")
fig, ax = plt.subplots()
binning = numpy.linspace(0,1,50)
ax.hist(train_sig_pred.detach().cpu().numpy(), bins=binning, label="Signal (training)", histtype='step', linestyle='--', density=True, color="darkorange")
ax.hist(train_bkg_pred.detach().cpu().numpy(), bins=binning, label="Background (training)", histtype='step', linestyle='--', density=True, color="steelblue")
ax.hist(val_sig_pred.detach().cpu().numpy(), bins=binning, label="Signal (validation)", alpha=0.5, density=True, color="darkorange")
ax.hist(val_bkg_pred.detach().cpu().numpy(), bins=binning, label="Background (validation)", alpha=0.5, density=True, color="steelblue")
if eff is not None:
    text = ["Training AUC = {:.3f}".format(train_auc), "Validation AUC = {:.3f}".format(val_auc), "6b Resonant TRSM signal, 5b data", "Linking length at sig-sig efficiency "+str(eff)]
elif linking_length is not None:
    text = ["Training AUC = {:.3f}".format(train_auc), "Validation AUC = {:.3f}".format(val_auc), "6b Resonant TRSM signal, 5b data", "Linking length "+str(linking_length)]
plotting.add_text(ax, text, doATLAS=False, startx=0.02, starty=0.95)
#ax.text(0.02, 0.95, "Training AUC = {:.3f}".format(train_auc), verticalalignment="bottom", size=9, transform=ax.transAxes)
#ax.text(0.02, 0.91, "Validation AUC = {:.3f}".format(val_auc), verticalalignment="bottom", size=9, transform=ax.transAxes)
#ax.text(0.02, 0.87, "6b Resonant TRSM signal, 5b Data", verticalalignment="bottom", size=9, transform=ax.transAxes)
#ax.text(0.02, 0.83, "Linking length at sig-sig efficiency "+str(eff), verticalalignment="bottom", size=9, transform=ax.transAxes)
ax.legend(loc='upper right', fontsize=9)
ax.set_xlabel("Output score", loc="right")
ax.set_ylabel("Normalised No. Events", loc="top")
ymin, ymax = ax.get_ylim()
ax.set_ylim((ymin, ymax*1.2))
fig.savefig(plot_path+variable+"_"+model_label+"_training_validation_pred.pdf", transparent=True)

logging.info("Plotting ROC curves ...")
fig, ax = plt.subplots()
plt.plot(train_fpr, train_tpr, label='Training ROC curve (AUC = {:.3f})'.format(train_auc))
plt.plot(val_fpr, val_tpr, label='Validation ROC curve (AUC = {:.3f})'.format(val_auc))
plt.legend(loc="upper left", fontsize=9)
plt.xlim(0,1)
plt.xlabel("Background Efficiency", loc="right")
plt.ylabel("Signal Efficiency", loc="top")
fig.savefig(plot_path+variable+"_"+model_label+"_training_validation_ROC.pdf", transparent=True)
