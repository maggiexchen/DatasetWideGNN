import pandas as pd
import uproot
import numpy
import h5py
import json
import math
import random
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
from utils.gcn_model import GCNClassifier
from utils.dnn_model import DNNClassifier

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
#import torch_geometric

import time
st = time.time()
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, auc
from sklearn.utils import shuffle

import shap
import logging
logging.getLogger().setLevel(logging.INFO)

#import pdb

def GetParser():
  """Argument parser for reading Ntuples script."""
  parser = argparse.ArgumentParser(
      description="Reading Ntuples command line options."
  )

  parser.add_argument(
      "--variable",
      "-v",
      type=str,
      required=True,
      help="Specify the type of kinematic variables to calculate distance for",
  )

  parser.add_argument(
      "--distance",
      "-d",
      type=str,
      required=True,
      help="Specify the type of distance to calculate",
  )

  parser.add_argument(
      "--eff",
      "-e",
      type=float,
      required=True,
      help="Specify sig-sig efficiency for the linking length",
  )

  parser.add_argument(
      "--path",
      "-p",
      type=str,
      required=False,
      help="Specify the path to store all the input/output data and results",
  )

  parser.add_argument(
      "--model",
      "-m",
      type=str,
      required=True,
      help="Specify the type of model",
  )

  args = parser.parse_args()
  return args

args = GetParser()

variable = str(args.variable)
distance = str(args.distance)
eff = args.eff
if eff not in [0.6, 0.7, 0.8, 0.9]:
    raise Exception("not given a supported efficiency, (0.6, 0.7, 0.8, 0.9)")

path = "/data/atlas/atlasdata3/maggiechen/gnn_project/"
if args.path:
    path = args.path
    if path[-1]!="/": path += "/"

kinematics = misc.get_kinematics(variable)
input_size = len(kinematics)
hidden_sizes = [16, 16, 16]
LR = 0.01
epochs = 100
train_loss = []
val_loss = []


modelname = str(args.model)
if (modelname.lower() == "gcn"):
    model = GCNClassifier(input_size=input_size, hidden_sizes=hidden_sizes, output_size=1)
elif (modelname.lower() == "dnn"):
    model = DNNClassifier(input_size=input_size, hidden_sizes=hidden_sizes, output_size=1)
else:
    raise Exception("Janky, pick a defined model (dnn, gcn)")


logging.info("variable set: "+variable)
logging.info("distance metric: "+distance)
logging.info("input/output path: "+path)
logging.info("desired efficieny: "+str(eff))
logging.info("chosen model: "+modelname)


# load training data file and kinematics
logging.info('Importing signal and background files...')
train_sig, train_bkg, train_x, train_wgts, train_truth_labels = adj.data_loader(path, "train", kinematics)
val_sig, val_bkg, val_x, val_wgts, val_truth_labels = adj.data_loader(path, "val", kinematics)

# read in linking length calculated from sampled training data
sigsig_eff = eff
ll_path = path+"linking_lengths/"+str(variable)+"_"+str(distance)+"_linking_length.json"
misc.create_dirs(ll_path)
with open(ll_path, 'r') as lfile:
    length_dict = json.load(lfile)
    lengths = length_dict["length"]
    linking_length = lengths[length_dict["sigsig_eff"].index(sigsig_eff)]
    logging.info("linking length ="+str(linking_length))

# calculate distances and generate adjacency matrix in batches
logging.info('Calculating training and validaiton distances in batches...')
train_adj_mat = adj.generate_adj_mat(train_x, train_wgts, distance, linking_length)
val_adj_mat = adj.generate_adj_mat(val_x, val_wgts, distance, linking_length)

# # Load in training dataset, adjacency matrix, labels
# # Load in validation dataset, adjacency matrix, labels
train_dataset = TensorDataset(train_x, train_adj_mat, train_truth_labels)
val_dataset = TensorDataset(val_x, val_adj_mat, val_truth_labels)

# Define loss function for binary classification and ADAM optimiser
loss_function = nn.BCELoss()
optimiser = torch.optim.Adam(model.parameters(), lr=LR)

## TODO: load in the training the validation data differently
## TODO: define the training and evaluation steps as functions
for epoch in range(epochs):
    model.train()
    optimiser.zero_grad()
    if args.model == "gcn":
        train_outputs = model(train_x, train_adj_mat)
    elif args.model == "dnn":
        train_outputs = model(train_x)
    loss = loss_function(train_outputs.squeeze(), train_truth_labels.squeeze())
    loss.backward()
    train_loss.append(loss.item())
    optimiser.step()

    model.eval()
    if args.model == "gcn":
        val_outputs = model(val_x, val_adj_mat)
    elif args.model == "dnn":
        val_outputs = model(val_x)
    validation_loss = loss_function(val_outputs.squeeze(), val_truth_labels.squeeze())
    val_loss.append(validation_loss.item())

    print(f'Epoch {epoch + 1}/{epochs}, Train Loss: {loss.item()}, Validation Loss: {validation_loss.item()}')

logging.info("Plotting training outputs ...")
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

#binning = numpy.linspace(0,1,50)
fig, ax = plt.subplots()
binning = numpy.linspace(0,1,50)
ax.hist(val_sig_pred.detach().numpy(), bins=binning, label="Signal", alpha=0.5, density=True)
ax.hist(val_bkg_pred.detach().numpy(), bins=binning, label="Background", alpha=0.5, density=True)
ax.text(0.04, 0.93, "Validation AUC = {:.3f}".format(val_auc), verticalalignment="bottom", size=10, transform=ax.transAxes)
ax.text(0.04, 0.88, "Standardised kinematics", verticalalignment="bottom", size=10, transform=ax.transAxes)
ax.legend(loc='upper right')
ax.set_xlabel("GNN Score", loc="right")
ax.set_ylabel("Normalised No. Events", loc="top")
fig_path = path + "plots/GCN/"
misc.create_dirs(fig_path)
fig.savefig(fig_path+variable+"_"+modelname+"_validation_pred.pdf", transparent=True)

fig, ax = plt.subplots()
binning = numpy.linspace(0,1,50)
ax.hist(train_sig_pred.detach().numpy(), bins=binning, label="Signal", alpha=0.5, density=True)
ax.hist(train_bkg_pred.detach().numpy(), bins=binning, label="Background", alpha=0.5, density=True)
ax.text(0.04, 0.93, "Training AUC = {:.3f}".format(train_auc), verticalalignment="bottom", size=10, transform=ax.transAxes)
ax.text(0.04, 0.88, "Standardised kinematics", verticalalignment="bottom", size=10, transform=ax.transAxes)
ax.legend(loc='upper right')
ax.set_xlabel("GNN Score", loc="right")
ax.set_ylabel("Normalised No. Events", loc="top")
fig_path = path + "plots/GCN/"
misc.create_dirs(fig_path)
fig.savefig(fig_path+variable+"_"+modelname+"_training_pred.pdf", transparent=True)

logging.info("Plotting ROC curves ...")
fig, ax = plt.subplots()
plt.plot(train_fpr, train_tpr, label='Training ROC curve (AUC = {:.3f})'.format(train_auc))
plt.legend(loc="upper left", fontsize=10)
ymin, ymax = plt.ylim()
plt.ylim(ymin, ymax*1.2)
plt.xlim(0,1)
plt.xlabel("Background Efficiency")
plt.ylabel("Signal Efficiency")
fig_path = path + "plots/GCN/"
misc.create_dirs(fig_path)
fig.savefig(fig_path+variable+"_"+modelname+"_training_ROC.pdf", transparent=True)

fig, ax = plt.subplots()
plt.plot(val_fpr, val_tpr, label='Validation ROC curve (AUC = {:.3f})'.format(val_auc))
plt.legend(loc="upper left", fontsize=10)
ymin, ymax = plt.ylim()
plt.ylim(ymin, ymax*1.2)
plt.xlim(0,1)
plt.xlabel("Background Efficiency")
plt.ylabel("Signal Efficiency")
fig_path = path + "plots/GCN/"
misc.create_dirs(fig_path)
fig.savefig(fig_path+variable+"_"+modelname+"_validation_ROC.pdf", transparent=True)
