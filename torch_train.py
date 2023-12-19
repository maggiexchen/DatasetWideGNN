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
import logging
logging.getLogger().setLevel(logging.INFO)
import argparse
import utils.normalisation as norm
import utils.torch_distances as dis
import utils.adj_mat as adj

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import torch_geometric

from utils.gcn_model import GCNClassifier

import time
st = time.time()


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

  args = parser.parse_args()
  return args

args = GetParser()

if args.variable == "mass":
    # mass-based kinematics
    kinematics = ["mH1","mH2","mH3","mHHH"]
elif args.variable == "angular":
    # angular kinematics
    kinematics = ["dRH1","dRH2","dRH3","meandRBB"]
elif args.variable == "shape":
    # event shape kinematics
    kinematics = ["sphere3dv2b","sphere3dv2btrans","aplan3dv2b","theta3dv2b"]
elif args.variable == "combined":
    kinematics = ["mH1","mH2","mH3","mHHH","dRH1","dRH2","dRH3","meandRBB","sphere3dv2b","sphere3dv2btrans","aplan3dv2b","theta3dv2b"]
else:
    print("bruh")

# load training data file and kinematics
logging.info('Importing signal and background files...')
file_path = "/data/atlas/atlasdata3/maggiechen/gnn_project/split_files/"
train_sig, train_bkg, train_x, train_wgts, train_truth_labels = adj.data_loader(file_path, "train", kinematics)
val_sig, val_bkg, val_x, val_wgts, val_truth_labels = adj.data_loader(file_path, "val", kinematics)

# read in linking length calculated from sampled training data
sigsig_eff = args.eff
with open('/data/atlas/atlasdata3/maggiechen/gnn_project/linking_lengths/'+args.variable+"_"+args.distance+"_linking_length.json", 'r') as lfile:
    length_dict = json.load(lfile)
    eff = length_dict["sigsig_eff"]
    ss_bb_lengths = length_dict["ss_bb_length"]
    linking_length = ss_bb_lengths[eff.index(sigsig_eff)]
    print("linking length", linking_length)

# calculate distances and generate adjacency matrix in batches
logging.info('Calculating training and validaiton distances in batches...')
chunksize = 20000
train_adj_mat = adj.generate_adj_mat(train_x, train_wgts, chunksize, args.distance, linking_length)
val_adj_mat = adj.generate_adj_mat(val_x, val_wgts, chunksize, args.distance, linking_length)

print(train_adj_mat)
input_size = len(kinematics)
hidden_sizes = [4]
LR = 0.001
epochs = 30
train_loss = []
val_loss = []

gcn_model = GCNClassifier(input_size=input_size, hidden_sizes=hidden_sizes, output_size=1)

# # Load in training dataset, adjacency matrix, labels
# # Load in validation dataset, adjacency matrix, labels
train_dataset = TensorDataset(train_x, train_adj_mat, train_truth_labels)
val_dataset = TensorDataset(val_x, val_adj_mat, val_truth_labels)

train_loader = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gcn_model.to(device)

loss_function = nn.BCEWithLogitsLoss()
optimiser = torch.optim.Adam(gcn_model.parameters(), lr=LR)

for epoch in range(epochs):
    gcn_model.train()
    for data in train_loader:
        train_inputs, train_adjacency_matrix, train_labels = data
        train_inputs, train_adjacency_matrix, train_labels = train_inputs.to(device), train_adjacency_matrix.to(device), train_labels.to(device)
        optimiser.zero_grad()
        train_outputs = gcn_model(train_inputs, train_adjacency_matrix)
        loss = loss_function(train_outputs.squeeze(), train_labels.squeeze())
        loss.backward()
        optimiser.step()
    train_loss.append(loss.item())

    gcn_model.eval()
    with torch.no_grad():
        total_val_loss = 0.0
        for data in val_loader:
            val_inputs, val_adjacency_matrix, val_labels = data
            val_inputs, val_adjacency_matrix, val_labels = val_inputs.to(device), val_adjacency_matrix.to(device), val_labels.to(device)
            val_outputs = gcn_model(val_inputs, val_adjacency_matrix)
            validation_loss = loss_function(val_outputs.squeeze(), val_labels.squeeze())
            total_val_loss += validation_loss.item()
    avg_val_loss = total_val_loss / len(val_loader)
    val_loss.append(avg_val_loss)

    print(f'Epoch {epoch + 1}/{epochs}, Train Loss: {loss.item()}, Validation Loss: {avg_val_loss}')


val_outputs = val_outputs.view(-1)
val_label_bool = val_labels.bool()
val_sig_pred = val_outputs[val_label_bool]
val_bkg_pred = val_outputs[torch.logical_not(val_label_bool)]
print("sig pred", val_sig_pred)
print("bkg pred", val_bkg_pred)
fig, ax = plt.subplots()
binning = numpy.linspace(0,1,20)
ax.hist(val_sig_pred, bins=binning, label="Signal", histtype="step", density=True)
ax.hist(val_bkg_pred, bins=binning, label="Background", histtype="step", density=True)
ax.legend(loc='upper right')
ax.set_xlabel("GNN Score", loc="right")
ax.set_ylabel("Normalised No. Events", loc="top")
fig.savefig("test_pred.pdf", transparent=True)
