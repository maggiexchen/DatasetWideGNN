from embedding import EventDataset
from embedding import EmbeddingNet
from model import MetricLearningModel
from model import contrastive_hinge_loss
from define_pair import PairDataset

import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
import utils.adj_mat as adj
import utils.misc as misc

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

import logging
logging.getLogger().setLevel(logging.INFO)
import argparse
import numpy as np

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

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
        "--userconfig",
        "-u",
        type=str,
        required=True,
        help="Specify the config for the user e.g. paths to store all the input/output data and results, signal model to look at",
    )

    args = parser.parse_args()
    return args

args = GetParser()
variable = str(args.variable)
kinematics = misc.get_kinematics(variable)

user_config_path = args.userconfig
user_config = misc.load_config(user_config_path)
h5_path = user_config["h5_path"]
signal = user_config["signal"]

# load in input files
logging.info('Importing signal and background files...')
train_sig, train_bkg, train_x, train_sig_wgts, train_bkg_wgts, train_truth_sig_labels, train_truth_bkg_labels = adj.data_loader(h5_path, "", "train", kinematics, signal=signal)
val_sig, val_bkg, val_x, val_sig_wgts, val_bkg_wgts, val_truth_sig_labels, val_truth_bkg_labels = adj.data_loader(h5_path, "", "val", kinematics, signal=signal)

train_pairs= PairDataset(train_sig, train_bkg, 100, 100)
val_pairs = PairDataset(val_sig, val_bkg, 50, 50)
print("training pairs", len(train_pairs))
print("validation pairs", len(val_pairs))

train_loader = DataLoader(train_pairs, batch_size=512, shuffle=True)
val_loader = DataLoader(val_pairs, batch_size=512, shuffle=True)

# Instantiate model, criterion, and optimizer
model = MetricLearningModel(input_dim=len(kinematics), embedding_dim=2)
criterion = contrastive_hinge_loss
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 50

train_output1 = torch.tensor([])
train_output2 = torch.tensor([])
train_labels = torch.tensor([])

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for batch in train_loader:
        signal_events, background_events, labels = batch
        signal_events = signal_events.float()
        background_events = background_events.float()
        labels = labels.float()

        optimizer.zero_grad()
        output1, output2 = model(signal_events, background_events)
        train_output1 = torch.cat((train_output1, output1))
        train_output2 = torch.cat((train_output2, output2))
        train_labels = torch.cat((train_labels, labels))
        loss = criterion(output1, output2, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * signal_events.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}')

val_output1 = torch.tensor([])
val_output2 = torch.tensor([])
val_labels = torch.tensor([])

model.eval()
with torch.no_grad():
    for signal_events, background_events, labels in val_loader:
        # Convert to appropriate tensor types if needed
        signal_events = signal_events.float()
        background_events = background_events.float()
        labels = labels.float()

        # Forward pass
        output1, output2 = model(signal_events, background_events)
        # Store outputs and labels for plotting
        val_output1=torch.cat((val_output1, output1))
        val_output2=torch.cat((val_output2, output2))
        val_labels=torch.cat((val_labels, labels))

val_sigsig_ind = (val_labels == 1)
val_sigbkg_ind = (val_labels == 0)

# Plot the reduced embeddings
plt.figure(figsize=(12, 10))
plt.scatter(val_output1[val_sigsig_ind].detach().numpy(), val_output2[val_sigsig_ind].detach().numpy(), c='red', label='Sig-sig edges', alpha=0.5)
plt.scatter(val_output1[val_sigbkg_ind].detach().numpy(), val_output2[val_sigbkg_ind].detach().numpy(), c='blue', label='Sig-bkg edges', alpha=0.5)

plt.xlabel('Embedded feature 1', loc="right", fontsize=14)
plt.ylabel('Embedded feature 2', loc="top", fontsize=14)
plt.legend(loc="best", fontsize=14)
plt.savefig("embedding.pdf")
