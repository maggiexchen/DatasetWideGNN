from embedding import EventDataset
from embedding import EmbeddingNet
from embedding import ContrastiveHingeLoss
# from model import contrastive_hinge_loss
from define_pair import PairDataset
import networkx as nx

import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
import utils.adj_mat as adj
import utils.misc as misc
import utils.normalisation as norm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
torch.manual_seed(42)

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

train_pairs= PairDataset(train_sig, train_bkg, 200, 200)
# val_pairs = PairDataset(val_sig, val_bkg, 100, 100)
print("training pairs", len(train_pairs))
# print("validation pairs", len(val_pairs))

train_loader = DataLoader(train_pairs, batch_size=512, shuffle=True)
# val_loader = DataLoader(val_pairs, batch_size=128, shuffle=True)

model = EmbeddingNet(input_dim=len(kinematics), embedding_dim=2)
margin = 0.5
contrastive_hinge_loss = ContrastiveHingeLoss(margin=margin, embedding_dim=2)
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 30

train_output1 = torch.tensor([])
train_output2 = torch.tensor([])
train_labels = torch.tensor([])

train_loss = []
val_loss = []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for batch in train_loader:
        event1, event2, labels = batch
        batchsize = event1.size(0)
        event1 = event1.float()
        event2 = event2.float()
        labels = labels.float()

        optimizer.zero_grad()
        # output1, output2 = model(signal_events, background_events)
        output1 = model(event1)
        output2 = model(event2)
        train_output1 = torch.cat((train_output1, output1))
        train_output2 = torch.cat((train_output2, output2))
        train_labels = torch.cat((train_labels, labels))
        loss = contrastive_hinge_loss(output1, output2, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * batchsize

    epoch_loss = running_loss / len(train_loader.dataset)
    print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_loss:.4f}')
    train_loss.append(epoch_loss)

model.eval()
val_running_loss = 0.0
num_batches = 0
with torch.no_grad():

    val_data = torch.cat((val_sig, val_bkg), dim=0)
    val_idx = torch.randperm(len(val_data))[:500]
    val_embeddings = model(val_data[val_idx])
val_labels = torch.cat((val_truth_sig_labels, val_truth_bkg_labels), dim=0)[val_idx]

def embedded_euclidean_dist(embeddings, labels):
    sig_label = (labels == 1)
    bkg_label = (labels == 0)
    sig_embedding = embeddings[sig_label]
    bkg_embedding = embeddings[bkg_label]
    
    sigsig_dist = torch.cdist(sig_embedding, sig_embedding, p=2)
    sigsig_dist_flat = sigsig_dist[torch.triu(torch.ones(sigsig_dist.size(0), sigsig_dist.size(0)), diagonal=1).bool()] 
    avg_sigsig_dist = sigsig_dist_flat.mean()
    
    bkgbkg_dist = torch.cdist(bkg_embedding, bkg_embedding, p=2)
    bkgbkg_dist_flat = bkgbkg_dist[torch.triu(torch.ones(bkgbkg_dist.size(0), bkgbkg_dist.size(0)), diagonal=1).bool()] 
    avg_bkgbkg_dist = bkgbkg_dist_flat.mean()

    sigbkg_dist = torch.cdist(sig_embedding, bkg_embedding, p=2)
    avg_sigbkg_dist = sigbkg_dist.flatten().mean()

    return sigsig_dist, sigbkg_dist, bkgbkg_dist, avg_sigsig_dist.detach().numpy(), avg_sigbkg_dist.detach().numpy(), avg_bkgbkg_dist.detach().numpy()

# define a threshold radius, and make connections for each event
def make_graph(sigsig_dist, sigbkg_dist, bkgbkg_dist, radius):
    top_half = torch.cat((sigsig_dist, sigbkg_dist), dim=1)  # Shape: s x (s + b)
    bottom_half = torch.cat((sigbkg_dist.T, bkgbkg_dist), dim=1)  # Shape: b x (s + b)

    sigsig_graph = adj.create_adj_mat(sigsig_dist, radius)
    bkgbkg_graph = adj.create_adj_mat(bkgbkg_dist, radius)
    sigbkg_graph = adj.create_adj_mat(sigbkg_dist, radius)

    top_graph = torch.cat((sigsig_graph, sigbkg_graph), dim=1)
    bottom_graph = torch.cat((sigbkg_graph.T, bkgbkg_graph), dim=1)

    # make adjacency matrix
    adj_mat = torch.cat((top_graph, bottom_graph), dim=0)  # Shape: (s + b) x (s + b)

    # calculate graph efficiency
    same_class = sigsig_dist.shape[0]**2 + bkgbkg_dist.shape[0]**2
    efficiency = (torch.sum(sigsig_graph) + torch.sum(bkgbkg_graph)) / same_class

    # calcualte graph purity
    num = abs(torch.sum(sigsig_graph) + torch.sum(bkgbkg_graph) - torch.sum(sigbkg_graph))
    den = abs(torch.sum(adj_mat) - torch.sum(sigbkg_graph))
    purity = num/den

    # calculate fraction of edges
    total = adj_mat.shape[0]**2
    edges = torch.sum(adj_mat)
    edge_frac = edges/total

    return efficiency.detach().numpy(), purity.detach().numpy(), edge_frac.detach().numpy()

val_sigsig_dist, val_sigbkg_dist, val_bkgbkg_dist, val_avg_sigsig_dist, val_avg_sigbkg_dist, val_avg_bkgbkg_dist = embedded_euclidean_dist(val_embeddings, val_labels)
print("Validation average sig-sig distance: ", val_avg_sigsig_dist)
print("Validation average bkg-bkg distance: ", val_avg_bkgbkg_dist)
print("Validation average sig-bkg distance: ", val_avg_sigbkg_dist)

def plot_embeddings(embeddings, labels, epoch, margin, radius=1.0):
    sigsig_dist, sigbkg_dist, bkgbkg_dist, avg_sigsig_dist, avg_sigbkg_dist, avg_bkgbkg_dist = embedded_euclidean_dist(embeddings, labels)
    eff, purity, edge_frac = make_graph(sigsig_dist, sigbkg_dist, bkgbkg_dist, radius)

    fig = plt.figure(figsize=(12, 10))
    plt.rc('text', usetex=True)
    ax = fig.add_subplot()
    sig_label = (labels == 1)
    bkg_label = (labels == 0)
    ax.scatter(norm.standardise_tensor(embeddings[:,0])[bkg_label], norm.standardise_tensor(embeddings[:,1])[bkg_label], c='b', label="Background")
    ax.scatter(norm.standardise_tensor(embeddings[:,0])[sig_label], norm.standardise_tensor(embeddings[:,1])[sig_label], c='r', label="Signal")
    ax.legend(loc="upper right", fontsize=16)
    ax.text(0.03, 0.95, r"\textbf{Signal} - Leptoquark, \textbf{Background} - $t\bar{t}$, Single top", size=16, transform=ax.transAxes)
    ax.text(0.03, 0.91, r"\textbf{Average distances:}", size=16, transform=ax.transAxes)
    ax.text(0.03, 0.88, f"sig-sig - {avg_sigsig_dist.item():.3f}, bkg-bkg - {avg_bkgbkg_dist.item():.3f}, sig-bkg - {avg_sigbkg_dist.item():.3f}", size=16, transform=ax.transAxes)
    ax.text(0.03, 0.84, r"\textbf{Graph at radius = }" + str(radius) + ":", size=16, transform=ax.transAxes)
    ax.text(0.03, 0.81, f"Efficiency - {eff:.3f}, purity - {purity:.3f}, edge fraction - {edge_frac:.3f}", size=16, transform=ax.transAxes)

    ax.set_xlabel('Embedded feature 1', loc="right", fontsize=16)
    ax.set_ylabel('Embedded feature 2', loc="top", fontsize=16)
    ymin, ymax = ax.get_ylim()
    ax.set_ylim((ymin, ymax*1.5))
    fig.savefig("embedding_e"+str(num_epochs)+"_m"+str(margin)+"_r"+str(radius)+".pdf")

# for radius in np.arange(0.08, 0.19, 0.01):
#     val_eff, val_pur, val_edge_frac = make_graph(val_sigsig_dist, val_sigbkg_dist, val_bkgbkg_dist, radius=radius)
#     print("At radius ", radius)
#     print("Same class edges efficiency - ", val_eff)
#     print("Edge purity - ", val_pur)
#     print("Edge fraciton - ", val_edge_frac)

radius = 0.15
plot_embeddings(val_embeddings, val_labels, num_epochs, margin, radius=radius)

# fig, ax = plt.subplots()
# x_epoch = np.arange(1,num_epochs+1,1)
# ax.plot(x_epoch, train_loss)
# ax.set_xlabel("Epoch", loc="right")
# ax.set_ylabel("Training loss", loc="top")
# fig.savefig("training_loss.pdf", transparent=True)
