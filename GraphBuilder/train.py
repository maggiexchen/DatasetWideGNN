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
test_sig, test_bkg, test_x, test_sig_wgts, test_bkg_wgts, test_truth_sig_labels, test_truth_bkg_labels = adj.data_loader(h5_path, "", "test", kinematics, signal=signal)

train_pairs= PairDataset(train_sig, train_bkg, 200, 200)
val_pairs = PairDataset(val_sig, val_bkg, 100, 100)
print("training pairs", len(train_pairs))
print("validation pairs", len(val_pairs))

train_loader = DataLoader(train_pairs, batch_size=512, shuffle=True)
val_loader = DataLoader(val_pairs, batch_size=128, shuffle=True)

model = EmbeddingNet(input_dim=len(kinematics), embedding_dim=2)
margin = user_config["margin"]
LR = user_config["LR"]
embedding_dim = user_config["embedding_dim"]
num_epoch = user_config["epoch"]
penalty = user_config["penalty"]
contrastive_hinge_loss = ContrastiveHingeLoss(margin=margin, embedding_dim=embedding_dim, pen=penalty)
optimizer = optim.Adam(model.parameters(), lr=LR)
num_epochs = num_epoch

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
    train_loss.append(epoch_loss)

    model.eval()
    running_val_loss = 0.0
    with torch.no_grad():  # Disable gradient computation
        for batch in val_loader:
            event1, event2, labels = batch
            batchsize = event1.size(0)
            event1 = event1.float()
            event2 = event2.float()
            labels = labels.float()

            # Forward pass
            output1 = model(event1)
            output2 = model(event2)

            # Compute the loss
            loss = contrastive_hinge_loss(output1, output2, labels)
            running_val_loss += loss.item() * batchsize

    # Compute average validation loss for the epoch
    epoch_val_loss = running_val_loss / len(val_loader.dataset)
    print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_loss:.4f}, Validation Loss: {epoch_val_loss:.4f}')
    val_loss.append(epoch_val_loss)

    model.train()

with torch.no_grad():
    test_data = torch.cat((test_sig, test_bkg), dim=0)
    test_idx = torch.randperm(len(test_data))[:500]
    test_embeddings = model(test_data[test_idx])
test_labels = torch.cat((test_truth_sig_labels, test_truth_bkg_labels), dim=0)[test_idx]

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
    purity = (torch.sum(adj_mat) - torch.sum(sigbkg_graph))/torch.sum(adj_mat)

    # calculate fraction of edges
    total = adj_mat.shape[0]**2
    edges = torch.sum(adj_mat)
    edge_frac = edges/total

    # sigisg efficiency and purity
    sigsig_eff = (torch.sum(sigsig_graph)) / sigsig_dist.shape[0]**2
    sigsig_pur = torch.sum(sigsig_graph) / torch.sum(adj_mat)

    # bkgbkg efficiency and purity
    bkgbkg_eff = (torch.sum(bkgbkg_graph)) / bkgbkg_dist.shape[0]**2
    bkgbkg_pur = torch.sum(bkgbkg_graph) / torch.sum(adj_mat)

    return efficiency.detach().numpy(), purity.detach().numpy(), edge_frac.detach().numpy(), sigsig_eff.detach().numpy(), sigsig_pur.detach().numpy(), bkgbkg_eff.detach().numpy(), bkgbkg_pur.detach().numpy()

test_sigsig_dist, test_sigbkg_dist, test_bkgbkg_dist, test_avg_sigsig_dist, test_avg_sigbkg_dist, test_avg_bkgbkg_dist = embedded_euclidean_dist(test_embeddings, test_labels)
print("Validation average sig-sig distance: ", test_avg_sigsig_dist)
print("Validation average bkg-bkg distance: ", test_avg_bkgbkg_dist)
print("Validation average sig-bkg distance: ", test_avg_sigbkg_dist)

def plot_embeddings(embeddings, labels, epoch, margin, radius=1.0, pen=1.0):
    sigsig_dist, sigbkg_dist, bkgbkg_dist, avg_sigsig_dist, avg_sigbkg_dist, avg_bkgbkg_dist = embedded_euclidean_dist(embeddings, labels)
    eff, purity, edge_frac, sigsig_eff, sigsig_pur, bkgbkg_eff, bkgbkg_pur = make_graph(sigsig_dist, sigbkg_dist, bkgbkg_dist, radius)

    fig = plt.figure(figsize=(12, 10))
    plt.rc('text', usetex=True)
    ax = fig.add_subplot()
    sig_label = (labels == 1)
    bkg_label = (labels == 0)
    ax.scatter(norm.standardise_tensor(embeddings[:,0])[bkg_label], norm.standardise_tensor(embeddings[:,1])[bkg_label], c='dodgerblue', label="Background")
    ax.scatter(norm.standardise_tensor(embeddings[:,0])[sig_label], norm.standardise_tensor(embeddings[:,1])[sig_label], c='deeppink', label="Signal")
    ax.legend(loc="upper right", fontsize=16)
    ax.text(0.03, 0.95, r"\textbf{Signal} - Leptoquark, \textbf{Background} - $t\bar{t}$, Single top", size=16, transform=ax.transAxes)
    ax.text(0.03, 0.91, r"\textbf{Average distances:}", size=16, transform=ax.transAxes)
    ax.text(0.03, 0.88, f"sig-sig - {avg_sigsig_dist.item():.3f}, bkg-bkg - {avg_bkgbkg_dist.item():.3f}, sig-bkg - {avg_sigbkg_dist.item():.3f}", size=16, transform=ax.transAxes)
    ax.text(0.03, 0.84, r"\textbf{Graph at radius }" + str(radius) + f", edge fraction - {edge_frac:.3f}", size=16, transform=ax.transAxes)
    ax.text(0.03, 0.81, f"Same class: efficiency - {eff:.3f}, purity - {purity:.3f}", size=16, transform=ax.transAxes)
    ax.text(0.03, 0.78, f"Sig-sig: efficiency - {sigsig_eff:.3f}, purity - {sigsig_pur:.3f}", size=16, transform=ax.transAxes)
    ax.text(0.03, 0.75, f"Bkg-bkg: efficiency - {bkgbkg_eff:.3f}, purity - {bkgbkg_pur:.3f}", size=16, transform=ax.transAxes)

    ax.set_xlabel('Embedded feature 1', loc="right", fontsize=16)
    ax.set_ylabel('Embedded feature 2', loc="top", fontsize=16)
    ax.set_xlim((-4, 4))
    ax.set_ylim((-4, 4))
    fig.savefig("embedding_e"+str(num_epochs)+"_m"+str(margin)+"_r"+str(radius)+"_Lambda"+str(pen)+".pdf")


radius = 0.1
plot_embeddings(test_embeddings, test_labels, num_epochs, margin, radius=radius, pen=penalty)

if num_epochs > 0:
    fig, ax = plt.subplots()
    x_epoch = np.arange(1,num_epochs+1,1)
    ax.plot(x_epoch, train_loss, color="cornflowerblue", label="Training")
    ax.plot(x_epoch, val_loss, color="darkorange", label="Validation")
    ax.set_xlabel("Epoch", loc="right")
    ax.set_ylabel("Loss", loc="top")
    ax.legend(loc="upper right")
    fig.savefig("loss_e"+str(num_epochs)+"_m"+str(margin)+"_r"+str(radius)+"_Lambda"+str(penalty)+".pdf", transparent=True)
