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
import utils.plotting as plotting

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
torch.manual_seed(42)

import logging
logging.getLogger().setLevel(logging.INFO)
import argparse
import numpy as np
import json

from sklearn.manifold import TSNE
tsne = TSNE(n_components=2, random_state=42)
import matplotlib.pyplot as plt

def GetParser():
    """Argument parser for reading Ntuples script."""
    parser = argparse.ArgumentParser(
        description="Reading Ntuples command line options."
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
user_config_path = args.userconfig
user_config = misc.load_config(user_config_path)

variable = user_config["variable"]
feature_dim = user_config["feature_dim"]
kinematics, kinematic_labels = misc.get_kinematics(variable, feature_dim)
signal = user_config["signal"]
signal_mass = user_config["signal_mass"]
assert signal in ["hhh", "LQ", "stau"], f"Invalid signal type: {signal}"
signal_label, background_label = plotting.get_plot_labels(signal)

h5_path = user_config["h5_path"]
model_save_path = user_config["model_path"]
plot_path = user_config["plot_path"]
signal = user_config["signal"]
embedding_dim = user_config["embedding_dim"]
os.makedirs(plot_path, exist_ok=True)

# load in input files
logging.info('Importing signal and background files...')
full_sig, full_bkg, full_x, sig_wgt, bkg_wgt, sig_labels, bkg_labels = adj.data_loader(h5_path, plot_path, kinematics, kinematic_labels, ex="", plot=False, signal=signal, signal_mass=signal_mass, standardisation=False)

dataset = PairDataset(full_sig, full_bkg, 400, 400, standardise=True)
train_pairs = dataset.train_pairs
val_pairs = dataset.val_pairs
train_means = dataset.means
train_stds = dataset.stds

# val_dataset = PairDataset(full_sig, full_bkg, 200, 200, standardise=True)
# val_pairs = val_dataset.pairs

print("training pairs", len(train_pairs))
print("validation pairs", len(val_pairs))

train_loader = DataLoader(train_pairs, batch_size=256, shuffle=True)
val_loader = DataLoader(val_pairs, batch_size=256, shuffle=True)
model = EmbeddingNet(input_dim=len(kinematics), embedding_dim=embedding_dim)
margin = user_config["margin"]
LR = user_config["LR"]
embedding_dim = user_config["embedding_dim"]
num_epoch = user_config["epoch"]
penalty = user_config["penalty"]
radius = margin/2

contrastive_hinge_loss = ContrastiveHingeLoss(margin=margin, embedding_dim=embedding_dim, pen=penalty)
optimiser = optim.Adam(model.parameters(), lr=LR)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimiser, 'min')
num_epochs = num_epoch

train_output1 = torch.tensor([])
train_output2 = torch.tensor([])
train_labels = torch.tensor([])

train_loss = []
val_loss = []
print("Training ...")
print("Margin: ", margin)
print("Embedding dim: ", embedding_dim)
print("LR: ", LR)
print("Penalty lambda: ", penalty)

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for batch in train_loader:
        event1, event2, labels = batch
        batchsize = event1.size(0)
        event1 = event1.float()
        event2 = event2.float()
        labels = labels.float()

        optimiser.zero_grad()
        output1 = model(event1)
        output2 = model(event2)
        train_output1 = torch.cat((train_output1, output1))
        train_output2 = torch.cat((train_output2, output2))
        train_labels = torch.cat((train_labels, labels))
        loss = contrastive_hinge_loss(output1, output2, labels)
        loss.backward()
        optimiser.step()
        running_loss += loss.item() * batchsize

    epoch_loss = running_loss / len(train_loader.dataset)
    train_loss.append(epoch_loss)

    model.eval()
    running_val_loss = 0.0
    with torch.no_grad():  # Disable gradient computation
        for batch in val_loader:
            # event 1 and event 2 are the events in a pair 
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

    scheduler.step(epoch_val_loss)
    model.train()

logging.info("Saving trained model and performance...")
model_file_name = "EmbeddingNet_m"+str(margin)+"_r"+str(radius)+"_Lambda"+str(penalty)+"_model.pth"
model_path = model_save_path + "embedding_"+str(embedding_dim)+"feats/"
os.makedirs(model_path, exist_ok=True)
torch.save({
    'model_state': model.state_dict(),
    'optimiser_state': optimiser.state_dict(),
    'normalisation_params': {"means": train_means, "stds": train_stds}
}, model_path+model_file_name)

with torch.no_grad():
    sig_sample_ind = torch.randperm(len(full_sig))[:250]
    bkg_sample_ind = torch.randperm(len(full_bkg))[:250]
    test_data = torch.cat((full_sig[sig_sample_ind], full_bkg[bkg_sample_ind]), dim=0)
    test_labels = torch.cat((sig_labels[sig_sample_ind], bkg_labels[bkg_sample_ind]), dim=0)
    test_embeddings = model(test_data)

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

def plot_embeddings(embeddings, labels, epoch, margin, feat, radius=1.0, pen=1.0):
    sigsig_dist, sigbkg_dist, bkgbkg_dist, avg_sigsig_dist, avg_sigbkg_dist, avg_bkgbkg_dist = embedded_euclidean_dist(embeddings, labels)
    eff, purity, edge_frac, sigsig_eff, sigsig_pur, bkgbkg_eff, bkgbkg_pur = make_graph(sigsig_dist, sigbkg_dist, bkgbkg_dist, radius)
    fig = plt.figure(figsize=(12, 10))
    plt.rc('text', usetex=True)
    ax = fig.add_subplot()
    sig_label = (labels == 1)
    bkg_label = (labels == 0)
    if feat==2:
        ax.scatter(embeddings[:,0][bkg_label], embeddings[:,1][bkg_label], c='dodgerblue', label="Background")
        ax.scatter(embeddings[:,0][sig_label], embeddings[:,1][sig_label], c='deeppink', label="Signal")
    else:
        # embeddings_2d = tsne.fit_transform(embeddings)
        # ax.scatter(embeddings_2d[:,0][bkg_label], embeddings_2d[:,1][bkg_label], c='dodgerblue', label="Background")
        # ax.scatter(embeddings_2d[:,0][sig_label], embeddings_2d[:,1][sig_label], c='deeppink', label="Signal")
        ax.scatter(embeddings[:,0][bkg_label], embeddings[:,1][bkg_label], c='dodgerblue', label="Background")
        ax.scatter(embeddings[:,0][sig_label], embeddings[:,1][sig_label], c='deeppink', label="Signal")

    ax.legend(loc="upper right", fontsize=16)
    ax.text(0.03, 0.95, r"\textbf{Signal} - Leptoquark, \textbf{Background} - $t\bar{t}$, Single top", size=16, transform=ax.transAxes)
    ax.text(0.03, 0.91, r"\textbf{Margin: }" + str(margin) + r", \textbf{penalty: }" +str(penalty) + r", \textbf{embedding dim: }" +str(embedding_dim), size=16, transform=ax.transAxes)
    # ax.text(0.03, 0.88, r"\textbf{Average distances:}", size=16, transform=ax.transAxes)
    # ax.text(0.03, 0.84, f"sig-sig {avg_sigsig_dist.item():.3f}, bkg-bkg {avg_bkgbkg_dist.item():.3f}, sig-bkg {avg_sigbkg_dist.item():.3f}", size=16, transform=ax.transAxes)
    # ax.text(0.03, 0.79, r"\textbf{Graph at radius }" + str(radius) + f", edge fraction {edge_frac:.3f}", size=16, transform=ax.transAxes)
    # ax.text(0.03, 0.76, f"Same class: efficiency {eff:.3f}, purity {purity:.3f}", size=16, transform=ax.transAxes)
    # ax.text(0.03, 0.73, f"Sig-sig: efficiency {sigsig_eff:.3f}, purity {sigsig_pur:.3f}", size=16, transform=ax.transAxes)
    # ax.text(0.03, 0.70, f"Bkg-bkg: efficiency {bkgbkg_eff:.3f}, purity {bkgbkg_pur:.3f}", size=16, transform=ax.transAxes)

    ax.set_xlabel('Embedded feature 1', loc="right", fontsize=16)
    ax.set_ylabel('Embedded feature 2', loc="top", fontsize=16)
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    ax.set_xlim((xmin, xmax*1.2))
    ax.set_ylim((ymin, ymax*1.1))
    plotting_path = plot_path+"embedding_"+str(feat)+"feats/"
    os.makedirs(plotting_path, exist_ok=True)
    if num_epochs == 0:
        fig.savefig(plotting_path+"embedding_e0.pdf")
    else:
        fig.savefig(plotting_path+"embedding_e"+str(num_epochs)+"_m"+str(margin)+"_r"+str(radius)+"_Lambda"+str(pen)+".pdf")

    # graph info saved into a dictionary
    graph_dict = {"loss_margin": margin,
                  "loss_penalty": pen,
                  "radius": radius,
                  "embedding_dim": feat,
                  "avg_sigsig_dist": avg_sigsig_dist,
                  "avg_sigbkg_dist": avg_sigbkg_dist,
                  "avg_bkgbkg_dist": avg_bkgbkg_dist,
                  "edge_fraction": edge_frac,
                  "same_class_eff": eff,
                  "same_class_purity": purity,
                  "sigsig_eff": sigsig_eff,
                  "sigsig_purity": sigsig_pur,
                  "bkgbkg_eff": bkgbkg_eff,
                  "bkgbkg_purity": bkgbkg_pur}
    graph_dict = {k: (v.tolist() if isinstance(v, np.ndarray) else float(v) if isinstance(v, (np.float32, np.float64)) else v) 
                          for k, v in graph_dict.items()}

    
    return graph_dict

graph_dict = plot_embeddings(test_embeddings, test_labels, num_epochs, margin, feat=embedding_dim, radius=radius, pen=penalty)

graph_dict_path = model_path + "EmbeddingNet_m"+str(margin)+"_r"+str(radius)+"_Lambda"+str(penalty)+"_dict.json"
with open(graph_dict_path, "w") as dictfile:
    json.dump(graph_dict, dictfile)

if num_epochs > 0:
    fig, ax = plt.subplots()
    x_epoch = np.arange(1,num_epochs+1,1)
    ax.plot(x_epoch, train_loss, color="cornflowerblue", label="Training")
    ax.plot(x_epoch, val_loss, color="darkorange", label="Validation")
    ax.set_xlabel("Epoch", loc="right")
    ax.set_ylabel("Loss", loc="top")
    ax.legend(loc="upper right")
    plotting_path = plot_path+"embedding_"+str(embedding_dim)+"feats/loss/"
    os.makedirs(plotting_path, exist_ok=True)
    os.makedirs(plotting_path, exist_ok=True)
    fig.savefig(plotting_path+"embedding_e"+str(num_epochs)+"_m"+str(margin)+"_r"+str(radius)+"_Lambda"+str(penalty)+".pdf")
