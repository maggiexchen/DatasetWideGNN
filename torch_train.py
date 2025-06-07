"""
Module to train the GCN!
"""
import json
#import glob
#import re
#import os
import argparse
import gc
import logging
import time

#import utils.normalisation as norm
#import utils.torch_distances as dis
import utils.adj_mat as adj
import utils.misc as misc
import utils.performance as perf
import utils.plotting as plotting
import utils.training as training
from utils.gcn_model import GCNClassifier
#from utils.dnn_model import DNNClassifier

#import pandas as pd
import numpy as np
#from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
#import mplhep as hep
import torch
#import torch.nn as nn
#import torch.nn.functional as F
#from torch.utils.data import DataLoader, TensorDataset, Subset
import torch.optim as optim
#from torch.utils.checkpoint import checkpoint
# from torchinfo import summary
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
from sklearn.metrics import roc_curve, auc #, roc_auc_score, precision_recall_curve
#from sklearn.utils import shuffle
#import shap

logging.getLogger().setLevel(logging.INFO)
st = time.time()
torch.cuda.empty_cache()

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
    help="""Specify the config for the user e.g. paths to store all the
            input/output data and results, signal model to look at""",
)

args = parser.parse_args()

### load user config
user_config_path = args.userconfig
user_config = misc.load_config(user_config_path)

print("Using user config ",user_config)
feature_h5_path = user_config["feature_h5_path"]
kinematic_h5_path = user_config["kinematic_h5_path"]
plot_path = user_config["plot_path"]
ll_path = user_config["ll_path"]
adj_path = user_config["adj_path"]
dist_path = user_config["dist_path"]
model_path = user_config["model_path"]
score_path = user_config["score_path"]
cuts = user_config["cuts"]
cutstring = misc.get_cutstring(cuts)

signal = str(user_config["signal"])
signal_mass = user_config["signal_mass"]
feature_dim = user_config["feature_dim"]
assert signal in ["hhh", "LQ", "stau", "embedding"], f"Invalid signal type: {signal}"

### set up CUDA/CPU device settings
run_with_cuda = user_config["run_with_cuda"]
print("CUDA is available? ", torch.cuda.is_available())
cpu = torch.device('cpu')
device = torch.device('cpu')
if torch.cuda.is_available() and run_with_cuda:
    torch.device('cuda')

# set random seed for training
torch.manual_seed(42)

### load training config
train_config_path = args.MLconfig
train_config = misc.load_config(train_config_path)
print("Using training config ",train_config)
hidden_sizes_gcn = train_config["hidden_sizes_gcn"]
do_gnn = True if len(hidden_sizes_gcn) > 0 else False

hidden_sizes_mlp = train_config["hidden_sizes_mlp"]
LR = train_config["LR"]
patience_LR = train_config["patience_LR"]
dropout_rates = train_config["dropout_rates"]
epochs = train_config["epochs"]
num_nb_list = train_config["num_nb_list"]
batch_size = train_config["batch_size"]
gnn_type = train_config["gnn_type"]
patience_early_stopping = train_config["patience_early_stopping"]
num_folds = user_config["n_folds"]
single_fold = train_config["single_fold"]
plot_conv_kins = train_config["plot_conv_kinematics"]
do_edge_wgt = train_config["edge_weights"]

## LR scheduler patience should be less than early stopping patience,
#  so that the LR can be reduced before training stops
assert patience_LR < patience_early_stopping, \
    "LR scheduler patience should be less than early stopping patience"


kinematic_variable = train_config["kinematic_variable"]
embedding_variable = train_config["embedding_variable"]
if kinematic_variable is None:
    raise ValueError("Need to specify a type of kinematic variable in the ML config")

if embedding_variable is None:
    embedding_variable = kinematic_variable

distance = str(train_config["distance"])
if distance is None:
    raise ValueError("Need to specify a distance metric for the adjacency matrix in the ML config")

# TODO resupport target_eff option
linking_length = train_config["linking_length"]
edge_frac_list = [0.1, 0.2, 0.3, 0.4, 0.5]
frac = train_config["edge_frac"]
if linking_length is None:
    if frac is None:
        raise ValueError("Need to specify an edge_frac for the adjacency matrix in the ML config")
    elif frac not in edge_frac_list:
        raise ValueError(f"not given a supported edge fraction, {edge_frac_list}")
    else:
        ll_str = "_LLFrac" + str(frac).replace(".", "p")
        adj_path = adj_path + "/" + distance + "_edge_frac_" + str(frac).replace(".", "p") + "/"
else:
    if frac is not None:
        # when both linking length and edge fraction are specified,
        # use the linking length at specified edge fraction
        ll_str = "_LLFrac" + str(frac).replace(".", "p")
        adj_path = adj_path + "/" + distance + "_edge_frac_" + str(frac).replace(".", "p") + "/"
    else:
        print("linking length is given in config, IGNORING the edge fraction in the config")
        ll_str = "_LL" + str(linking_length).replace(".", "p")
        adj_path = adj_path + "/" + distance + "_linking_length_" + \
            str(linking_length).replace(".", "p") + "/"

### str for train/val split label. If single fold, then val_frac is 1/num_folds.
# Otherwise, nf is num_folds
if single_fold is True:
    val_frac = 1/num_folds
    nf_str = f"_val_frac{val_frac:.2f}"
else:
    nf_str = "_nf" + str(num_folds)

### create model label and result plot path
if len(hidden_sizes_gcn) == 0:
    model_label = signal\
          + "_MLP" + "-".join(map(str, hidden_sizes_mlp)).replace(".", "p")\
          + "_lr" + str(LR).replace(".", "p")\
          + "_dr" + "-".join(map(str, dropout_rates)).replace(".", "p")\
          + "_bs" + str(batch_size)\
          + "_e" + str(epochs)\
          + nf_str
else:
    model_label = signal\
            + f"_{gnn_type}" + "-".join(map(str, hidden_sizes_gcn)).replace(".", "p")\
            + "_MLP" + "-".join(map(str, hidden_sizes_mlp)).replace(".", "p")\
            + "_nb" + "-".join(map(str, num_nb_list))\
            + "_lr" + str(LR).replace(".", "p")\
            + ll_str\
            + "_dr" + "-".join(map(str, dropout_rates)).replace(".", "p")\
            + "_bs" + str(batch_size)\
            + "_e" + str(epochs)\
            + nf_str

kinematic_plot_path = plot_path + "/training_kinematics/"+distance+"_frac" + str(frac) + "/"
if not do_gnn:
    plot_path = plot_path + "/MLP/" + model_label + "/"
plot_path = plot_path + distance + "_models/" + model_label + "/"
misc.create_dirs(plot_path)

if signal == "stau":
    kinematics = misc.get_kinematics_staus(kinematic_variable)
else:
    kinematics = misc.get_kinematics(kinematic_variable)
input_size = len(kinematics)

logging.info("signal: %s", signal)
logging.info("chosen model: %s", model_label)
logging.info("kinematic variable set: %s", kinematic_variable)
logging.info("embedding variable set: %s", embedding_variable)
logging.info("input data path: %s", feature_h5_path)
logging.info("input ll json path: %s", ll_path)
logging.info("input distances path: %s", dist_path)
logging.info("output plot path: %s", plot_path)
logging.info("adj matrix storage path: %s", adj_path)
logging.info("model storage path: %s", model_path)
model_path = model_path + distance + "_models/" + model_label + "/" + gnn_type + "/"

logging.info("distance metric: %s", distance)
if frac is not None:
    logging.info("desired edge fraction: %s", str(frac))
elif linking_length is not None:
    logging.info("linking length: %s", str(linking_length))

# load training data file and kinematics
logging.info('Importing signal and background files...')

# normal loading setup
full_sig, full_bkg, full_x, \
full_sig_wgts, full_bkg_wgts, \
full_sig_labels, full_bkg_labels, \
sig_fold, bkg_fold = adj.data_loader(kinematic_h5_path, kinematics, ex=cutstring,
                                     signal=signal, signal_mass=signal_mass, num_folds=num_folds)

len_sig = len(full_sig)
len_bkg = len(full_bkg)
print("full sig size ", full_sig.size())
print("full bkg size ", full_bkg.size())
print("full sig wgt size ", full_sig_wgts.size())
print("full bkg wgt size ", full_bkg_wgts.size())

full_x = full_x.to(device)
full_y = torch.cat((full_sig_labels, full_bkg_labels), dim=0).to(device)
len_full = len(full_y)
del full_sig, full_bkg, full_sig_labels, full_bkg_labels

print("full sig yields ", full_sig_wgts.sum())
print("full bkg yields ", full_bkg_wgts.sum())
full_wgts = torch.cat((full_sig_wgts, full_bkg_wgts), dim=0).to(device)
print("full wgt size ", full_wgts.size())
del full_sig_wgts, full_bkg_wgts

print("sig_fold count:")
values, counts = np.unique(sig_fold, return_counts=True)
for val, count in zip(values, counts):
    print(f"{val}: {count}")

print("bkg_fold count:")
values, counts = np.unique(bkg_fold, return_counts=True)
for val, count in zip(values, counts):
    print(f"{val}: {count}")

fold_assignment = np.concatenate((sig_fold, bkg_fold), axis=0)

logging.info("Loaded signal and background data.")
logging.info("Time taken so far: %s", str(time.time()-st))

### load edge indices if gnn layers are used
edge_ind = None
edge_wgts = None
if do_gnn:
    print("constructing sparse adjacency matrix ...")
    print("loading row indices ...")
    row_ind = torch.load(adj_path+'row_ind.pt')
    print("loading col indices ...")
    col_ind = torch.load(adj_path+'col_ind.pt')
    print("stacking row and col indices ...")
    edge_ind = torch.stack((row_ind, col_ind)).to(device)
    print("deleting row and col indices ...")
    print(torch.max(row_ind), torch.max(col_ind), torch.max(edge_ind))
    del row_ind
    del col_ind
    print("Edge fraction: ", edge_ind.shape[1] / len(full_y)**2)
    if do_edge_wgt:
        print("loading edge weights ...")
        edge_wgts = torch.load(adj_path+'edge_wgts.pt').to(device)
        # edge weights from MC source node:
        edge_weights_from_MC = full_wgts[edge_ind[0]]
        if gnn_type != "GAT":
            edge_wgts = edge_wgts * edge_weights_from_MC
            edge_weights_from_MC = None


if plot_conv_kins:
    if do_edge_wgt:
        sparse_adj_matrix = torch.sparse_coo_tensor(edge_ind, edge_wgts,
                                                    size=(len(full_y), len(full_y)))
    else:
        edges = torch.ones(edge_ind.shape[1], dtype=torch.float32)
        sparse_adj_matrix = torch.sparse_coo_tensor(edge_ind, edges,
                                                    size=(len(full_y), len(full_y)))
        del edges
### commented out this dense adj for plotting -> takes up too much mem.
# Will have to try and do some kind of subsampling for the plotting purpose.
#    adj_mat = sparse_adj_matrix.to_dense()
#    print("Adjacency Matrix: ", adj_mat)
    del sparse_adj_matrix
#    for nconv in range(3):
#        plotting.plot_conv_kinematics(adj_mat, full_x, len_sig, kinematics,
#                                      signal, frac, kinematic_plot_path,
#                                      normalisation="D_half_inv", standardise=False,
#                                      nconv=nconv, edge_wgts=do_edge_wgt)
#    del adj_mat
misc.print_mem_info()

logging.info("Training ...")
print("full x", len(full_x))
print("full y", len(full_y))
if len(hidden_sizes_gcn) > 0:
    print("Checking edge indices dim: ", len(edge_ind))

gc.collect()
torch.cuda.empty_cache()

### create data object, train and val loaders
if do_edge_wgt and do_gnn:
    data = Data(x=full_x, y=full_y, edge_index=edge_ind,
                node_weight=full_wgts, edge_weight=edge_wgts,
                mc_weight=edge_weights_from_MC if gnn_type == "GAT" else \
                          torch.tensor([], device=full_wgts.device))
    del edge_ind, edge_wgts
else:
    data = Data(x=full_x, y=full_y, edge_index=edge_ind, node_weight=full_wgts)
    del edge_ind

try:
    train_losses = []
    val_losses = []

    train_outputs = torch.tensor([]).to(cpu)
    train_outputs_per_fold = {}
    train_truth_labels = torch.tensor([]).to(cpu)
    train_wgts = torch.tensor([]).to(cpu)
    val_outputs = torch.tensor([]).to(cpu)
    val_outputs_per_fold = {}
    val_truth_labels = torch.tensor([]).to(cpu)
    val_wgts = torch.tensor([]).to(cpu)

    logging.info("Starting k-fold cross validation ...")
    logging.info("Time taken so far: %s", str(time.time()-st))

    for fold_no in range(num_folds):

        train_idx = np.where(fold_assignment != fold_no)[0]
        val_idx = np.where(fold_assignment == fold_no)[0]

        print("starting fold %s/%s", fold_no+1, num_folds)
        print("train idx", len(train_idx))
        print("val idx", len(val_idx))

        ### standardise input data to training set and move to cpu after standardisation
        means, stds = misc.get_train_mean_std(full_x[train_idx])
        data_standardised = data.clone()
        data_standardised.x = misc.torch_standardise(data_standardised.x, means, stds)
        means, stds = means.to(cpu), stds.to(cpu)
        model = GCNClassifier(input_size=input_size, hidden_sizes_gcn=hidden_sizes_gcn,
                              hidden_sizes_mlp=hidden_sizes_mlp, output_size=1,
                              dropout_rates=dropout_rates, gnn_type=gnn_type)
        model = model.to(device)

        optimiser = torch.optim.Adam(model.parameters(), lr=LR)
        ### NOTE: patience for the scheculer is different from the early stopping patience
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimiser, mode='min',
                                                         patience=patience_LR)

        train_loss = []
        val_loss = []

        all_labels = data_standardised.y[train_idx].cpu().numpy()
        all_node_weights = data_standardised.node_weight[train_idx].cpu().numpy()
        class_weights = training.binary_class_weights(all_labels, all_node_weights).to(device)
        print("Training class weights: ")
        print("         signal: ", class_weights[1])
        print("         backgrounds: ", class_weights[0])
        if do_gnn:
            logging.info("Graph sub-sampling for training and validation ...")
        else:
            logging.info("Loading for training and validation ...")

        train_loader = NeighborLoader(
            data_standardised,
            input_nodes=train_idx,
            num_neighbors=num_nb_list,
            shuffle=True,
            batch_size=batch_size,
        )
        val_loader = NeighborLoader(
            data_standardised,
            input_nodes=val_idx,
            num_neighbors=num_nb_list,
            shuffle=False,
            batch_size=batch_size,
        )

        best_val_loss = float('inf')
        patience_counter = 0
        logging.info("Starting training ...")
        for epoch in range(epochs):

            ### start training loop in the epoch
            model.train()
            total_examples = 0
            total_loss = 0
            train_outputs_fold = torch.tensor([]).to(cpu)
            train_truth_labels_fold = torch.tensor([]).to(cpu)
            train_wgts_fold = torch.tensor([]).to(cpu)
            train_x_fold = torch.tensor([])
            for batch in train_loader:
                optimiser.zero_grad()
                batch = batch.to(device)
                tmp_batch_size = batch.batch_size
                if do_edge_wgt and do_gnn:
                    outputs = model(batch.x, batch.edge_index, batch.edge_weight, batch.mc_weight)
                else:
                    outputs = model(batch.x, batch.edge_index)

                ### NOTE only consider predictions and labels of seed nodes
                y = batch.y[:tmp_batch_size]
                outputs = outputs[:tmp_batch_size]
                event_wgts = batch.node_weight[:tmp_batch_size]

                loss = training.weighted_bce_loss(outputs.squeeze(), y.squeeze().float(),
                                                  class_weights, event_wgts)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimiser.step()

                torch.cuda.empty_cache()
                total_examples += tmp_batch_size
                total_loss += float(loss) * tmp_batch_size
                train_outputs_fold = torch.cat((train_outputs_fold, outputs.detach().to(cpu)))
                train_truth_labels_fold = torch.cat((train_truth_labels_fold, y.detach().to(cpu)))
                train_wgts_fold = torch.cat((train_wgts_fold, event_wgts.detach().to(cpu)))
                train_x_fold = torch.cat((train_x_fold, batch.x.detach().to(cpu)))

            avg_tr_loss = total_loss / total_examples
            train_loss.append(avg_tr_loss)

            ### start validation loop in the epoch
            model.eval()
            total_examples = total_loss = 0
            val_outputs_fold= torch.tensor([]).to(cpu)
            val_truth_labels_fold = torch.tensor([]).to(cpu)
            val_wgts_fold = torch.tensor([]).to(cpu)
            val_x_fold = torch.tensor([]).to(cpu)
            for batch in val_loader:

                batch = batch.to(device)
                tmp_batch_size = batch.batch_size
                if do_edge_wgt and do_gnn:
                    outputs = model(batch.x, batch.edge_index, batch.edge_weight, batch.mc_weight)
                else:
                    outputs = model(batch.x, batch.edge_index, gnn_type)

                ### NOTE only consider predictions and labels of seed nodes
                y = batch.y[:tmp_batch_size]
                outputs = outputs[:tmp_batch_size]
                event_wgts = batch.node_weight[:tmp_batch_size]

                loss = training.weighted_bce_loss(outputs.squeeze(), y.squeeze().float(),
                                                  class_weights, event_wgts)

                total_examples += tmp_batch_size
                total_loss += float(loss) * tmp_batch_size
                val_outputs_fold = torch.cat((val_outputs_fold, outputs.detach().to(cpu)))
                val_truth_labels_fold = torch.cat((val_truth_labels_fold, y.detach().to(cpu)))
                val_wgts_fold = torch.cat((val_wgts_fold, event_wgts.detach().to(cpu)))
                val_x_fold = torch.cat((val_x_fold, batch.x.detach().to(cpu)))


            avg_vl_loss = total_loss / total_examples
            val_loss.append(avg_vl_loss)

            current_lr = optimiser.param_groups[0]['lr']
            scheduler.step(avg_vl_loss)
            new_lr = optimiser.param_groups[0]['lr']
            if new_lr < current_lr:
                print(f"Learning rate reduced to: {new_lr}")

            if avg_vl_loss < best_val_loss:
                best_val_loss = avg_vl_loss
                patience_counter = 0
            else:
                patience_counter += 1
                print(f"No improvement in validation loss for {patience_counter} epoch(s).")

            print(f'Epoch {epoch + 1}/{epochs}, Train Loss: {avg_tr_loss:.6f}, \
                  Validation Loss: {avg_vl_loss:.6f}')

            if patience_counter >= patience_early_stopping:
                print(f"Early stopping after {epoch+1} epochs.")
                break

        train_outputs_per_fold["fold_"+str(fold_no+1)+"_outputs"] = train_outputs_fold.flatten()
        val_outputs_per_fold["fold_"+str(fold_no+1)+"_outputs"] = val_outputs_fold.flatten()

        logging.info("Finished fold %s/%s", fold_no, num_folds)
        logging.info("Number of epochs: %s/%s", str(epoch+1), str(epochs))
        logging.info("Final train Loss: %s", avg_tr_loss)
        logging.info("Final validation Loss: %s", avg_vl_loss)
        logging.info("Time taken so far: %s", str(time.time()-st))
        logging.info("Saving trained model and performance...")
        model_file_name = f"model_fold_{fold_no}.pth"

        misc.create_dirs(model_path)
        torch.save({
            'model_state': model.state_dict(),
            'optimiser_state': optimiser.state_dict(),
            'normalisation_params': {"means": means, "stds": stds}
        }, model_path+model_file_name)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_outputs = torch.cat((train_outputs, train_outputs_fold))
        train_truth_labels = torch.cat((train_truth_labels, train_truth_labels_fold))
        train_wgts = torch.cat((train_wgts, train_wgts_fold))
        val_outputs = torch.cat((val_outputs, val_outputs_fold))
        val_truth_labels = torch.cat((val_truth_labels, val_truth_labels_fold))
        val_wgts = torch.cat((val_wgts, val_wgts_fold))
        del train_loader, val_loader, model, optimiser, scheduler
        del train_outputs_fold, val_outputs_fold, train_truth_labels_fold, val_truth_labels_fold
        torch.cuda.empty_cache()
        gc.collect()

        if single_fold is True:
            print("Single fold training, breaking loop ...")
            break

    print("plotting model outputs per fold")
    fig_fold, ax_fold = plt.subplots()
    fold_colours = ["steelblue", "darkorange", "forestgreen"]
    for k in range(num_folds):
        print(f"Training Fold {str(k+1)}", train_outputs_per_fold["fold_"+str(k+1)+"_outputs"])
        print(f"Validation Fold {str(k+1)}", val_outputs_per_fold["fold_"+str(k+1)+"_outputs"])
        fig_fold, ax_fold = plt.subplots()
        hist, binning, _ = ax_fold.hist(train_outputs_per_fold["fold_"+str(k+1)+"_outputs"],
                                        bins=40, label="Train fold "+str(k), histtype='step',
                                        linestyle='--', density=True, color=fold_colours[k])
        ax_fold.hist(val_outputs_per_fold["fold_"+str(k+1)+"_outputs"], bins=binning,
                     label="Val fold "+str(k), alpha=0.5,
                     density=True, color=fold_colours[k])
        plotting.draw_labels_legends(ax_fold, "Training output score", "Normalised # Events",
                                     log_y=True)
        plotting.save_fig(fig_fold, plot_path+"outputs_fold_"+str(k))

finally:
    logging.info("Training complete.")
    print("train truth labels", len(train_truth_labels))
    print("val truth labels", len(val_truth_labels))

    ### compute ROC curve and AUC
    train_outputs = train_outputs.view(-1)
    train_label_bool = train_truth_labels.bool()
    train_sig_pred = train_outputs[train_label_bool]
    train_sig_wgts = train_wgts[train_label_bool]
    train_bkg_pred = train_outputs[torch.logical_not(train_label_bool)]
    train_bkg_wgts = train_wgts[torch.logical_not(train_label_bool)]

    train_fpr, train_tpr, train_cut = roc_curve(train_truth_labels.detach().cpu().numpy(),
                                                train_outputs.detach().cpu().numpy(),
                                                sample_weight=train_wgts.detach().cpu().numpy())
    if signal == "stau":
        # stau fpr needs to be clipped and sorted due to rounding errors
        train_fpr = np.clip(train_fpr, 0, 1)
        train_fpr = np.sort(train_fpr)
    train_auc = auc(train_fpr, train_tpr)
    print("Training AUC", train_auc)

    val_outputs = val_outputs.view(-1)
    val_label_bool = val_truth_labels.bool()
    val_sig_pred = val_outputs[val_label_bool]
    val_sig_wgts = val_wgts[val_label_bool]
    val_bkg_pred = val_outputs[torch.logical_not(val_label_bool)]
    val_bkg_wgts = val_wgts[torch.logical_not(val_label_bool)]

    val_fpr, val_tpr, val_cut = roc_curve(val_truth_labels.detach().cpu().numpy(),
                                          val_outputs.detach().cpu().numpy(),
                                          sample_weight=val_wgts.detach().cpu().numpy())
    if signal == "stau": ### stau fpr needs to be clipped and sorted due to rounding errors
        val_fpr = np.clip(val_fpr, 0, 1)
        val_fpr = np.sort(val_fpr)
    val_auc = auc(val_fpr, val_tpr)
    print("Validation AUC", val_auc)

    # save performance to json
    perf.save_performance(train_loss, train_fpr, train_tpr, train_cut, train_auc,
                          val_loss, val_fpr, val_tpr, val_cut, val_auc, model_path)
    perf.save_metadata_kfold(len(val_sig_pred), len(val_bkg_pred), num_folds,
                             hidden_sizes_gcn, hidden_sizes_mlp, LR, dropout_rates,
                             epochs, model_path)

    logging.info("Plotting training/validation losses ...")
    fig_loss, ax_loss = plt.subplots()
    x_epoch = np.arange(1,epochs+1,1)
    for loss_loop, (train_loss, val_loss) in enumerate(zip(train_losses, val_losses)):
        train_line, = ax_loss.plot(np.arange(len(train_loss)), train_loss,
                                   label="Fold " + str(loss_loop) + " (Train)")
        colour = train_line.get_color()
        ax_loss.plot(np.arange(len(val_loss)), val_loss,
                     label="Fold " + str(loss_loop) + " (Val)", color=colour, linestyle="-.")
    ax_loss.legend(loc='upper right', fontsize=9)
    if do_gnn:
        model_text = [str(gnn_type)+" model",
                      "GNN layers "+ str(hidden_sizes_gcn),
                      "MLP layers "+ str(hidden_sizes_mlp),
                      "Batchsize " + str(batch_size), 
                      "Neighbour sampling " + str(num_nb_list)]
    else:
        model_text = [str(gnn_type) + " model",
                      "MLP layers "+ str(hidden_sizes_mlp),
                      "Batchsize " + str(batch_size)]
    plotting.add_text(ax_loss, model_text, do_atlas=False, startx=0.02, starty=0.95)

    ymin, ymax = ax_loss.get_ylim()
    plotting.draw_labels_legends(ax_loss, "Epoch", "Loss", yrange=[ymin, ymax*1.2])

    misc.create_dirs(plot_path)
    logging.info("Saving plots to %s", plot_path)
    plotting.save_fig(fig_loss, plot_path+"training_validation_loss")

    logging.info("Plotting model outputs ...")
    linking_length_label = ""
    if do_gnn:
        if frac is not None:
            linking_length_label = "Linking length at "+str(frac)+" edge fraction"
        elif linking_length is not None:
            linking_length_label = "Linking length "+str(linking_length)
    signal_label, background_label = plotting.get_plot_labels(signal, signal_mass)
    text = [f"Training AUC = {train_auc:.3f}", f"Validation AUC = {val_auc:.3f}",
            signal_label, background_label, linking_length_label]

    fig_pred, ax_pred = plt.subplots()
    binning = np.linspace(0,1,51)
    ax_pred.hist(train_sig_pred.detach().cpu().numpy(), bins=binning,
                 label="Signal (training)", histtype='step', linestyle='--', density=True,
                 color="darkorange", weights=train_sig_wgts.detach().cpu().numpy())
    ax_pred.hist(train_bkg_pred.detach().cpu().numpy(), bins=binning,
                 label="Background (training)", histtype='step', linestyle='--', density=True,
                 color="steelblue", weights=train_bkg_wgts.detach().cpu().numpy())
    ax_pred.hist(val_sig_pred.detach().cpu().numpy(), bins=binning,
                 label="Signal (validation)", alpha=0.5, density=True,
                 color="darkorange", weights=val_sig_wgts.detach().cpu().numpy())
    ax_pred.hist(val_bkg_pred.detach().cpu().numpy(), bins=binning,
                 label="Background (validation)", alpha=0.5, density=True,
                 color="steelblue", weights=val_bkg_wgts.detach().cpu().numpy())
    plotting.add_text(ax_pred, text, do_atlas=False, startx=0.02, starty=0.95)
    ymin, ymax = ax_pred.get_ylim()
    plotting.draw_labels_legends(ax_pred, "Output score", "Normalised # Events",
                                 yrange=[ymin, ymax], log_y=True)
    plotting.save_fig(fig_pred, plot_path+"training_validation_pred")

    score_path = score_path + model_label + "/"
    misc.create_dirs(score_path)

    np.save(score_path+"train_sig_pred.npy", train_sig_pred.detach().cpu().numpy())
    np.save(score_path+"train_sig_wgts.npy", train_sig_wgts.detach().cpu().numpy())

    np.save(score_path+"train_bkg_pred.npy", train_bkg_pred.detach().cpu().numpy())
    np.save(score_path+"train_bkg_wgts.npy", train_bkg_wgts.detach().cpu().numpy())

    np.save(score_path+"val_sig_pred.npy", val_sig_pred.detach().cpu().numpy())
    np.save(score_path+"val_sig_wgts.npy", val_sig_wgts.detach().cpu().numpy())

    np.save(score_path+"val_bkg_pred.npy", val_bkg_pred.detach().cpu().numpy())
    np.save(score_path+"val_bkg_wgts.npy", val_bkg_wgts.detach().cpu().numpy())

    logging.info("Plotting ROC curves ...")
    fig_roc, ax_roc = plt.subplots()
    plt.plot(train_fpr, train_tpr, label=f'Training ROC curve (AUC = {train_auc:.3f})')
    plt.plot(val_fpr, val_tpr, label=f'Validation ROC curve (AUC = {val_auc:.3f})')
    plt.xlim(0,1)
    plotting.add_text(ax_roc, model_text, do_atlas=False, startx=0.02, starty=0.2)
    plotting.draw_labels_legends(ax_roc, "Background Efficiency", "Signal Efficiency",
                                 legendloc="upper left", yrange=[0., 1.2], log_y=False)
    plotting.save_fig(fig_roc, plot_path+"training_validation_ROC")

    logging.info("Saving ROC curves to json files ...")
    roc_json_path = plot_path+"roc.json"
    roc_dict = {"train_fpr": train_fpr.tolist(),
                "train_tpr": train_tpr.tolist(),
                "val_fpr": val_fpr.tolist(),
                "val_trp": val_tpr.tolist(),
                "train_auc": [train_auc],
                "val_auc": [val_auc]
               }
    with open(roc_json_path, 'w', encoding="utf-8") as json_file:
        json.dump(roc_dict, json_file)
