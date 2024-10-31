import pandas as pd
import numpy
import json
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
import torch.optim as optim
from torch.utils.checkpoint import checkpoint
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
torch.manual_seed(42)

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

### load user config
user_config_path = args.userconfig
user_config = misc.load_config(user_config_path)
print("Using user config ",user_config)
h5_path = user_config["h5_path"]
plot_path = user_config["plot_path"]
ll_path = user_config["ll_path"]
adj_path = user_config["adj_path"]
dist_path = user_config["dist_path"]
model_path = user_config["model_path"]
score_path = user_config["score_path"]

signal = user_config["signal"]
h5_path = h5_path + "/" + signal + "_split_files/"
assert signal in ["hhh", "LQ", "stau"], f"Invalid signal type: {signal}"
signal_label, background_label = plotting.get_plot_labels(signal)

### load training config 
train_config_path = args.MLconfig
train_config = misc.load_config(train_config_path)
print("Using training config ",train_config)
hidden_sizes_gcn = train_config["hidden_sizes_gcn"]
hidden_sizes_mlp = train_config["hidden_sizes_mlp"]
LR = train_config["LR"]
patience_LR = train_config["patience_LR"]
dropout_rates = train_config["dropout_rates"]
epochs = train_config["epochs"]
num_nb_list = train_config["num_nb_list"] 
batch_size = train_config["batch_size"]
gnn_type = train_config["gnn_type"]
patience_early_stopping = train_config["patience_early_stopping"]
### LR scheduler patience should be less than early stopping patience, so that the LR can be reduced before training stops
assert patience_LR < patience_early_stopping, "LR scheduler patience should be less than early stopping patience"

variable = train_config["variable"]
if variable is None:
    print("Need to specify a type of kinematic variable in the config")

distance = train_config["distance"]
if distance is None:
    print("Need to specify a type of distance metric for the adjacency matrix in the config")

linking_length = train_config["linking_length"]
eff = train_config["sigsig_eff"]
if linking_length is None:
    if eff is None:
        raise Exception("Need to specify a sig-sig efficiency for the adjacency matrix when training a gcn in the config")
    elif eff not in [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        raise Exception("not given a supported efficiency, (0.4, 0.5, 0.6, 0.7, 0.8, 0.9)")
    else:
        ll_str = "_LLEff" + str(eff).replace(".", "p")
        adj_path = adj_path + "/" + f"sigsig_eff_{eff}/"
else:
    if eff is not None:
        # when both linking length and sigsig eff are specified, use the linking length at specified sigsig efficiency
        ll_str = "_LLEff" + str(eff).replace(".", "p")
        adj_path = adj_path + "/" + f"sigsig_eff_{eff}/"
    else:
        print("linking length is given in config, IGNORING the sigsig_eff in the config!")
        ll_str = "_LL" + str(linking_length).replace(".", "p")
        adj_path = adj_path + "/" + f"linking_length_{linking_length}/"

### create model label and result plot path
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
plot_path = plot_path + model_label + "/"
misc.create_dirs(plot_path)

if signal == "stau":
    kinematics = misc.get_kinematics_staus(variable)
else:
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

logging.info("distance metric: "+distance)
if eff is not None:
    logging.info("desired efficieny: "+str(eff))
elif linking_length is not None:
    logging.info("linking length: "+str(linking_length))

# load training data file and kinematics
logging.info('Importing signal and background files...')

# normalised signal and background kinematics
train_sig, train_bkg, train_x, train_sig_wgts, train_bkg_wgts, train_sig_labels, train_bkg_labels = adj.data_loader(h5_path, plot_path, "train", kinematics, plot=False, signal=signal)
val_sig, val_bkg, val_x, val_sig_wgts, val_bkg_wgts, val_sig_labels, val_bkg_labels = adj.data_loader(h5_path, plot_path, "val", kinematics, plot=False, signal=signal)
test_sig, test_bkg, test_x, test_sig_wgts, test_bkg_wgts, test_sig_labels, test_bkg_labels = adj.data_loader(h5_path, plot_path, "test", kinematics, plot=False, signal=signal)

print("train sig", len(train_sig))
print("train bkg", len(train_bkg))
print("val sig", len(val_sig))
print("val bkg", len(val_bkg))
print("test sig", len(test_sig))
print("test bkg", len(test_bkg))


full_sig = torch.cat((train_sig, val_sig, test_sig), dim=0)
full_sig_labels = torch.cat((train_sig_labels, val_sig_labels, test_sig_labels))

full_bkg = torch.cat((train_bkg, val_bkg, test_bkg), dim=0)
full_bkg_labels = torch.cat((train_bkg_labels, val_bkg_labels, test_bkg_labels))

# raw_full_sig = torch.cat((raw_train_sig, raw_val_sig), dim=0)
# raw_full_bkg = torch.cat((raw_train_bkg, raw_val_bkg), dim=0)

full_x = torch.cat((full_sig, full_bkg), dim=0).to(device)
del full_sig
del full_bkg

full_y = torch.cat((full_sig_labels, full_bkg_labels), dim=0).to(device)
full_y = full_y.float()
full_wgts = torch.cat((torch.cat((train_sig_wgts, val_sig_wgts, test_sig_wgts), dim=0), torch.cat((train_bkg_wgts, val_bkg_wgts, test_bkg_wgts), dim=0)), dim=0)#.cuda()

### load edge indices if gnn layers are used
edge_ind = None
if len(hidden_sizes_gcn) > 0:
    print("constructing sparse adjacency matrix ...")
    print("loading row indices ...")
    row_ind = torch.load(adj_path+'row_ind.pt')
    print("loading col indices ...")
    col_ind = torch.load(adj_path+'col_ind.pt')
    print("stacking row and col indices ...")
    edge_ind = torch.stack((row_ind, col_ind)).type(torch.int64).to(device)
    print("deleting row and col indices ...")
    del row_ind
    del col_ind
    # print("loading edge weights ...")
    # edge_wgts = full_wgts[row_ind]

misc.print_mem_info()

# def compute_class_weights(labels):
#     '''
#     Compute class weights for binary classification
#     '''
#     labels = labels.astype(int)  # Ensure labels are integers
#     class_counts = np.bincount(labels)
#     class_weights = 1.0 / class_counts
#     class_weights = class_weights / class_weights.sum()  # Normalize to sum to 1
#     return torch.tensor(class_weights, dtype=torch.float)

def compute_class_weights(labels, event_weights):
    '''
    Compute weighted class weights for binary classification with event weights
    
    Args:
        labels (np.ndarray): Array of labels.
        event_weights (np.ndarray): Array of event weights corresponding to each label.
    
    Returns:
        torch.Tensor: Tensor of class weights.
    '''
    labels = labels.astype(int)  # Ensure labels are integers
    unique_labels = np.unique(labels)

    # Initialize an array to hold the weighted count for each class
    weighted_counts = np.zeros(len(unique_labels), dtype=float)

    # Accumulate the weighted counts for each class
    for label in unique_labels:
        weighted_counts[label] = np.sum(event_weights[labels == label])
    
    # Compute the class weights as the inverse of the weighted counts
    class_weights = 1.0 / weighted_counts
    class_weights = class_weights / class_weights.sum()  # Normalize to sum to 1

    return torch.tensor(class_weights, dtype=torch.float)

### define loss function and optimiser
def weighted_bce_loss(output, target, class_weights, event_weights):
    sig_loss = event_weights * (class_weights[1] * target * torch.log(output+1e-10))
    bkg_loss = event_weights * (class_weights[0] * (1-target) * torch.log(1-output+1e-10))
    loss = sig_loss+bkg_loss
    return -loss.mean()

optimiser = torch.optim.Adam(model.parameters(), lr=LR)
### NOTE: patience for the scheculer is different from the early stopping patience
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimiser, mode = 'min', patience = patience_LR)

def binary_class_weights(labels, event_weights):
    num_sig = np.sum(event_weights[labels == 1])
    num_bkg = np.sum(event_weights[labels == 0])
    # bkg_weight = num_sig/num_bkg
    # sig_weight = 1
    bkg_weight = 1
    sig_weight = num_bkg/num_sig
    return torch.tensor([bkg_weight, sig_weight], dtype=torch.float)

logging.info("Training ...")
print("full x", len(full_x))
print("full y", len(full_y))
if len(hidden_sizes_gcn) > 0:
    print("edge ind", len(edge_ind))

gc.collect()
torch.cuda.empty_cache()

### create data object, train and val loaders
data = Data(x = full_x, y = full_y, edge_index = edge_ind, wgts = full_wgts)#, edge_weight = edge_wgts)
del edge_ind

train_idx = torch.cat((torch.arange(len(train_sig)), torch.arange(len(train_sig)+len(val_sig), len(train_sig)+len(val_sig)+len(train_bkg))), dim=0).tolist()
val_idx = torch.cat((torch.arange(len(train_sig), len(train_sig)+len(val_sig)), torch.arange(len(full_x)-len(val_bkg), len(full_x))), dim=0).tolist()
print("train idx", len(train_idx))
print("val idx", len(val_idx))

all_labels = data.y[train_idx].cpu().numpy()
all_wgts = data.wgts[train_idx].cpu().numpy()
# class_weights = compute_class_weights(all_labels, all_wgts).to(device)
class_weights = binary_class_weights(all_labels, all_wgts).to(device)
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
    shuffle = False,
    batch_size = batch_size,
    # num_workers = 6,
    # persistent_workers = True
)

# BCE_loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([sig_class_weight]))
best_val_loss = float('inf')
patience_counter = 0
logging.info("Starting training ...")
for epoch in range(epochs):

    ### start training loop in the epoch
    model.train()
    total_examples = total_loss = 0
    train_outputs = torch.tensor([])
    train_truth_labels = torch.tensor([])
    train_wgts = torch.tensor([])
    for batch in train_loader:

        optimiser.zero_grad()
        batch = batch.to(device)
        batch_size = batch.batch_size
        outputs = model(batch.x, batch.edge_index) #, batch.edge_weight)
        
        ### NOTE only consider predictions and labels of seed nodes
        y = batch.y[:batch_size]
        outputs = outputs[:batch_size]
        event_wgts = batch.wgts[:batch_size]

        loss = weighted_bce_loss(outputs.squeeze(), y.squeeze().float(), class_weights, event_wgts) 
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimiser.step()

        torch.cuda.empty_cache()
        total_examples += batch_size
        total_loss += float(loss) * batch_size
        train_outputs = torch.cat((train_outputs, outputs.detach()))
        train_truth_labels = torch.cat((train_truth_labels, y.detach()))
        train_wgts = torch.cat((train_wgts, event_wgts.detach()))
        
    avg_tr_loss = total_loss / total_examples
    train_loss.append(avg_tr_loss)
        
    ### start validation loop in the epoch
    model.eval()
    total_examples = total_loss = 0
    val_outputs = torch.tensor([])
    val_truth_labels = torch.tensor([])
    val_wgts = torch.tensor([])
    for batch in val_loader:
        
        batch = batch.to(device)
        batch_size = batch.batch_size
        outputs = model(batch.x, batch.edge_index) #, batch.edge_weight)

        ### NOTE only consider predictions and labels of seed nodes
        y = batch.y[:batch_size]
        outputs = outputs[:batch_size]
        event_wgts = batch.wgts[:batch_size]

        loss = weighted_bce_loss(outputs.squeeze(), y.squeeze().float(), class_weights, event_wgts)
        
        total_examples += batch_size
        total_loss += float(loss) * batch_size
        val_outputs = torch.cat((val_outputs, outputs.detach()))
        val_truth_labels = torch.cat((val_truth_labels, y.detach()))
        val_wgts = torch.cat((val_wgts, event_wgts.detach()))

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

    print(f'Epoch {epoch + 1}/{epochs}, Train Loss: {avg_tr_loss}, Validation Loss: {avg_vl_loss}')

    if patience_counter >= patience_early_stopping:
        print(f"Early stopping after {epoch+1} epochs.")
        break

        
    
logging.info("Training complete.")
print("train truth labels", len(train_truth_labels))
print("val truth labels", len(val_truth_labels))

logging.info("Saving trained model and performance...")
model_file_name = "model.pth"
model_path = model_path+model_label+"/"
misc.create_dirs(model_path)
torch.save({
    'model_state': model.state_dict(),
    'optimiser_state': optimiser.state_dict(),
}, model_path+model_file_name)

### compute ROC curve and AUC
train_outputs = train_outputs.view(-1).to(device)
train_label_bool = train_truth_labels.bool()
train_sig_pred = train_outputs[train_label_bool]
train_sig_wgts = train_wgts[train_label_bool]
train_bkg_pred = train_outputs[torch.logical_not(train_label_bool)]
train_bkg_wgts = train_wgts[torch.logical_not(train_label_bool)]

train_fpr, train_tpr, train_cut = roc_curve(train_truth_labels.detach().cpu().numpy(), train_outputs.detach().cpu().numpy(), sample_weight = train_wgts.detach().cpu().numpy())
if signal == "stau": ### stau fpr needs to be clipped and sorted due to rounding errors
    train_fpr = np.clip(train_fpr, 0, 1)
    train_fpr = np.sort(train_fpr)
train_auc = auc(train_fpr, train_tpr)
# train_auc = roc_auc_score(train_truth_labels.detach().cpu().numpy(), train_outputs.detach().cpu().numpy(), sample_weight = train_wgts.detach().cpu().numpy())
print("Training AUC", train_auc)

val_outputs = val_outputs.view(-1).to(device)
val_label_bool = val_truth_labels.bool()
val_sig_pred = val_outputs[val_label_bool]
val_sig_wgts = val_wgts[val_label_bool]
val_bkg_pred = val_outputs[torch.logical_not(val_label_bool)]
val_bkg_wgts = val_wgts[torch.logical_not(val_label_bool)]

val_fpr, val_tpr, val_cut = roc_curve(val_truth_labels.detach().cpu().numpy(), val_outputs.detach().cpu().numpy(), sample_weight = val_wgts.detach().cpu().numpy())
if signal == "stau": ### stau fpr needs to be clipped and sorted due to rounding errors
    val_fpr = np.clip(val_fpr, 0, 1)
    val_fpr = np.sort(val_fpr)
val_auc = auc(val_fpr, val_tpr)
#  val_auc = roc_auc_score(val_truth_labels.detach().cpu().numpy(), val_outputs.detach().cpu().numpy(), sample_weight = val_wgts.detach().cpu().numpy())
print("Validation AUC", val_auc)

# save performance to json
perf.save_performance(train_loss, train_fpr, train_tpr, train_cut, train_auc, val_loss, val_fpr, val_tpr, val_cut, val_auc, model_path)
perf.save_metadata(len(train_sig), len(train_bkg), len(val_sig), len(val_bkg), hidden_sizes_gcn, hidden_sizes_mlp, LR, dropout_rates, epochs, model_path)

logging.info("Plotting training/validation losses ...")
fig, ax = plt.subplots()
x_epoch = numpy.arange(1,epochs+1,1)
ax.plot(np.arange(len(train_loss)), train_loss, label="Training loss")
ax.plot(np.arange(len(val_loss)), val_loss, label="Validation loss")
ax.legend(loc='upper right', fontsize=9)
ax.text(0.02, 0.95, model_label, verticalalignment="bottom", size=9, transform=ax.transAxes)
ax.set_xlabel("Epoch", loc="right")
ax.set_ylabel("Loss", loc="top")
misc.create_dirs(plot_path)
logging.info("Saving plots to "+plot_path)
fig.savefig(plot_path+variable+"_"+model_label+"_training_validation_loss.pdf", transparent=True)

logging.info("Plotting model outputs ...")
fig, ax = plt.subplots()
binning = np.linspace(0,1,51)

# ### save histograms values
# train_sig_pred_hist, bin_edges = np.histogram(train_sig_pred.detach().cpu().numpy(), bins=binning, density=True, range = (0,1))
# train_bkg_pred_hist, _ = np.histogram(train_bkg_pred.detach().cpu().numpy(), bins=binning, density=True, range = (0,1))
# val_sig_pred_hist, _ = np.histogram(val_sig_pred.detach().cpu().numpy(), bins=binning, density=True, range = (0,1))
# val_bkg_pred_hist, _ = np.histogram(val_bkg_pred.detach().cpu().numpy(), bins=binning, density=True, range = (0,1))
# np.save(plot_path+"bin_edges.npy", bin_edges)
# np.save(plot_path+"train_sig_pred_hist.npy", train_sig_pred_hist)
# np.save(plot_path+"train_bkg_pred_hist.npy", train_bkg_pred_hist)
# np.save(plot_path+"val_sig_pred_hist.npy", val_sig_pred_hist)
# np.save(plot_path+"val_bkg_pred_hist.npy", val_bkg_pred_hist)

ax.hist(train_sig_pred.detach().cpu().numpy(), bins=binning, label="Signal (training)", histtype='step', linestyle='--', density=True, color="darkorange", weights=train_sig_wgts.detach().cpu().numpy())
ax.hist(train_bkg_pred.detach().cpu().numpy(), bins=binning, label="Background (training)", histtype='step', linestyle='--', density=True, color="steelblue", weights=train_bkg_wgts.detach().cpu().numpy())
ax.hist(val_sig_pred.detach().cpu().numpy(), bins=binning, label="Signal (validation)", alpha=0.5, density=True, color="darkorange", weights=val_sig_wgts.detach().cpu().numpy())
ax.hist(val_bkg_pred.detach().cpu().numpy(), bins=binning, label="Background (validation)", alpha=0.5, density=True, color="steelblue", weights=val_bkg_wgts.detach().cpu().numpy())

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

if signal == "hhh":
    if eff is not None:
        text = ["Training AUC = {:.3f}".format(train_auc), "Validation AUC = {:.3f}".format(val_auc), "6b Resonant TRSM signal, 5b data", "Linking length at sig-sig efficiency "+str(eff)]
    elif linking_length is not None:
        text = ["Training AUC = {:.3f}".format(train_auc), "Validation AUC = {:.3f}".format(val_auc), "6b Resonant TRSM signal, 5b data", "Linking length "+str(linking_length)]
elif signal == "stau":
    if eff is not None:
        text = ["Training AUC = {:.3f}".format(train_auc), "Validation AUC = {:.3f}".format(val_auc), "stau stau signal", "Linking length at sig-sig efficiency "+str(eff)]
    elif linking_length is not None:
        text = ["Training AUC = {:.3f}".format(train_auc), "Validation AUC = {:.3f}".format(val_auc), "stau stau signal", "Linking length "+str(linking_length)]
elif signal == "LQ":
    if eff is not None:
        text = ["Training AUC = {:.3f}".format(train_auc), "Validation AUC = {:.3f}".format(val_auc), "LQ signal", "Linking length at sig-sig efficiency "+str(eff)]
    elif linking_length is not None:
        text = ["Training AUC = {:.3f}".format(train_auc), "Validation AUC = {:.3f}".format(val_auc), "LQ signal", "Linking length "+str(linking_length)]

plotting.add_text(ax, text, doATLAS=False, startx=0.02, starty=0.95)
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
