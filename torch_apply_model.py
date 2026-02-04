import numpy as np
import matplotlib.pyplot as plt
import torch
import argparse
import utils.misc as misc
import utils.plotting as plotting
import utils.adj_mat as adj
from utils.gcn_model import GCNClassifier
import utils.user_config as uconfig
import utils.ml_config as mlconfig
import json
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
import time
st = time.time()
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, auc
import gc
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
        help="""Specify the config for the user e.g. paths to store all the input/output data and
        results, signal model to look at""",
    )

    return parser

parser = GetParser()
args = parser.parse_args()

### load user config
user_config_path = args.userconfig
user = uconfig.UserConfig.from_yaml(user_config_path)

### load training config
ml_config_path = args.MLconfig
ml = mlconfig.MLConfig.from_yaml(ml_config_path)

signal_label, background_label = plotting.get_plot_labels(user.signal, user.signal_mass)

### set up CUDA/CPU device settings
logging.info("CUDA is available? %s", str(torch.cuda.is_available()))
if torch.cuda.is_available() and user.run_with_cuda:
    logging.info("      Using cuda")
    device = torch.device('cuda')
    cpu = torch.device('cpu')
else:
    logging.info("      Using cpu")
    device = torch.device('cpu')
    cpu = device

# set random seed for training
torch.manual_seed(42)

do_gnn = True if len(ml.hidden_sizes_gcn) > 0 else False
do_edge_wgt = ml.edge_weights

if do_gnn and (ml.distance is None):
    raise ValueError("Need to specify a distance metric for the adjacency matrix in the ML config")

if do_gnn:

    # Determine which variable is used for building the graph (embedding_variable or distance_variable)
    # This matches the logic in torch_adj_builder and torch_train
    distance_variable = ml.embedding_variable if ml.embedding_variable is not None \
        else ml.distance_variable

    if ml.embedding_variable is not None:
        logging.info("Loading graph built with embedding variable set: %s", distance_variable)
    else:
        logging.info("Loading graph built with distance variable set: %s", distance_variable)

    # Check that exactly one linking length method is specified
    num_methods = sum([
        ml.linking_length is not None,
        ml.edge_frac is not None,
        ml.targettarget_eff is not None
    ])
    
    if num_methods > 1:
        raise ValueError("Only one of linking_length, edge_frac, or targettarget_eff can be set in ML config!")
    if num_methods == 0:
        raise ValueError("Must specify one of linking_length, edge_frac, or targettarget_eff in ML config!")
    
    # Set ll_str and adj_path based on which method is used
    if ml.linking_length is not None:
        logging.info("Using manual linking length from config: %s", str(ml.linking_length))
        ll_str = "_LL" + str(ml.linking_length).replace(".", "p")
        adj_path = user.adj_path + "/" + str(distance_variable) + "_" + str(ml.distance) + "_linking_length_" + \
            str(ml.linking_length).replace(".", "p") + "/"
    
    elif ml.edge_frac is not None:
        logging.info("Using edge_frac to define linking length: %s", str(ml.edge_frac))
        if ml.edge_frac not in [0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]:
            raise ValueError("""Not given a supported edge fraction, must be one of:
                             (0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5)""")
        ll_str = "_LLFrac" + str(ml.edge_frac).replace(".", "p")
        adj_path = user.adj_path + "/" + str(distance_variable) + "_" + str(ml.distance) + "_edge_frac_" + \
            str(ml.edge_frac).replace(".", "p") + "/"
    
    else:  # ml.targettarget_eff is not None
        logging.info("Using targettarget_eff to define linking length: %s", str(ml.targettarget_eff))
        if ml.targettarget_eff not in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            raise ValueError("""Not given a supported target efficiency, must be one of:
                             (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)""")
        ll_str = "_LLTargetEff" + str(ml.targettarget_eff).replace(".", "p")
        adj_path = user.adj_path + "/" + str(distance_variable) + "_" + str(ml.distance) + "_targettarget_eff_" + \
            str(ml.targettarget_eff).replace(".", "p") + "/"

### str for train/val split label. If single fold, then val_frac is 1/num_folds. Otherwise, nf is num_folds
if ml.single_fold:
    val_frac = 1/ml.num_folds
    nf_str = f"_val_frac{val_frac:.2f}"
else:
    nf_str = "_nf" + str(ml.num_folds)

### create model label and result plot path
if len(ml.hidden_sizes_gcn) == 0:
    model_label = user.signal\
          + "_MLP" + "-".join(map(str, ml.hidden_sizes_mlp)).replace(".", "p")\
          + "_lr" + str(ml.LR).replace(".", "p") + "P" + str(ml.patience_LR)\
          + "_dr" + "-".join(map(str, ml.dropout_rates)).replace(".", "p")\
          + "_bs" + str(ml.batch_size)\
          + "_e" + str(ml.epochs)\
          + nf_str
else:
    model_label = user.signal\
            + f"_{ml.gnn_type}" + "-".join(map(str, ml.hidden_sizes_gcn)).replace(".", "p")\
            + "_MLP" + "-".join(map(str, ml.hidden_sizes_mlp)).replace(".", "p")\
            + "_nb" + "-".join(map(str, ml.num_nb_list))\
            + "_lr" + str(ml.LR).replace(".", "p") + "P" + str(ml.patience_LR)\
            + ll_str\
            + "_dr" + "-".join(map(str, ml.dropout_rates)).replace(".", "p")\
            + "_bs" + str(ml.batch_size)\
            + "_e" + str(ml.epochs)\
            + nf_str

if do_gnn:
    plot_path = user.plot_path + ml.distance + "_models/" + model_label + "/"
else:
    plot_path = user.plot_path + "/MLP/" + model_label + "/"
misc.create_dirs(plot_path)

if user.signal == "stau":
    kinematics = misc.get_kinematics_staus(ml.ml_variable)
else:
    kinematics = misc.get_kinematics(ml.ml_variable)
input_size = len(kinematics)

logging.info("signal: %s", user.signal)
logging.info("chosen model: %s", model_label)
logging.info("kinematic variable set: %s", ml.ml_variable)
logging.info("graph built with variable set: %s", distance_variable)
logging.info("input data path: %s", user.kinematic_h5_path)
logging.info("input ll json path: %s", user.ll_path)
logging.info("input distances path: %s", user.dist_path)
logging.info("output plot path: %s", plot_path)
if do_gnn:
    logging.info("adj matrix storage path: %s", adj_path)
    model_path = user.model_path + ml.distance + "_models/" + model_label + "/" + ml.gnn_type + "/"
else:
    model_path = user.model_path + "dnn_models/" + model_label + "/"
logging.info("model storage path: %s", model_path)

logging.info("distance metric: %s", ml.distance)
if ml.edge_frac is not None:
    logging.info("desired edge fraction: %s", str(ml.edge_frac))
elif ml.targettarget_eff is not None:    
    logging.info("desired target efficiency: %s", str(ml.targettarget_eff))
elif ml.linking_length is not None:
    logging.info("linking length: %s", str(ml.linking_length))

# load training data file and kinematics
logging.info('Importing signal and background files...')

# normal loading setup
full_sig, full_bkg, full_x, \
full_sig_wgts, full_bkg_wgts, \
full_sig_labels, full_bkg_labels, \
sig_fold, bkg_fold = adj.data_loader(user.kinematic_h5_path, kinematics, ex=user.cutstring,
                                     signal=user.signal, signal_mass=user.signal_mass,
                                     num_folds=ml.num_folds)

len_sig = len(full_sig)
len_bkg = len(full_bkg)
logging.info("full sig size %s", str(full_sig.size()))
logging.info("full bkg size %s", str(full_bkg.size()))

full_x = full_x.to(cpu)
full_y = torch.cat((full_sig_labels, full_bkg_labels), dim=0).to(cpu)
len_full = len(full_y)
del full_sig, full_bkg, full_sig_labels, full_bkg_labels

logging.info("full sig yields %s", str(full_sig_wgts.sum()))
logging.info("full bkg yields %s", str(full_bkg_wgts.sum()))
full_wgts = torch.cat((full_sig_wgts, full_bkg_wgts), dim=0).to(cpu)
del full_sig_wgts, full_bkg_wgts

logging.info("sig_fold count:")
values, counts = np.unique(sig_fold, return_counts=True)
for val, count in zip(values, counts):
    logging.info("%s: %s", str(val), str(count))
logging.info("bkg_fold count:")
values, counts = np.unique(bkg_fold, return_counts=True)
for val, count in zip(values, counts):
    logging.info("%s: %s", str(val), str(count))

fold_assignment = np.concatenate((sig_fold, bkg_fold), axis=0)

logging.info("Loaded signal and background data.")
logging.info("Time taken so far: %s", str(time.time()-st))

### load edge indices if gnn layers are used
if do_gnn:
    logging.info("constructing sparse adjacency matrix ...")
    logging.info("loading row indices ...")
    row_ind = torch.load(adj_path+'row_ind.pt')
    logging.info("loading col indices ...")
    col_ind = torch.load(adj_path+'col_ind.pt')
    logging.info("stacking row and col indices ...")
    edge_ind = torch.stack((row_ind, col_ind)).to(cpu)
    logging.info("deleting row and col indices ...")
    del row_ind, col_ind
    if ml.gnn_type == "Graph":
        edge_ind = edge_ind.to(torch.int64)

    logging.info("Edge fraction: %s", str(edge_ind.shape[1] / len(full_y)**2))
    if do_edge_wgt:
        logging.info("loading edge weights ...")
        edge_wgts = torch.load(adj_path+'edge_wgts.pt').to(cpu)
        # edge weights from MC source node:
        edge_weights_from_MC = full_wgts[edge_ind[0]]
        if ml.gnn_type != "GAT":
            edge_wgts = edge_wgts * edge_weights_from_MC
            edge_weights_from_MC = None
else:
    edge_ind = None
    edge_wgts = None

if ml.plot_conv_kinematics and do_gnn:
    if do_edge_wgt:
        sparse_adj_matrix = torch.sparse_coo_tensor(edge_ind, edge_wgts,
                                                    size=(len(full_y), len(full_y)))
    else:
        edges = torch.ones(edge_ind.shape[1], dtype=torch.float32)
        sparse_adj_matrix = torch.sparse_coo_tensor(edge_ind, edges,
                                                    size=(len(full_y), len(full_y)))
        del edges
    del sparse_adj_matrix

gc.collect()
torch.cuda.empty_cache()
if torch.cuda.is_available():
    misc.print_mem_info()

### create data object, train and val loaders
if do_edge_wgt and do_gnn:
    data = Data(x=full_x, y=full_y, edge_index=edge_ind,
                node_weight=full_wgts, edge_weight=edge_wgts,
                mc_weight=edge_weights_from_MC if ml.gnn_type == "GAT" else \
                          torch.tensor([], device=full_wgts.device))
    del edge_ind, edge_wgts, edge_weights_from_MC
else:
    data = Data(x=full_x, y=full_y, edge_index=edge_ind, node_weight=full_wgts)
    del edge_ind

del full_y, full_wgts
gc.collect()
torch.cuda.empty_cache()

val_outputs = torch.tensor([]).to(cpu)
val_truth_labels = torch.tensor([]).to(cpu)
val_wgts = torch.tensor([]).to(cpu)

logging.info("Starting k-fold model application ...")
for fold_no in range(ml.num_folds):

    model_file_name = f"model_fold_{fold_no}.pth"
    logging.info("Loading model: %s", model_file_name)
    #### finish loading model to use mean and std from model to standardise data
    model_state_dict = torch.load(model_path + model_file_name, weights_only=True)
    means, stds = model_state_dict['normalisation_params']['means'], \
                  model_state_dict['normalisation_params']['stds']
    data_standardised = data
    data_standardised.x = misc.torch_standardise(data.x, means, stds)
    means, stds = means.to(cpu), stds.to(cpu)

    val_idx = np.where(fold_assignment == fold_no)[0]

    val_loader = NeighborLoader(
        data_standardised,
        input_nodes=val_idx,
        num_neighbors=ml.num_nb_list,
        shuffle=False,
        batch_size=ml.batch_size,
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
    )

    model = GCNClassifier(input_size=input_size, hidden_sizes_gcn=ml.hidden_sizes_gcn,
                          hidden_sizes_mlp=ml.hidden_sizes_mlp, output_size=1,
                          dropout_rates=ml.dropout_rates, gnn_type=ml.gnn_type)
    model.load_state_dict(model_state_dict["model_state"])
    model.eval()
    model.to(device)

    val_outputs_fold = []
    val_truth_labels_fold = []
    val_wgts_fold = []
    
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device, non_blocking=True)
            tmp_batch_size = batch.batch_size
            if do_edge_wgt and do_gnn:
                outputs = model(batch.x, batch.edge_index, batch.edge_weight, batch.mc_weight)
            else:
                outputs = model(batch.x, batch.edge_index)

            ### NOTE only consider predictions and labels of seed nodes (transductive learning)
            y = batch.y[:tmp_batch_size]
            outputs = outputs[:tmp_batch_size]
            event_wgts = batch.node_weight[:tmp_batch_size]

            val_outputs_fold.append(outputs.detach())
            val_truth_labels_fold.append(y.detach())
            val_wgts_fold.append(event_wgts.detach())

    val_outputs_fold = torch.cat(val_outputs_fold).to(cpu)
    val_truth_labels_fold = torch.cat(val_truth_labels_fold).to(cpu)
    val_wgts_fold = torch.cat(val_wgts_fold).to(cpu)

    val_outputs = torch.cat((val_outputs, val_outputs_fold))
    val_truth_labels = torch.cat((val_truth_labels, val_truth_labels_fold))
    val_wgts = torch.cat((val_wgts, val_wgts_fold))
    
    del model, val_loader
    gc.collect()
    torch.cuda.empty_cache()
    
    if ml.single_fold:
        logging.info("Single fold application, breaking loop ...")
        break

val_outputs = val_outputs.view(-1)
val_label_bool = val_truth_labels.bool()
val_sig_pred = val_outputs[val_label_bool]
val_sig_wgts = val_wgts[val_label_bool]
val_bkg_pred = val_outputs[torch.logical_not(val_label_bool)]
val_bkg_wgts = val_wgts[torch.logical_not(val_label_bool)]

val_fpr, val_tpr, val_cut = roc_curve(val_truth_labels.detach().cpu().numpy(),
                                      val_outputs.detach().cpu().numpy(),
                                      sample_weight=val_wgts.detach().cpu().numpy())
if user.signal == "stau":  ### stau fpr needs to be clipped and sorted due to rounding errors
    val_fpr = np.clip(val_fpr, 0, 1)
    val_fpr = np.sort(val_fpr)
val_auc = auc(val_fpr, val_tpr)
logging.info("Validation AUC %s", str(val_auc))

logging.info("Plotting model outputs ...")
linking_length_label = ""
if do_gnn:
    if ml.edge_frac is not None:
        linking_length_label = "Linking length at "+str(ml.edge_frac)+" edge fraction"
    elif ml.targettarget_eff is not None:
        linking_length_label = "Linking length at "+str(ml.targettarget_eff)+" target efficiency"
    elif ml.linking_length is not None:
        linking_length_label = "Linking length "+str(ml.linking_length)

text = [f"Validation AUC = {val_auc:.3f}", signal_label, background_label, linking_length_label]

fig_pred, ax_pred = plt.subplots()
binning = np.linspace(0, 1, 41)
ax_pred.hist(val_sig_pred.detach().cpu().numpy(), bins=binning,
             label="Signal (validation)", alpha=0.5, density=True,
             color="darkorange", weights=val_sig_wgts.detach().cpu().numpy())
ax_pred.hist(val_bkg_pred.detach().cpu().numpy(), bins=binning,
             label="Background (validation)", alpha=0.5, density=True,
             color="steelblue", weights=val_bkg_wgts.detach().cpu().numpy())
plotting.add_text(ax_pred, text, do_atlas=False, startx=0.02, starty=0.95)
ymin, ymax = ax_pred.get_ylim()
ax_pred.set_ylim(0.5*ymin, 10*ymax)
plotting.draw_labels_legends(ax_pred, "Output score", "Normalised # Events",
                             yrange=[0.5*ymin, 10*ymax], log_y=True)
plotting.save_fig(fig_pred, plot_path+"validation_pred")

score_path = user.score_path + model_label + "/"
misc.create_dirs(score_path)

np.save(score_path+"val_sig_pred.npy", val_sig_pred.detach().cpu().numpy())
np.save(score_path+"val_sig_wgts.npy", val_sig_wgts.detach().cpu().numpy())

np.save(score_path+"val_bkg_pred.npy", val_bkg_pred.detach().cpu().numpy())
np.save(score_path+"val_bkg_wgts.npy", val_bkg_wgts.detach().cpu().numpy())

logging.info("Plotting ROC curves ...")
fig_roc, ax_roc = plt.subplots()
plt.plot(val_fpr, val_tpr, label=f'Validation ROC curve (AUC = {val_auc:.3f})')
plt.xlim(0, 1)
plotting.draw_labels_legends(ax_roc, "Background Efficiency", "Signal Efficiency",
                             legendloc="upper left", yrange=[0., 1.2], log_y=False)
plotting.save_fig(fig_roc, plot_path+"validation_ROC")

logging.info("Saving ROC curves to json files ...")
roc_json_path = plot_path+"roc.json"
roc_dict = {"val_fpr": val_fpr.tolist(),
            "val_tpr": val_tpr.tolist(),
            "val_auc": [val_auc]
           }
with open(roc_json_path, 'w', encoding='utf-8') as json_file:
    json.dump(roc_dict, json_file)