"""
Module to train the GCN!
"""
import json
import argparse
import gc
import logging
import time
from tqdm import tqdm
import sys

import utils.adj_mat as adj
import utils.misc as misc
import utils.performance as perf
import utils.plotting as plotting
import utils.training as training
import utils.user_config as uconfig
import utils.ml_config as mlconfig
from utils.gcn_model import GCNClassifier
#from utils.dnn_model import DNNClassifier

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
from sklearn.metrics import roc_curve, auc

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
user = uconfig.UserConfig.from_yaml(user_config_path)

### load training config
ml_config_path = args.MLconfig
ml = mlconfig.MLConfig.from_yaml(ml_config_path)

if user.wandb_project is not None and user.wandb_entity is not None:
    import wandb


### set up CUDA/device device settings
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

## LR scheduler patience should be less than early stopping patience,
#  so that the LR can be reduced before training stops
assert ml.patience_LR < ml.patience_early_stopping, \
    "LR scheduler patience should be less than early stopping patience"

if do_gnn and (ml.distance is None):
    raise ValueError("Need to specify a distance metric for the adjacency matrix in the ML config")

if do_gnn:

    # Determine which variable is used for building the graph (embedding_variable or distance_variable)
    # This matches the logic in torch_adj_builder
    distance_variable = ml.embedding_variable if ml.embedding_variable is not None \
        else ml.distance_variable
    if ml.embedding_variable is not None:
        logging.info("Loading graph constructed with embedding variables: %s", distance_variable)
    else:
        logging.info("Loading graph constructed with distance variables: %s", distance_variable)

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
        
### str for train/val split label. If single fold, then val_frac is 1/num_folds.
# Otherwise, nf is num_folds
if ml.single_fold is True:
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

# Set up wandb
if user.wandb_project is not None and user.wandb_entity is not None:
    logging.info("Setting up wandb ...")
    wandb.init(
                project=user.wandb_project,
                entity=user.wandb_entity,
                config=ml,
                name=model_label,
            )

kinematic_plot_path = user.plot_path + "/training_kinematics/" 
if do_gnn:
    if ml.edge_frac is not None:
        kinematic_plot_path += str(distance_variable) + "_" + str(ml.distance) + "_frac" + str(ml.edge_frac) + "/"
    elif ml.targettarget_eff is not None:
        kinematic_plot_path += str(distance_variable) + "_" + str(ml.distance) + "_targeteff" + str(ml.targettarget_eff) + "/"
    else:
        kinematic_plot_path += str(distance_variable) + "_" + str(ml.distance) + "_ll" + str(ml.linking_length) + "/"
if do_gnn:
    plot_path = user.plot_path + str(distance_variable) + "_" + str(ml.distance) + "_models/" + model_label + "/"
else:
    plot_path = user.plot_path + "/MLP/" + model_label + "/"
misc.create_dirs(plot_path)

if user.signal == "stau":
    kinematics = misc.get_kinematics_staus(ml.ml_variable)
else:
    kinematics = misc.get_kinematics(ml.ml_variable)
input_size = len(kinematics)

logging.info("chosen model: %s", model_label)
logging.info("input data path: %s", user.feature_h5_path)
logging.info("output plot path: %s", plot_path)
if do_gnn:
    logging.info("graph built with variable set: %s", distance_variable)
    logging.info("input distances path: %s", user.dist_path)
    logging.info("input ll json path: %s", user.ll_path)
    logging.info("adj matrix storage path: %s", adj_path)
    model_path = user.model_path + str(distance_variable) + "_" + str(ml.distance) + "_models/" + model_label + "/" + ml.gnn_type + "/"
else:
    model_path = user.model_path + "dnn_models/" + model_label + "/"
logging.info("model storage path: %s", model_path)

logging.info("distance metric: %s", ml.distance)
if ml.edge_frac is not None:
    logging.info("desired edge fraction: %s", str(ml.edge_frac))
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
logging.info("full sig wgt size %s", str(full_sig_wgts.size()))
logging.info("full bkg wgt size %s", str(full_bkg_wgts.size()))

full_x = full_x.to(cpu)
full_y = torch.cat((full_sig_labels, full_bkg_labels), dim=0).to(cpu)
len_full = len(full_y)
del full_sig, full_bkg, full_sig_labels, full_bkg_labels

logging.info("full sig yields %s", str(full_sig_wgts.sum()))
logging.info("full bkg yields %s", str(full_bkg_wgts.sum()))
full_wgts = torch.cat((full_sig_wgts, full_bkg_wgts), dim=0).to(cpu)
logging.info("full wgt size %s", str(full_wgts.size()))
del full_sig_wgts, full_bkg_wgts
gc.collect()
torch.cuda.empty_cache()

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

# note this is throwing a bug :( need to fix!
if ml.plot_conv_kinematics:
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
#    logging.info("Adjacency Matrix: ", adj_mat)
    del sparse_adj_matrix
#    for nconv in range(3):
#        plotting.plot_conv_kinematics(adj_mat, full_x, len_sig, kinematics,
#                                      signal, ml.edge_frac, kinematic_plot_path,
#                                      normalisation="D_half_inv", standardise=False,
#                                      nconv=nconv, edge_wgts=do_edge_wgt)
#    del adj_mat

gc.collect()
torch.cuda.empty_cache()
if torch.cuda.is_available():
    misc.print_mem_info()

logging.info("Training ...")
logging.info("full x %s", str(len(full_x)))
logging.info("full y %s", str(len(full_y)))
if len(ml.hidden_sizes_gcn) > 0:
    logging.info("Checking edge indices dim: %s", str(len(edge_ind)))

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

try:
    train_losses = []
    val_losses = []

    train_outputs = []
    train_outputs_per_fold = {}
    train_truth_labels = []
    train_wgts = []
    val_outputs = []
    val_outputs_per_fold = {}
    val_truth_labels = []
    val_wgts = []

    logging.info("Starting k-fold cross validation ...")
    logging.info("Time taken so far: %s", str(time.time()-st))

    for fold_no in range(ml.num_folds):
        train_idx = np.where(fold_assignment != fold_no)[0]
        val_idx = np.where(fold_assignment == fold_no)[0]

        logging.info("starting fold %s/%s", str(fold_no+1), str(ml.num_folds))
        logging.info("train idx %s", str(len(train_idx)))
        logging.info("val idx %s", str(len(val_idx)))

        ### calculating the mean and std of only the training data in each fold
        means, stds = misc.get_train_mean_std(full_x[train_idx])
        ### standardise the entire dataset
        data_standardised = data
        data_standardised.x = misc.torch_standardise(data_standardised.x, means, stds)
        means, stds = means.to(cpu), stds.to(cpu)
        model = GCNClassifier(input_size=input_size, hidden_sizes_gcn=ml.hidden_sizes_gcn,
                              hidden_sizes_mlp=ml.hidden_sizes_mlp, output_size=1,
                              dropout_rates=ml.dropout_rates, gnn_type=ml.gnn_type)
        model = model.to(device)

        optimiser = torch.optim.Adam(model.parameters(), lr=ml.LR)
        ### NOTE: patience for the scheculer is different from the early stopping patience
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimiser, mode='min',
                                                         patience=ml.patience_LR, factor = 0.5)

        train_loss = []
        val_loss = []

        all_labels = data_standardised.y[train_idx].to(cpu).numpy()
        all_node_weights = data_standardised.node_weight[train_idx].to(cpu).numpy()
        class_weights = training.binary_class_weights(all_labels, all_node_weights).to(device)
        logging.info("Training class weights: ")
        logging.info("         signal: %s", str(class_weights[1]))
        logging.info("         backgrounds: %s", str(class_weights[0]))
        if do_gnn:
            logging.info("Graph sub-sampling for training and validation ...")
        else:
            logging.info("Loading for training and validation ...")

        ### load in the training and validation dataset for the fold using train_idx/val_idx
        train_loader = NeighborLoader(
            data_standardised,
            input_nodes=train_idx,
            num_neighbors=ml.num_nb_list,
            shuffle=True,
            batch_size=ml.batch_size,
            num_workers=4*int(user.run_with_cuda),
            pin_memory=torch.cuda.is_available(),
        )
        val_loader = NeighborLoader(
            data_standardised,
            input_nodes=val_idx,
            num_neighbors=ml.num_nb_list,
            shuffle=False,
            batch_size=ml.batch_size,
            num_workers=4*int(user.run_with_cuda),
            pin_memory=torch.cuda.is_available(),
        )

        best_val_loss = float('inf')
        patience_counter = 0
        for epoch in range(ml.epochs):
            logging.info("Epoch %s", str(epoch))
            ### start training loop in the epoch
            model.train()
            train_outputs_fold = []
            train_truth_labels_fold = []
            train_wgts_fold = []
            train_x_fold = []
            epoch_wloss_sum = 0
            epoch_w_sum = 0

            logging.info("Training ...")
            for batch_idx, batch in tqdm(enumerate(train_loader), total=len(train_loader)):
                optimiser.zero_grad()
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

                pos_w, neg_w = class_weights[1], class_weights[0]
                sample_w = torch.where(y.squeeze() > 0.5, pos_w, neg_w) * event_wgts.squeeze() #multiply by class and event weights

                loss = training.weighted_bce_loss(outputs.squeeze(), y.squeeze().float(), class_weights, event_wgts)

                loss.backward()

                batch_w_sum = sample_w.sum()
                epoch_wloss_sum += loss.detach() * batch_w_sum
                epoch_w_sum += batch_w_sum
                

                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimiser.step()

                torch.cuda.empty_cache()
                train_outputs_fold.append(outputs.detach())
                train_truth_labels_fold.append(y.detach())
                train_wgts_fold.append(event_wgts.detach())
                train_x_fold.append(batch.x.detach())

                # wandb.log({
                #     f"train_batch_loss/fold_{fold_no}": float(loss) * tmp_batch_size},
                #     step=fold_no * ml.epochs * len(train_loader) + epoch * len(train_loader) + batch_idx
                # )
                
            train_outputs_fold = torch.cat(train_outputs_fold).detach().to(cpu)
            train_truth_labels_fold = torch.cat(train_truth_labels_fold).detach().to(cpu)
            train_wgts_fold = torch.cat(train_wgts_fold).detach().to(cpu)
            train_x_fold = torch.cat(train_x_fold).detach().to(cpu)

            avg_tr_loss = (epoch_wloss_sum / (epoch_w_sum + 1e-12)).item() #no div by 0, .item() to convert to float
            train_loss.append(avg_tr_loss)

            ### start validation loop in the epoch
            model.eval()
            val_outputs_fold= []
            val_truth_labels_fold = []
            val_wgts_fold = []
            val_x_fold = []
            epoch_wloss_sum = 0
            epoch_w_sum = 0

            logging.info("Validating ...")
            with torch.no_grad():
                for batch_idx, batch in tqdm(enumerate(val_loader), total=len(val_loader)):
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

                    pos_w, neg_w = class_weights[1], class_weights[0]
                    sample_w = torch.where(y.squeeze() > 0.5, pos_w, neg_w) * event_wgts.squeeze() #multiply class and event weights

                    loss = training.weighted_bce_loss(outputs.squeeze(), y.squeeze().float(), class_weights, event_wgts)

                    batch_w_sum = sample_w.sum()
                    epoch_wloss_sum += loss.detach() * batch_w_sum
                    epoch_w_sum += batch_w_sum

                    val_outputs_fold.append(outputs.detach())
                    val_truth_labels_fold.append(y.detach())
                    val_wgts_fold.append(event_wgts.detach())
                    val_x_fold.append(batch.x.detach())

                    # wandb.log({
                    #     f"val_batch_loss/fold_{fold_no}": float(loss) * tmp_batch_size},
                    #     step=fold_no * ml.epochs * len(val_loader) + epoch * len(val_loader) + batch_idx
                    # )

            val_outputs_fold = torch.cat(val_outputs_fold).detach().to(cpu)
            val_truth_labels_fold = torch.cat(val_truth_labels_fold).detach().to(cpu)
            val_wgts_fold = torch.cat(val_wgts_fold).detach().to(cpu)
            val_x_fold = torch.cat(val_x_fold).detach().to(cpu)

            avg_vl_loss = (epoch_wloss_sum / (epoch_w_sum + 1e-12)).item() #no div by 0, .item() to convert to float
            val_loss.append(avg_vl_loss)

            current_lr = optimiser.param_groups[0]['lr']
            scheduler.step(avg_vl_loss)
            new_lr = optimiser.param_groups[0]['lr']
            if new_lr < current_lr:
                logging.info("Learning rate reduced to: %s", str(new_lr))

            if avg_vl_loss < best_val_loss:
                best_val_loss = avg_vl_loss
                patience_counter = 0
            else:
                patience_counter += 1
                logging.info("No improvement in validation loss for %s epoch(s).", str(patience_counter))

            logging.info('Epoch %s/%s, Train Loss: %s, Validation Loss: %s', str(epoch+1), str(ml.epochs), str(avg_tr_loss), str(avg_vl_loss))

            if patience_counter >= ml.patience_early_stopping:
                logging.info("Early stopping after %s epochs.", str(epoch+1))
                break

            if user.wandb_project is not None and user.wandb_entity is not None:
                wandb.log({
                    f"train_epoch_loss/fold_{fold_no}": avg_tr_loss,
                    f"val_epoch_loss/fold_{fold_no}": avg_vl_loss,
                    "epoch": epoch,
                })

        train_outputs_per_fold["fold_"+str(fold_no+1)+"_outputs"] = train_outputs_fold.flatten().numpy()
        val_outputs_per_fold["fold_"+str(fold_no+1)+"_outputs"] = val_outputs_fold.flatten().numpy()

        logging.info("Finished fold %s/%s", str(fold_no), str(ml.num_folds))
        logging.info("Number of epochs: %s/%s", str(epoch+1), str(ml.epochs))
        logging.info("Final train Loss: %s", str(avg_tr_loss))
        logging.info("Final validation Loss: %s", str(avg_vl_loss))
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
        train_outputs.append(train_outputs_fold)
        train_truth_labels.append(train_truth_labels_fold)
        train_wgts.append(train_wgts_fold)
        val_outputs.append(val_outputs_fold)
        val_truth_labels.append(val_truth_labels_fold)
        val_wgts.append(val_wgts_fold)
        del train_loader, val_loader, model, optimiser, scheduler
        del train_outputs_fold, val_outputs_fold, train_truth_labels_fold, val_truth_labels_fold
        gc.collect()
        torch.cuda.empty_cache()

        if ml.single_fold is True:
            logging.info("Single fold training, breaking loop ...")
            break
    
    train_outputs = torch.cat(train_outputs)
    train_truth_labels = torch.cat(train_truth_labels)
    train_wgts = torch.cat(train_wgts)
    val_outputs = torch.cat(val_outputs)
    val_truth_labels = torch.cat(val_truth_labels)
    val_wgts = torch.cat(val_wgts)

    # TODO: move plotting function into plotting script
    logging.info("plotting model outputs per fold")
    fig_fold, ax_fold = plt.subplots()
    fold_colours = ["steelblue", "darkorange", "forestgreen"]
    for k in range(ml.num_folds):
        logging.info("Training Fold %s %s", str(k+1), str(train_outputs_per_fold["fold_"+str(k+1)+"_outputs"]))
        logging.info("Validation Fold %s %s", str(k+1), str(val_outputs_per_fold["fold_"+str(k+1)+"_outputs"]))
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

## Print errors occurring in the block above
## Abort training in case of error
except Exception as e:
    print("**ERROR - An error occurred during training: \n")
    print("Error message: ", e)

finally:

    # TODO: save model outputs and move ROC plotting elsewhere
    ### compute ROC curve and AUC
    logging.info("Training complete.")
    logging.info("train truth labels %s", str(len(train_truth_labels)))
    logging.info("val truth labels %s", str(len(val_truth_labels)))

    train_outputs = train_outputs.view(-1)
    train_label_bool = train_truth_labels.bool()
    train_sig_pred = train_outputs[train_label_bool]
    train_sig_wgts = train_wgts[train_label_bool]
    train_bkg_pred = train_outputs[torch.logical_not(train_label_bool)]
    train_bkg_wgts = train_wgts[torch.logical_not(train_label_bool)]

    train_fpr, train_tpr, train_cut = roc_curve(train_truth_labels.detach().to(cpu).numpy(),
                                                train_outputs.detach().to(cpu).numpy(),
                                                sample_weight=train_wgts.detach().to(cpu).numpy())

    if user.signal == "stau":
        # stau fpr needs to be clipped and sorted due to rounding errors
        train_fpr = np.clip(train_fpr, 0, 1)
        train_fpr = np.sort(train_fpr)
    train_auc = auc(train_fpr, train_tpr)
    logging.info("Training AUC %s", str(train_auc))

    val_outputs = val_outputs.view(-1)
    val_label_bool = val_truth_labels.bool()
    val_sig_pred = val_outputs[val_label_bool]
    val_sig_wgts = val_wgts[val_label_bool]
    val_bkg_pred = val_outputs[torch.logical_not(val_label_bool)]
    val_bkg_wgts = val_wgts[torch.logical_not(val_label_bool)]

    val_fpr, val_tpr, val_cut = roc_curve(val_truth_labels.detach().to(cpu).numpy(),
                                          val_outputs.detach().to(cpu).numpy(),
                                          sample_weight=val_wgts.detach().to(cpu).numpy())
    if user.signal == "stau": ### stau fpr needs to be clipped and sorted due to rounding errors
        val_fpr = np.clip(val_fpr, 0, 1)
        val_fpr = np.sort(val_fpr)
    val_auc = auc(val_fpr, val_tpr)
    logging.info("Validation AUC %s", str(val_auc))

    # save performance to json
    perf.save_performance(train_loss, train_fpr, train_tpr, train_cut, train_auc,
                          val_loss, val_fpr, val_tpr, val_cut, val_auc, model_path)
    perf.save_metadata_kfold(len(val_sig_pred), len(val_bkg_pred), ml.num_folds,
                             ml.hidden_sizes_gcn, ml.hidden_sizes_mlp, ml.LR, ml.dropout_rates,
                             ml.epochs, model_path)

    logging.info("Plotting training/validation losses ...")
    fig_loss, ax_loss = plt.subplots()
    x_epoch = np.arange(1,ml.epochs+1,1)
    for loss_loop, (train_loss, val_loss) in enumerate(zip(train_losses, val_losses)):
        train_line, = ax_loss.plot(np.arange(len(train_loss)), train_loss,
                                   label="Fold " + str(loss_loop) + " (Train)")
        colour = train_line.get_color()
        ax_loss.plot(np.arange(len(val_loss)), val_loss,
                     label="Fold " + str(loss_loop) + " (Val)", color=colour, linestyle="-.")
    ax_loss.legend(loc='upper right', fontsize=9)
    if do_gnn:
        model_text = [str(ml.gnn_type)+" model",
                      "GNN layers "+ str(ml.hidden_sizes_gcn),
                      "MLP layers "+ str(ml.hidden_sizes_mlp),
                      "Batchsize " + str(ml.batch_size), 
                      "Neighbour sampling " + str(ml.num_nb_list)]
    else:
        model_text = [str(ml.gnn_type) + " model",
                      "MLP layers "+ str(ml.hidden_sizes_mlp),
                      "Batchsize " + str(ml.batch_size),
                      "LR " + str(ml.LR) + ", patience " + str(ml.patience_LR)]
    plotting.add_text(ax_loss, model_text, do_atlas=False, startx=0.04, starty=0.95)

    ymin, ymax = ax_loss.get_ylim()
    plotting.draw_labels_legends(ax_loss, "Epoch", "Loss", yrange=[ymin, ymax*1.2])

    misc.create_dirs(plot_path)
    logging.info("Saving plots to %s", plot_path)
    plotting.save_fig(fig_loss, plot_path+"training_validation_loss")

    logging.info("Plotting model outputs ...")
    linking_length_label = ""
    if do_gnn:
        if ml.edge_frac is not None:
            linking_length_label = "Linking length at "+str(ml.edge_frac)+" edge fraction"
        elif ml.linking_length is not None:
            linking_length_label = "Linking length "+str(ml.linking_length)
    signal_label, background_label = plotting.get_plot_labels(user.signal, user.signal_mass)
    text = [f"Training AUC = {train_auc:.3f}", f"Validation AUC = {val_auc:.3f}",
            signal_label, background_label, linking_length_label]

    fig_pred, ax_pred = plt.subplots()
    binning = np.linspace(0, 1, 41)
    ax_pred.hist(train_sig_pred.detach().to(cpu).numpy(), bins=binning,
                 label="Signal (training)", histtype='step', linestyle='--', density=True,
                 color="darkorange", weights=train_sig_wgts.detach().to(cpu).numpy())
    ax_pred.hist(train_bkg_pred.detach().to(cpu).numpy(), bins=binning,
                 label="Background (training)", histtype='step', linestyle='--', density=True,
                 color="steelblue", weights=train_bkg_wgts.detach().to(cpu).numpy())
    ax_pred.hist(val_sig_pred.detach().to(cpu).numpy(), bins=binning,
                 label="Signal (validation)", alpha=0.5, density=True,
                 color="darkorange", weights=val_sig_wgts.detach().to(cpu).numpy())
    ax_pred.hist(val_bkg_pred.detach().to(cpu).numpy(), bins=binning,
                 label="Background (validation)", alpha=0.5, density=True,
                 color="steelblue", weights=val_bkg_wgts.detach().to(cpu).numpy())
    plotting.add_text(ax_pred, text, do_atlas=False, startx=0.02, starty=0.95)
    ymin, ymax = ax_pred.get_ylim()
    ax_pred.set_ylim(0.5*ymin, 10*ymax)
    plotting.draw_labels_legends(ax_pred, "Output score", "Normalised # Events",
                                 yrange=[0.5*ymin, 10*ymax], log_y=True)
    plotting.save_fig(fig_pred, plot_path+"training_validation_pred")

    score_path = user.score_path + model_label + "/"
    misc.create_dirs(score_path)

    np.save(score_path+"train_sig_pred.npy", train_sig_pred.detach().to(cpu).numpy())
    np.save(score_path+"train_sig_wgts.npy", train_sig_wgts.detach().to(cpu).numpy())

    np.save(score_path+"train_bkg_pred.npy", train_bkg_pred.detach().to(cpu).numpy())
    np.save(score_path+"train_bkg_wgts.npy", train_bkg_wgts.detach().to(cpu).numpy())
    np.save(score_path+"val_sig_pred.npy", val_sig_pred.detach().to(cpu).numpy())
    np.save(score_path+"val_sig_wgts.npy", val_sig_wgts.detach().to(cpu).numpy())

    np.save(score_path+"val_bkg_pred.npy", val_bkg_pred.detach().to(cpu).numpy())
    np.save(score_path+"val_bkg_wgts.npy", val_bkg_wgts.detach().to(cpu).numpy())
    logging.info("Plotting ROC curves ...")
    fig_roc, ax_roc = plt.subplots()
    plt.plot(train_fpr, train_tpr, label=f'Training ROC curve (AUC = {train_auc:.3f})')
    plt.plot(val_fpr, val_tpr, label=f'Validation ROC curve (AUC = {val_auc:.3f})')
    plt.xlim(0,1)
    plotting.add_text(ax_roc, model_text, do_atlas=False, startx=0.04, starty=0.3)
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

