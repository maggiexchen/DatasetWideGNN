"""
Module to calculate the adjacency matrix
"""
import os
import json
import argparse
import logging
import time

import utils.normalisation as norm
import utils.adj_mat as adj
import utils.misc as misc
import utils.user_config as uconfig
import utils.ml_config as mlconfig

import numpy as np
import matplotlib.pyplot as plt
import torch

st = time.time()

logging.getLogger().setLevel(logging.INFO)

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

parser.add_argument(
    "--batchsize",
    "-b",
    type=int,
    default=10000,
    required=False,
    help="",
)

args = parser.parse_args()

user_config_path = args.userconfig
user = uconfig.UserConfig.from_yaml(user_config_path)

### load training config
ml_config_path = args.MLconfig
ml = mlconfig.MLConfig.from_yaml(ml_config_path)

print("CUDA is available? ", torch.cuda.is_available())  # Outputs True if GPU is available
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(torch.cuda.mem_get_info())
batch_size = args.batchsize

os.makedirs(user.adj_path, exist_ok=True)

variable = ml.kinematic_variable if ml.embedding_variable==ml.embedding_variable \
    else ml.embedding_variable
do_edge_wgt = ml.edge_weights

os.makedirs(user.ll_path, exist_ok=True)

# TODO support edge_frac or target_eff
if ml.linking_length is not None:
    logging.info("linking length is given in config,\
                 IGNORING edge_frac/targettarget_eff if present!")
    adj_path = user.adj_path + "/" + str(ml.distance) + "_" + "linking_length_" + \
        str(ml.linking_length).replace(".","p") + "/"

elif ml.edge_frac is not None and ml.targettarget_eff is not None:
    raise ValueError("edge_frac and targettarget_eff in ML config, pick just one!")

elif ml.edge_frac is not None:
    logging.info("Will try to use edge_frac to define linking length....")
    if ml.edge_frac not in [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]:
        raise ValueError("""not given a supported edge fraction,
                         (0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5)""")
    ll_path = user.ll_path + "edge_frac"
    adj_path = user.adj_path + "/" + str(ml.distance) + "_" + "edge_frac_" + \
        str(ml.edge_frac).replace(".","p") + "/"

elif ml.targettarget_eff is not None:
    logging.info("Will try to use targettarget_eff to define linking length....")
    if ml.targettarget_eff not in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        raise ValueError("""not given a supported sig-sig efficiency,
                         (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)""")

    ll_path = user.ll_path + "targettarget_eff_"
    adj_path = user.adj_path + "/" + str(ml.distance) + "_" + "targettarget_eff_" + \
        str(ml.targettarget_eff).replace(".","p") + "/"

else:
    raise ValueError("Neither manual LL, edge_frac or targettarget_eff in ML config, pick one!")

ll_path = user.ll_path + variable + "_" + ml.distance + user.cutstring + "_linking_length.json"

if ml.linking_length is None:
    print("Saving linking length to ", ll_path)
    with open(ll_path, 'r', encoding="utf-8") as lfile:
        length_dict = json.load(lfile)
        lengths = length_dict["length"]
        if ml.edge_frac is not None:
            linking_length = lengths[length_dict["edge_frac"].index(ml.edge_frac)]
        else:
            linking_length = lengths[length_dict["bkgbkg_eff"].index(ml.targettarget_eff)]
        logging.info("linking length = %s", str(linking_length))

misc.create_dirs(adj_path)

kinematics = misc.get_kinematics(ml.kinematic_variable, user.feature_dim)
input_size = len(kinematics)

logging.info("kinematic variable set: %s", ml.kinematic_variable)
logging.info("embedding variable set: %s", ml.embedding_variable)
logging.info("distance metric: %s", ml.distance)
if ml.edge_frac is not None:
    logging.info("desired edge fraction: %s", str(ml.edge_frac))
elif ml.targettarget_eff is not None:
    logging.info("desired edge fraction: %s", str(ml.targettarget_eff))
else:
    logging.info("linking length: %s", str(ml.linking_length))

# load training data file and kinematics
logging.info('Importing signal and background files...')

# normalised signal and background kinematics
logging.info('Importing signal and background files...')
full_sig, full_bkg, full_x, sig_wgt, bkg_wgt, sig_labels, bkg_labels, _, _ = adj.data_loader(
    user.feature_h5_path,
    kinematics,
    ex=user.cutstring,
    signal=user.signal,
    signal_mass=user.signal_mass,
    standardisation=True
    )

print("Number of total events: ",full_x.size(0))
del full_x
full_event_weights = torch.cat((sig_wgt, bkg_wgt))
del sig_wgt, bkg_wgt

### load distances and apply linking length to receieve indices
logging.info("Batch applying the linking length and getting non-zero indices ...")
logging.info("For sigsig ...")
sigsig_result = adj.generate_batched_nonzero_ind(user.dist_path, variable, ml.distance, "sigsig",
                                                 linking_length, batch_size, user.cutstring,
                                                 friend_graph=ml.friend_graph, edge_wgt=do_edge_wgt)
if do_edge_wgt:
    sigsig_ind, sigsig_edge_wgts = sigsig_result
else:
    sigsig_ind = sigsig_result
print("sigsig: ",sigsig_ind.shape)
print("fraction of egdes in sigsig: ", sigsig_ind.shape[0]/(len(full_sig)**2))

logging.info("For sigbkg ...")
sigbkg_result = adj.generate_batched_nonzero_ind(user.dist_path, variable, ml.distance, "sigbkg",
                                                 linking_length, batch_size, user.cutstring,
                                                 friend_graph=ml.friend_graph, edge_wgt=do_edge_wgt)
if do_edge_wgt:
    sigbkg_ind, sigbkg_edge_wgts = sigbkg_result
else:
    sigbkg_ind = sigbkg_result

print("sigbg: ", sigbkg_ind.shape)
print("fraction of egdes in sigbkg: ", sigbkg_ind.shape[0]/(len(full_sig)*len(full_bkg)))

logging.info("For bkgsig ...")
bkgsig_ind = torch.clone(sigbkg_ind)
bkgsig_ind = bkgsig_ind[:, [1, 0]]
if do_edge_wgt:
    bkgsig_edge_wgts = torch.clone(sigbkg_edge_wgts)
print("bgsig: ", bkgsig_ind.shape)
print("fraction of egdes in bkgsig: ", bkgsig_ind.shape[0]/(len(full_bkg)*len(full_sig)))

logging.info("For bkgbkg ...")
bkgbkg_result = adj.generate_batched_nonzero_ind(user.dist_path, variable, ml.distance, "bkgbkg",
                                                 linking_length, batch_size, user.cutstring,
                                                 friend_graph=ml.friend_graph, edge_wgt=do_edge_wgt)
if do_edge_wgt:
    bkgbkg_ind, bkgbkg_edge_wgts = bkgbkg_result
else:
    bkgbkg_ind = bkgbkg_result
print("bgbg: ", bkgbkg_ind.shape)

# adding to the indices to form the full matrix indices
logging.info("Stitching together the non-zero indices ...")
sigbkg_ind[:,1] += len(full_sig)
bkgsig_ind[:,0] += len(full_sig)
bkgbkg_ind += len(full_sig)

logging.info("Concatenating the indices ...")
full_ind = torch.cat((sigsig_ind, sigbkg_ind, bkgsig_ind, bkgbkg_ind)).to(torch.int32)

if do_edge_wgt:
    full_edge_wgts = torch.cat((sigsig_edge_wgts, sigbkg_edge_wgts,
                                bkgsig_edge_wgts, bkgbkg_edge_wgts)).to(torch.float32)

    ### define indices for plotting random subsets of the edge weights
    sigsig_plot_ind = torch.randperm(len(sigsig_edge_wgts))[:2000]
    sigbkg_plot_ind = torch.randperm(len(sigbkg_edge_wgts))[:1000]
    bkgsig_plot_ind = torch.randperm(len(bkgsig_edge_wgts))[:1000]
    bkgbkg_plot_ind = torch.randperm(len(bkgbkg_edge_wgts))[:2000]

    ### plot the edge weights before minmax normalisation (1/d)
    fig, ax = plt.subplots()
    _, binning, _ = ax.hist(torch.cat((sigbkg_edge_wgts[sigbkg_plot_ind],
                                       bkgsig_edge_wgts[bkgsig_plot_ind])),
                            bins=70, color="darkorange", alpha=0.5, label="sig-bkg")
    ax.hist(sigsig_edge_wgts[sigsig_plot_ind], bins=binning,
            color="steelblue", alpha=0.5, label="sig-sig")
    ax.hist(torch.cat((sigbkg_edge_wgts[sigbkg_plot_ind], bkgsig_edge_wgts[bkgsig_plot_ind])),
            bins=binning, color="darkorange", alpha=0.5, label="sig-bkg")
    ax.hist(bkgbkg_edge_wgts[bkgbkg_plot_ind],
            bins=binning, color="forestgreen", alpha=0.5, label="bkg-bkg")
    ax.set_xlabel("Edge weights (excluding event weights)", loc="right")
    ax.set_ylabel("Edges / Bin", loc="top")
    ax.legend(loc="upper right")
    ax.set_yscale("log")
    fig.tight_layout()
    fig.savefig(adj_path+"Unnormed_edge_wgts_no_EventWeights.pdf", transparent=True)

    ### min_max normalise the edge weights
    inf_mask = torch.isinf(full_edge_wgts)
    max_wgt = torch.max(full_edge_wgts[~inf_mask])
    min_wgt = torch.min(full_edge_wgts[~inf_mask])
    full_edge_wgts = norm.minmax(full_edge_wgts, min_wgt, max_wgt)
    full_edge_wgts[inf_mask] = 1

    ### plot the edge weights after minmax normalisation (1/d)
    fig, ax = plt.subplots()
    binning = np.linspace(0, 1, 70)
    ax.hist(full_edge_wgts[sigsig_plot_ind], bins=binning,
            color="steelblue", alpha=0.5, label="sig-sig")
    ax.hist(torch.cat((full_edge_wgts[sigbkg_plot_ind+len(sigsig_edge_wgts)],
                       full_edge_wgts[bkgsig_plot_ind+len(sigsig_edge_wgts)+\
                                      len(sigbkg_edge_wgts)])),
            bins=binning, color="darkorange", alpha=0.5, label="sig-bkg")
    ax.hist(full_edge_wgts[bkgbkg_plot_ind+len(sigsig_edge_wgts)+len(sigbkg_edge_wgts)+\
                           len(bkgsig_edge_wgts)],
            bins=binning, color="forestgreen", alpha=0.5, label="bkg-bkg")
    ax.set_xlabel("Edge weights (including event weights)", loc="right")
    ax.set_ylabel("Edges / Bin", loc="top")
    ax.legend(loc="upper right")
    ax.set_yscale("log")
    fig.tight_layout()
    fig.savefig(adj_path+"Normed_edge_wgts_no_EventWeights.pdf", transparent=True)

    ### we are only combining the MC events later, in torch_train.py,
    # depending on the gnn model type
    # fig, ax = plt.subplots()
    # _, binning, _ = ax.hist(normed_full_edge_wgts[bkgbkg_plot_ind+len(sigsig_edge_wgts)+\
    #                                               len(sigbkg_edge_wgts)+len(bkgsig_edge_wgts)],\
    #                         bins=70, color="forestgreen", alpha=0.5, label="bkg-bkg")
    # ax.hist(torch.cat((normed_full_edge_wgts[sigbkg_plot_ind+len(sigsig_edge_wgts)],
    #           normed_full_edge_wgts[bkgsig_plot_ind+len(sigsig_edge_wgts)+\
    #                                 len(sigbkg_edge_wgts)])),
    #           bins=binning, color="darkorange", alpha=0.5, label="sig-bkg")
    # ax.hist(normed_full_edge_wgts[sigsig_plot_ind], bins=binning,
    #           color="steelblue", alpha=0.5, label="sig-sig")
    # ax.set_xlabel("Edge weights (including event weights)", loc="right")
    # ax.set_ylabel("Edges / Bin", loc="top")
    # ax.legend(loc="upper right")
    # ax.set_yscale("log")
    # fig.tight_layout()
    # fig.savefig(adj_path+"Normed_edge_wgts_with_EventWeights.pdf", transparent=True)

    del sigsig_edge_wgts, sigbkg_edge_wgts, bkgsig_edge_wgts, bkgbkg_edge_wgts

total_edges = sigsig_ind.shape[0]+sigbkg_ind.shape[0]+bkgbkg_ind.shape[0]
total_pairs = (len(full_sig)+len(full_bkg))**2
if ml.edge_frac is not None:
    print("Linking length at edge fraction ", ml.edge_frac)
else:
    print("Linking length at targettarget_eff ", ml.targettarget_eff)
print("The fraction of edges in graph is ", total_edges / total_pairs)
del sigsig_ind, sigbkg_ind, bkgsig_ind, bkgbkg_ind

misc.print_mem_info()
logging.info("Saving sparse adjacency matrix ... to %s", adj_path)

### saving the adjaceny matrix indices as edge indices
print("full ind rows: ", full_ind[:,0], full_ind[:,0].shape)
print("full ind cols: ", full_ind[:,1], full_ind[:,1].shape)
torch.save(full_ind[:,0], adj_path+'row_ind.pt')
torch.save(full_ind[:,1], adj_path+'col_ind.pt')
del full_ind
if do_edge_wgt:
    print("full edge wgts: ", full_edge_wgts, full_edge_wgts.shape)
    torch.save(full_edge_wgts, adj_path+'edge_wgts.pt')
