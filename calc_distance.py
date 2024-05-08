import pandas as pd
import numpy as np
import h5py
import os
os.environ["NUMEXPR_MAX_THREADS"] = "16"
import matplotlib.pyplot as plt
import mplhep as hep
from sklearn.model_selection import train_test_split
import logging
logging.getLogger().setLevel(logging.INFO)
import argparse
import torch
import utils.torch_distances as dis
import utils.normalisation as norm
import utils.misc as misc
import utils.plotting as plotting
import utils.adj_mat as adj
import torch
torch.manual_seed(42)

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
        "--path",
        "-p",
        type=str,
        required=False,
        help="Specify the path to store all the input/output data and results",
    )

    parser.add_argument(
        "--data_dir",
        "-i",
        type=str,
        required=False,
        help="Specify the path to store all the input/output data and results",
    )


    args = parser.parse_args()
    return args

args = GetParser()

variable = str(args.variable)
distance = str(args.distance)

path = "/data/atlas/atlasdata3/maggiechen/gnn_project/" # maggies path
#path = "/home/srutherford/GNN_shared/hhhgraph/data/" # sebs path
if args.path:
    path = args.path
    if path[-1]!="/": path += "/"

data_dir = "data/"
if args.data_dir:
    data_dir = args.data_dir
    if data_dir[-1]!="/": data_dir += "/"

logging.info("variable set: "+variable)
logging.info("distance metric: "+distance)
logging.info("output path: "+path)
logging.info("input path: "+data_dir)

kinematics = misc.get_kinematics(variable)

# load in input files
logging.info('Importing signal and background files...')
SF_4b5b = 0.07 # placeholder value for HHH data-driven background, MC backgrounds would take eventWeights instead
train_sig, train_bkg, train_x, train_sig_wgts, train_bkg_wgts, train_truth_labels = adj.data_loader(data_dir, "train", kinematics)
val_sig, val_bkg, val_x, val_sig_wgts, val_bkg_wgts, val_truth_labels = adj.data_loader(data_dir, "val", kinematics)
full_sig = torch.cat((train_sig, val_sig), dim=0)
full_bkg = torch.cat((train_bkg, val_bkg), dim=0)
sig_wgt = torch.cat((train_sig_wgts, val_sig_wgts), dim=0)
bkg_wgt = torch.cat((train_bkg_wgts, val_bkg_wgts), dim=0)*SF_4b5b

# calculate distances in batches
logging.info('Calculating distances in batches...')
print("The number of signal events ", len(full_sig))
print("The number of background events", len(full_bkg))

def distance_calc(a, b, metric):
    if metric == "euclidean":
        d = dis.euclidean(a,b)
    elif metric == "cityblock":
        d = dis.cityblock(a,b)
    elif metric == "cosine":
        d = dis.cosine(a,b)
    else:
        d = None
        print("Please specify a valid distance metric, from euclidean, cityblock or cosine")
    
    if torch.sum(torch.isnan(d)).item() == 0:
        return d
    
def check_nans(dist):
    nans = tf.reduce_sum(tf.cast(tf.math.is_nan(dist), tf.int32))

batch_size = 30000
num_sig_events = full_sig.shape[0]
num_sig_batches = (num_sig_events + batch_size - 1) // batch_size
num_bkg_events = full_bkg.shape[0]
num_bkg_batches = (num_bkg_events + batch_size - 1) // batch_size

save_path = path+"/batched_"+variable +"_"+distance+"_distances/"
misc.create_dirs(save_path)
sigsig_batch_counter_i = 0
logging.info('Calculating sigsig distances ...')
for i in range(num_sig_batches):
    start_idx_sig_i = i * batch_size
    end_idx_sig_i = min((i + 1) * batch_size, num_sig_events)
    batch_sig_i = full_sig[start_idx_sig_i:end_idx_sig_i]
    batch_sig_wgt_i = sig_wgt[start_idx_sig_i:end_idx_sig_i]
    sigsig_batch_counter_j = 0
    for j in range(num_sig_batches):
        # don't need to save the redundant ones.
        if (i < j): continue
        start_idx_sig_j = j * batch_size
        end_idx_sig_j = min((j + 1) * batch_size, num_sig_events)
        batch_sig_j = full_sig[start_idx_sig_j:end_idx_sig_j]
        batch_sig_wgt_j = sig_wgt[start_idx_sig_j:end_idx_sig_j]

        batch_sigsig = distance_calc(batch_sig_i, batch_sig_j, distance)
        batch_sigsig_wgt = torch.ger(batch_sig_wgt_i, batch_sig_wgt_j)

        batch_dict = {'distance': batch_sigsig, 'weight': batch_sigsig_wgt}
        print("Sigsig file ", sigsig_batch_counter_i, sigsig_batch_counter_j)
        torch.save(batch_dict, save_path + f'sigsig_distances_batch_{sigsig_batch_counter_i}_{sigsig_batch_counter_j}.pt')
        sigsig_batch_counter_j += 1
    sigsig_batch_counter_i+=1

logging.info('Calculating bkgbkg distances ...')
bkgbkg_batch_counter_i = 0
bkgbkg_count = 0
for i in range(num_bkg_batches):
    start_idx_bkg_i = i * batch_size
    end_idx_bkg_i = min((i + 1) * batch_size, num_bkg_events)
    batch_bkg_i = full_bkg[start_idx_bkg_i:end_idx_bkg_i]
    batch_bkg_wgt_i = bkg_wgt[start_idx_bkg_i:end_idx_bkg_i]

    bkgbkg_batch_counter_j = 0
    for j in range(num_bkg_batches):
        # don't need to save the redundant ones.
        if (i < j): continue
        start_idx_bkg_j = j * batch_size
        end_idx_bkg_j = min((j + 1) * batch_size, num_bkg_events)
        batch_bkg_j = full_bkg[start_idx_bkg_j:end_idx_bkg_j]
        batch_bkg_wgt_j = bkg_wgt[start_idx_bkg_j:end_idx_bkg_j]
        batch_bkgbkg = distance_calc(batch_bkg_i, batch_bkg_j, distance)
        batch_bkgbkg_wgt = torch.ger(batch_bkg_wgt_i, batch_bkg_wgt_j)
        batch_dict = {'distance': batch_bkgbkg, 'weight': batch_bkgbkg_wgt}
        print("Bkgbkg file ", bkgbkg_batch_counter_i, bkgbkg_batch_counter_j)
#        torch.save(batch_dict, save_path + f'bkgbkg_distances_batch_{bkgbkg_batch_counter_i}_{bkgbkg_batch_counter_j}.pt')
        bkgbkg_batch_counter_j += 1
    bkgbkg_batch_counter_i += 1

logging.info('Calculating sigbkg distances ...')
sigbkg_batch_counter_i = 0
for i in range(num_sig_batches):
    start_idx_sig = i * batch_size
    end_idx_sig = min((i + 1) * batch_size, num_sig_events)
    batch_sig = full_sig[start_idx_sig:end_idx_sig]
    batch_sig_wgt = sig_wgt[start_idx_sig:end_idx_sig]

    sigbkg_batch_counter_j = 0
    for j in range(num_bkg_batches):
        start_idx_bkg = j * batch_size
        end_idx_bkg = min((j + 1) * batch_size, num_bkg_events)
        batch_bkg = full_bkg[start_idx_bkg:end_idx_bkg]
        batch_bkg_wgt = bkg_wgt[start_idx_bkg:end_idx_bkg]

        batch_sigbkg = distance_calc(batch_sig, batch_bkg, distance)
        batch_sigbkg_wgt = torch.ger(batch_sig_wgt, batch_bkg_wgt)

        batch_dict = {'distance': batch_sigbkg, 'weight': batch_sigbkg_wgt}
        print("Sigbkg file ", sigbkg_batch_counter_i, sigbkg_batch_counter_j)
        torch.save(batch_dict, save_path + f'sigbkg_distances_batch_{sigbkg_batch_counter_i}_{sigbkg_batch_counter_j}.pt')
        sigbkg_batch_counter_j += 1
    sigbkg_batch_counter_i += 1

# plot the MAD-normed distances from the first batch
logging.info("Plotting ... ")
sigsig = torch.load(save_path + 'sigsig_distances_batch_0_0.pt')
bkgbkg = torch.load(save_path + 'bkgbkg_distances_batch_0_0.pt')
sigbkg = torch.load(save_path + 'sigbkg_distances_batch_0_0.pt')
np_sigsig = sigsig['distance'].numpy().flatten()
np_sigbkg = sigbkg['distance'].numpy().flatten()
np_bkgbkg = bkgbkg['distance'].numpy().flatten()
np_sigsig_wgt = sigsig['weight'].numpy().flatten()
np_sigbkg_wgt = sigbkg['weight'].numpy().flatten()
np_bkgbkg_wgt = bkgbkg['weight'].numpy().flatten()

plot_path = path+"plots/standardised_weighted/"+variable+"/"
misc.create_dirs(plot_path)
plotting.plot_distances(np_sigsig, np_sigbkg, np_bkgbkg, np_sigsig_wgt, np_sigbkg_wgt, np_bkgbkg_wgt, variable, distance, plot_path)
