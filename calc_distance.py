import pandas as pd
import numpy as np
import math
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
import energyflow as ef
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
        help="Specify the type of distance to calculate (euclidean, cosine, cityblock, emd)",
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
distance = str(args.distance)

user_config_path = args.userconfig
user_config = misc.load_config(user_config_path)
feature_h5_path = user_config["feature_h5_path"]
plot_path = user_config["plot_path"]
dist_path = user_config["dist_path"]

signal = user_config["signal"]
signal_mass = str(user_config["signal_mass"])
feature_dim = user_config["feature_dim"]
objects = user_config["objects"]
assert signal in ["hhh", "LQ", "stau"], f"Invalid signal type: {signal}"

logging.info("distance metric: "+distance)
logging.info("signal: "+signal)
logging.info("variable set: "+variable)
logging.info("input data path: "+feature_h5_path)
logging.info("input distances path: "+dist_path)
logging.info("output plot path: "+plot_path)
kinematics, kinematic_labels = misc.get_kinematics(variable, feature_dim)

# load in input files
logging.info('Importing signal and background files...')
if signal == "hhh": SF_4b5b = 0.07 # placeholder value for HHH data-driven background, MC backgrounds would take eventWeights instead

if distance == "emd":
    # don't standardise the kinematic variables before emd calculation
    standardise_bool = False
else:
    # but standardise them for other distances
    standardise_bool = True
full_sig, full_bkg, full_x, sig_wgt, bkg_wgt, sig_labels, bkg_labels = adj.data_loader(feature_h5_path, plot_path, kinematics, kinematic_labels, ex="", plot=True, signal=signal, signal_mass=signal_mass, standardisation=standardise_bool)

global_bkg_wgt = 1.0
if signal == "hhh": global_bkg_wgt = SF_4b5b
bkg_wgt = bkg_wgt*global_bkg_wgt

# calculate distances in batches
logging.info('Calculating distances in batches...')

def distance_calc(a, b, metric):
    if metric == "euclidean":
        d = dis.euclidean(a,b)
    elif metric == "cityblock":
        d = dis.cityblock(a,b)
    elif metric == "cosine":
        d = dis.cosine(a,b)
    elif metric == "emd":
        d = dis.torch_emd(a, b, objects, kinematics)
    else:
        d = None
        print("Please specify a valid distance metric, from euclidean, cityblock or cosine")
    
    if torch.sum(torch.isnan(d)).item() == 0:
        return d
    else:
        print("There are NaNs in your distances, please check!")
        return None
    
batch_size = 30000

num_sig_events = full_sig.shape[0]
num_sig_batches = math.ceil(num_sig_events/batch_size)

num_bkg_events = full_bkg.shape[0]
num_bkg_batches = math.ceil(num_bkg_events/batch_size)

print(num_sig_events," signal events; ",num_bkg_events," background events")
print(num_sig_batches," signal batches; ",num_bkg_batches," background batches")

save_path = dist_path+"/batched_"+variable +"_"+distance+"_distances/"
misc.create_dirs(save_path)

logging.info('Calculating sigsig distances ...')
for i in range(num_sig_batches):
    start_idx_sig_i = i * batch_size
    end_idx_sig_i = min((i + 1) * batch_size, num_sig_events)
    batch_sig_i = full_sig[start_idx_sig_i:end_idx_sig_i]
    batch_sig_wgt_i = sig_wgt[start_idx_sig_i:end_idx_sig_i]
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

        print("Sigsig file ", i, j)
        torch.save(batch_dict, save_path + f'sigsig_distances_batch_{i}_{j}.pt')

logging.info('Calculating bkgbkg distances ...')
for i in range(num_bkg_batches):
    start_idx_bkg_i = i * batch_size
    end_idx_bkg_i = min((i + 1) * batch_size, num_bkg_events)
    batch_bkg_i = full_bkg[start_idx_bkg_i:end_idx_bkg_i]
    batch_bkg_wgt_i = bkg_wgt[start_idx_bkg_i:end_idx_bkg_i]
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
        print("Bkgbkg file ij ", i, j)
        torch.save(batch_dict, save_path + f'bkgbkg_distances_batch_{i}_{j}.pt')

logging.info('Calculating sigbkg distances ...')
for i in range(num_sig_batches):
    start_idx_sig = i * batch_size
    end_idx_sig = min((i + 1) * batch_size, num_sig_events)
    batch_sig = full_sig[start_idx_sig:end_idx_sig]
    batch_sig_wgt = sig_wgt[start_idx_sig:end_idx_sig]

    for j in range(num_bkg_batches):
        start_idx_bkg = j * batch_size
        end_idx_bkg = min((j + 1) * batch_size, num_bkg_events)

        batch_bkg = full_bkg[start_idx_bkg:end_idx_bkg]
        batch_bkg_wgt = bkg_wgt[start_idx_bkg:end_idx_bkg]

        batch_sigbkg = distance_calc(batch_sig, batch_bkg, distance)
        batch_sigbkg_wgt = torch.ger(batch_sig_wgt, batch_bkg_wgt)

        batch_dict = {'distance': batch_sigbkg, 'weight': batch_sigbkg_wgt}
        print("Sigbkg file ", i, j)
        torch.save(batch_dict, save_path + f'sigbkg_distances_batch_{i}_{j}.pt')

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

plot_path = plot_path+"/"+variable+"/"
misc.create_dirs(plot_path)
plotting.plot_distances(np_sigsig, np_sigbkg, np_bkgbkg, np_sigsig_wgt, np_sigbkg_wgt, np_bkgbkg_wgt, variable, distance, plot_path)
