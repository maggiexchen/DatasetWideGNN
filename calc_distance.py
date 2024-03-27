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
import tensorflow as tf
import utils.torch_distances as dis
import utils.normalisation as norm
import utils.misc as misc
import utils.plotting as plotting
import utils.adj_mat as adj
import torch

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
SF_4b5b = 0.07
train_sig, train_bkg, train_x, train_sig_wgts, train_bkg_wgts, train_truth_labels = adj.data_loader(data_dir, "train", kinematics)
val_sig, val_bkg, val_x, val_sig_wgts, val_bkg_wgts, val_truth_labels = adj.data_loader(data_dir, "val", kinematics)
full_sig = torch.cat((train_sig, val_sig), dim=0)
full_bkg = torch.cat((train_bkg, val_bkg), dim=0)
sig_wgt = torch.cat((train_sig_wgts, val_sig_wgts), dim=0)
bkg_wgt = torch.cat((train_bkg_wgts, val_bkg_wgts), dim=0)*SF_4b5b

# mutliple events kinematics by the corresponding event weights and calcualte distances
logging.info('Getting MC event weights and calcualte weight matrix ...')
# The scale factor that scales 5b data down to the expected 6b yields, this is just taken as the ratio between 5b data/4b data for now
sigsig_wgt = torch.ger(sig_wgt, sig_wgt)
sigbkg_wgt = torch.ger(sig_wgt, bkg_wgt)
bkgbkg_wgt = torch.ger(bkg_wgt, bkg_wgt)

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
    return d

def check_nans(dist):
    nans = tf.reduce_sum(tf.cast(tf.math.is_nan(dist), tf.int32))

batch_size = 10000
num_sig_events = full_sig.shape[0]
num_sig_batches = (num_sig_events + batch_size - 1) // batch_size
num_bkg_events = full_bkg.shape[0]
num_bkg_batches = (num_bkg_events + batch_size - 1) // batch_size

misc.create_dirs(path+"/batched_distances")
sigsig_batch_counter = 0
for i in range(num_sig_batches):
    start_idx_sig = i * batch_size
    end_idx_sig = min((i + 1) * batch_size, num_sig_events)
    batch_sig = full_sig[start_idx_sig:end_idx_sig]
    batch_sigsig = distance_calc(batch_sig, batch_sig, distance)
    torch.save(batch_sigsig, path+f'batched_distances/sigsig_distances_batch_{sigsig_batch_counter}.pt')
    sigsig_batch_counter += 1

bkgbkg_batch_counter = 0
for j in range(num_bkg_batches):
    start_idx_bkg = j * batch_size
    end_idx_bkg = min((j + 1) * batch_size, num_bkg_events)
    batch_bkg = full_bkg[start_idx_bkg:end_idx_bkg]
    batch_bkgbkg = distance_calc(batch_bkg, batch_bkg, distance)
    torch.save(batch_sigsig, path+f'batched_distances/bkgbkg_distances_batch_{bkgbkg_batch_counter}.pt')
    bkgbkg_batch_counter += 1

sigbkg_batch_counter = 0
for i in range(num_sig_batches):
    start_idx_sig = i * batch_size
    end_idx_sig = min((i + 1) * batch_size, num_sig_events)
    batch_sig = full_sig[start_idx_sig:end_idx_sig]
    sigbkg_distances = []
    for j in range(num_bkg_batches):
        start_idx_bkg = j * batch_size
        end_idx_bkg = min((j + 1) * batch_size, num_bkg_events)
        batch_bkg = full_bkg[start_idx_bkg:end_idx_bkg]
        batch_sigbkg = distance_calc(batch_sig, batch_bkg, distance)
        sigbkg_distances.append(batch_sigbkg)
    sigbkg_distances = torch.cat(sigbkg_distances, dim=0)
    torch.save(sigbkg_distances, path+f'batched_distances/sigbkg_distances_batch_{sigbkg_batch_counter}.pt')
    sigbkg_batch_counter += 1


# logging.info("Checking for NaNs in distances ... ")
# print(tf.reduce_sum(tf.cast(tf.math.is_nan(sigsig), tf.int32)))
# print(tf.reduce_sum(tf.cast(tf.math.is_nan(sigbkg), tf.int32)))
# print(tf.reduce_sum(tf.cast(tf.math.is_nan(bkgbkg), tf.int32)))

# # plot the (sampled) MAD-normed distances
# logging.info("Converting distance and weight tensors to np arrays for saving and plotting ... ")
# np_sigsig = sigsig.numpy().flatten()
# np_sigbkg = sigbkg.numpy().flatten()
# np_bkgbkg = bkgbkg.numpy().flatten()
# np_sigsig_wgt = sigsig_wgt.numpy().flatten()
# np_sigbkg_wgt = sigbkg_wgt.numpy().flatten()
# np_bkgbkg_wgt = bkgbkg_wgt.numpy().flatten()

# logging.info('Writing to h5...')
# save_path = path+"distances/"
# misc.create_dirs(save_path)
# sigsig_file, sigbkg_file, bkgbkg_file = misc.get_h5_paths(save_path, variable, distance)
# f_sigsig = h5py.File(sigsig_file, "w")
# f_sigbkg = h5py.File(sigbkg_file, "w")
# f_bkgbkg = h5py.File(bkgbkg_file, "w")

# dtype = np.dtype([('distance', np.float32), ('weight', np.float32)])
# sigsig_dset = f_sigsig.create_dataset("sigsig", shape=(len(np_sigsig),), dtype=dtype, chunks=True, compression="gzip")
# sigbkg_dset = f_sigbkg.create_dataset("sigbkg", shape=(len(np_sigbkg),), dtype=dtype, chunks=True, compression="gzip")
# bkgbkg_dset = f_bkgbkg.create_dataset("bkgbkg", shape=(len(np_bkgbkg),), dtype=dtype, chunks=True, compression="gzip")
# # writing distances, and weights in chunks

# sigsig_dset['distance'] = np_sigsig
# sigsig_dset['weight'] = np_sigsig_wgt
# sigbkg_dset['distance'] = np_sigbkg
# sigbkg_dset['weight'] = np_sigbkg_wgt
# bkgbkg_dset['distance'] = np_bkgbkg
# bkgbkg_dset['weight'] = np_bkgbkg_wgt

# f_sigsig.close()
# f_sigbkg.close()
# f_bkgbkg.close()

# logging.info("Plotting ...")
# if distance == "cityblock":
#     x_max = 40
# elif distance == "euclidean":
#     x_max = 20
# elif distance == "cosine":
#     x_max = 2
# else:
#     raise Exception('Eh?, pick a better distance metric (cityblock, eucidean, cosine)')

# plot_path = path+"plots/standardised_weighted/"+variable+"/"
# misc.create_dirs(plot_path)
# plotting.plot_distances(np_sigsig, np_sigbkg, np_bkgbkg, np_sigsig_wgt, np_sigbkg_wgt, np_bkgbkg_wgt, variable, distance, plot_path)