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

# path = "/data/atlas/atlasdata3/maggiechen/gnn_project/" # maggies path
path = "/home/srutherford/GNN_shared/hhhgraph/data/" # sebs path
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
sigsig_wgt = torch.outer(sig_wgt, sig_wgt)
sigbkg_wgt = torch.outer(sig_wgt, bkg_wgt)
bkgbkg_wgt = torch.outer(bkg_wgt, bkg_wgt)

# calculate distances
logging.info('Calculating distances...')
if distance == "euclidean":
    sigsig = dis.euclidean(full_sig, full_sig)
    sigbkg = dis.euclidean(full_sig, full_bkg)
    bkgbkg = dis.euclidean(full_bkg, full_bkg)
elif distance == "cityblock":
    sigsig = dis.cityblock(full_sig, full_sig)
    sigbkg = dis.cityblock(full_sig, full_bkg)
    bkgbkg = dis.cityblock(full_bkg, full_bkg)
elif distance == "cosine":
    sigsig = dis.cosine(full_sig, full_sig)
    sigbkg = dis.cosine(full_sig, full_bkg)
    bkgbkg = dis.cosine(full_bkg, full_bkg)
else:
    print("Specify a valid distance please!")

logging.info("Checking for NaNs in distances ... ")
print(torch.sum(torch.isnan(sigsig)).item())
print(torch.sum(torch.isnan(sigbkg)).item())
print(torch.sum(torch.isnan(bkgbkg)).item())

# plot the (sampled) MAD-normed distances
logging.info("Converting distance and weight tensors to np arrays for saving and plotting ... ")
np_sigsig = sigsig.numpy().flatten()
np_sigbkg = sigbkg.numpy().flatten()
np_bkgbkg = bkgbkg.numpy().flatten()
np_sigsig_wgt = sigsig_wgt.numpy().flatten()
np_sigbkg_wgt = sigbkg_wgt.numpy().flatten()
np_bkgbkg_wgt = bkgbkg_wgt.numpy().flatten()

logging.info('Writing to h5...')
save_path = path+"distances/"
misc.create_dirs(save_path)
sigsig_file, sigbkg_file, bkgbkg_file = misc.get_h5_paths(save_path, variable, distance)
f_sigsig = h5py.File(sigsig_file, "w")
f_sigbkg = h5py.File(sigbkg_file, "w")
f_bkgbkg = h5py.File(bkgbkg_file, "w")

dtype = np.dtype([('distance', np.float32), ('weight', np.float32)])
sigsig_dset = f_sigsig.create_dataset("sigsig", shape=(len(np_sigsig),), dtype=dtype, chunks=True, compression="gzip")
sigbkg_dset = f_sigbkg.create_dataset("sigbkg", shape=(len(np_sigbkg),), dtype=dtype, chunks=True, compression="gzip")
bkgbkg_dset = f_bkgbkg.create_dataset("bkgbkg", shape=(len(np_bkgbkg),), dtype=dtype, chunks=True, compression="gzip")
# writing distances, and weights in chunks

sigsig_dset['distance'] = np_sigsig
sigsig_dset['weight'] = np_sigsig_wgt
sigbkg_dset['distance'] = np_sigbkg
sigbkg_dset['weight'] = np_sigbkg_wgt
bkgbkg_dset['distance'] = np_bkgbkg
bkgbkg_dset['weight'] = np_bkgbkg_wgt

f_sigsig.close()
f_sigbkg.close()
f_bkgbkg.close()

logging.info("Plotting ...")
if distance == "cityblock":
    x_max = 40
elif distance == "euclidean":
    x_max = 20
elif distance == "cosine":
    x_max = 2
else:
    raise Exception('Eh?, pick a better distance metric (cityblock, eucidean, cosine)')

plot_path = path+"plots/standardised_weighted/"+variable+"/"
misc.create_dirs(plot_path)
plotting.plot_distances(np_sigsig, np_sigbkg, np_bkgbkg, np_sigsig_wgt, np_sigbkg_wgt, np_bkgbkg_wgt, variable, distance, plot_path)