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
import utils.distances as dis
import utils.normalisation as norm
import utils.misc as misc
import utils.plotting as plotting

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
        "--sample",
        "-s",
        action="store_true",
        help="Specify whether the datasets are sampled",
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
sample = args.sample

# path = "/data/atlas/atlasdata3/maggiechen/gnn_project/" # maggies path
path = "/home/srutherford/GNN_shared/hhhgraph/data/" # sebs path
if args.path:
    path = args.path
    if path[-1]!="/": path += "/"

data_dir = "data/hhh_split_files/"
if args.data_dir:
    data_dir = args.data_dir
    if data_dir[-1]!="/": data_dir += "/"

logging.info("variable set: "+variable)
logging.info("distance metric: "+distance)
logging.info("do sampling? "+str(sample))
logging.info("output path: "+path)
logging.info("input path: "+data_dir)

kinematics = misc.get_kinematics(variable)

# load in input files
logging.info('Importing signal and background files...')
df_sig = pd.read_hdf(data_dir+"sig_train.h5", key="sig_train")
df_bkg = pd.read_hdf(data_dir+"bkg_train.h5", key="bkg_train")

# randomly sample from the training datasets for linking length calculation if specified
if sample == True:
    logging.info("Sampling...")
    df_sig = df_sig.sample(n=1500)
    df_bkg = df_bkg.sample(n=3000)

logging.info("MAD scaling...")
# normalise kinematic values using MAD scaling
for var in kinematics:
    df_sig.loc[:, var], df_bkg.loc[:, var] = norm.MAD_norm(df_sig.loc[:, var], df_bkg.loc[:, var])

# convert pandas dataframes to tf tensors
# only the kinematics used in distance calculation and weights need to be converted to tensors here for matrix multiplications

logging.info("Converting to tf tensors...")
tf_sig = df_sig[kinematics]
tf_bkg = df_bkg[kinematics]
tf_sig = tf.convert_to_tensor(tf_sig, dtype_hint="float32")
tf_bkg = tf.convert_to_tensor(tf_bkg, dtype_hint="float32")
y_sig = df_sig["target"]
y_bkg = df_bkg["target"]

# mutliple events kinematics by the corresponding event weights and calcualte distances
logging.info('Getting MC event weights and calcualte weight matrix ...')
# The scale factor that scales 5b data down to the expected 6b yields, this is just taken as the ratio between 5b data/4b data for now
SF_4b5b = 0.07
sig_wgt = tf.convert_to_tensor(df_sig["eventWeight"], dtype_hint="float32")
bkg_wgt = tf.convert_to_tensor(df_bkg["eventWeight"]*SF_4b5b, dtype_hint="float32")
sigsig_wgt = tf.matmul(tf.reshape(sig_wgt, [-1,1]), tf.reshape(sig_wgt, [-1,1]), transpose_b=True)
sigbkg_wgt = tf.matmul(tf.reshape(sig_wgt, [-1,1]), tf.reshape(bkg_wgt, [-1,1]), transpose_b=True)
bkgbkg_wgt = tf.matmul(tf.reshape(bkg_wgt, [-1,1]), tf.reshape(bkg_wgt, [-1,1]), transpose_b=True)

# calculate distances
logging.info('Calculating distances...')
if distance == "euclidean":
    sigsig = dis.euclidean(tf_sig, tf_sig)
    sigbkg = dis.euclidean(tf_sig, tf_bkg)
    bkgbkg = dis.euclidean(tf_bkg, tf_bkg)
elif distance == "cityblock":
    sigsig = dis.cityblock(tf_sig, tf_sig)
    sigbkg = dis.cityblock(tf_sig, tf_bkg)
    bkgbkg = dis.cityblock(tf_bkg, tf_bkg)
elif distance == "cosine":
    sigsig = dis.cosine(tf_sig, tf_sig)
    sigbkg = dis.cosine(tf_sig, tf_bkg)
    bkgbkg = dis.cosine(tf_bkg, tf_bkg)
else:
    print("Specify a valid distance please!")

logging.info("Checking for NaNs in distances ... ")
print(tf.reduce_sum(tf.cast(tf.math.is_nan(sigsig), tf.int32)))
print(tf.reduce_sum(tf.cast(tf.math.is_nan(sigbkg), tf.int32)))
print(tf.reduce_sum(tf.cast(tf.math.is_nan(bkgbkg), tf.int32)))

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

plot_path = path+"plots/MAD_norm_weighted/"+variable+"/"
misc.create_dirs(plot_path)
plotting.plot_distances(np_sigsig, np_sigbkg, np_bkgbkg, np_sigsig_wgt, np_sigbkg_wgt, np_bkgbkg_wgt, variable, distance, plot_path)