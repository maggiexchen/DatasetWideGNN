import pandas as pd
import uproot
import numpy
import h5py
import random
import os
os.environ["NUMEXPR_MAX_THREADS"] = "16"
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
import mplhep as hep
from sklearn.model_selection import train_test_split
import logging
logging.getLogger().setLevel(logging.INFO)
import argparse
import utils.misc as misc

def GetParser():
    """Argument parser for reading Ntuples script."""
    parser = argparse.ArgumentParser(
        description="Reading Ntuples command line options."
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

user_config_path = args.userconfig
user_config = misc.load_config(user_config_path)
h5_path = user_config["h5_path"]
ntuple_path = user_config["ntuple_path"]

signal = user_config["signal"]
feature_dim = user_config["feature_dim"]
assert signal in ["hhh", "LQ", "stau"], f"Invalid signal type: {signal}"

logging.info("signal: "+signal)
logging.info("input ntuple path: "+ntuple_path)
logging.info("output h5 data path: "+h5_path)

# load in input files
logging.info('Importing signal and background files...')
signal_file = uproot.open(ntuple_path + "6b_resonant_TRSM/out3_reco_6j_521176.root:tree")
background_file = uproot.open(ntuple_path + "data/out3_data_reco_5j.root:tree")
features = signal_file.keys()
df_sig = signal_file.arrays(library="pd")
df_bkg = background_file.arrays(library="pd")

# split into train validation and test here
logging.info('Train/validation/test splitting...')
y_sig = [0]*len(df_sig["eventWeight"])
y_bkg = [1]*len(df_bkg["eventWeight"])

# train:validation:test = 5:3:2
x_sig_train_val, x_sig_test, y_sig_train_val, y_sig_test = train_test_split(df_sig, y_sig, test_size=0.2, shuffle=False)
x_sig_train, x_sig_val, y_sig_train, y_sig_val = train_test_split(x_sig_train_val, y_sig_train_val, test_size=0.375, shuffle=False)

x_bkg_train_val, x_bkg_test, y_bkg_train_val, y_bkg_test = train_test_split(df_bkg, y_bkg, test_size=0.2, shuffle=False)
x_bkg_train, x_bkg_val, y_bkg_train, y_bkg_val = train_test_split(x_bkg_train_val, y_bkg_train_val, test_size=0.375, shuffle=False)

logging.info("Creating dataFrames")

df_sig_train = pd.DataFrame({str(var): x_sig_train[var] for var in features})
df_sig_val = pd.DataFrame({str(var): x_sig_val[var] for var in features})
df_sig_test = pd.DataFrame({str(var): x_sig_test[var] for var in features})
df_bkg_train = pd.DataFrame({str(var): x_bkg_train[var] for var in features})
df_bkg_val = pd.DataFrame({str(var): x_bkg_val[var] for var in features})
df_bkg_test = pd.DataFrame({str(var): x_bkg_test[var] for var in features})

df_sig_train["target"] = y_sig_train
df_sig_val["target"] = y_sig_val
df_sig_test["target"] = y_sig_test
df_bkg_train["target"] = y_bkg_train
df_bkg_val["target"] = y_bkg_val
df_bkg_test["target"] = y_bkg_test

#save_path = "/data/atlas/atlasdata3/maggiechen/gnn_project/split_files"
logging.info("Saving h5 files")
df_sig_train.to_hdf(h5_path + "sig_train.h5", key="sig_train", mode="w")
df_sig_val.to_hdf(h5_path + "sig_val.h5", key="sig_val", mode="w")
df_sig_test.to_hdf(h5_path + "sig_test.h5", key="sig_test", mode="w")
df_bkg_train.to_hdf(h5_path + "bkg_train.h5", key="bkg_train", mode="w")
df_bkg_val.to_hdf(h5_path + "bkg_val.h5", key="bkg_val", mode="w")
df_bkg_test.to_hdf(h5_path + "bkg_test.h5", key="bkg_test", mode="w")

