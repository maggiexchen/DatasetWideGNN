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
h5_path = user_config["kinematic_h5_path"]
ntuple_path = user_config["ntuple_path"]

signal = user_config["signal"]
feature_dim = user_config["feature_dim"]
assert signal in ["hhh", "LQ", "stau"], f"Invalid signal type: {signal}"

logging.info("signal: "+signal)
logging.info("input ntuple path: "+ntuple_path)
logging.info("output h5 data path: "+h5_path)

# load in input files
logging.info('Importing signal and background files...')
if signal == "hhh":
    background_types = [""]
    signal_file, background_file = misc.get_ntuples(ntuple_path, signal)
    features = signal_file.keys()
    df_sig = signal_file.arrays(library="pd")
    df_bkg = background_file.arrays(library="pd")

elif signal == "LQ":
    background_types = ["singletop", "ttbar"]
    signal_file, singletop_file, ttbar_file = misc.get_ntuples(ntuple_path, signal)
    features = signal_file.keys()
    df_sig = signal_file.arrays(library="pd")
    df_singletop = singletop_file.arrays(library="pd")
    df_ttbar = ttbar_file.arrays(library="pd")
    df_background = pd.concat([df_singletop, df_ttbar], axis=0)

print("Features in file: \n", features)
# TODO: making this mutli-class!

# split into train validation and test here
logging.info('Train/validation/test splitting...')
key = signal_file.keys()[0]
y_sig = [1]*len(df_sig[key])
# y_bkg = [0]*len(df_bkg["eventWeight"])
y_singletop = [0]*len(df_singletop[key])
y_ttbar = [0]*len(df_ttbar[key])

if signal == "LQ":
    df_sig["eventWeight"] = df_sig["lumiWeight"] * df_sig["xsec"] * df_sig["genWeight"] * df_sig["lumi"] / sum(df_sig["lumiWeight"] * df_sig["xsec"] * df_sig["genWeight"] * df_sig["lumi"])
    df_singletop["eventWeight"] = df_singletop["lumiWeight"] * df_singletop["xsec"] * df_singletop["genWeight"] * df_singletop["lumi"] / sum(df_singletop["lumiWeight"] * df_singletop["xsec"] * df_singletop["genWeight"] * df_singletop["lumi"])
    df_ttbar["eventWeight"] = df_ttbar["lumiWeight"] * df_ttbar["xsec"] * df_ttbar["genWeight"] * df_ttbar["lumi"] / sum(df_ttbar["lumiWeight"] * df_ttbar["xsec"] * df_ttbar["genWeight"] * df_ttbar["lumi"])
features.append("eventWeight")
# train:validation:test = 5:3:2
x_sig_train_val, x_sig_test, y_sig_train_val, y_sig_test = train_test_split(df_sig, y_sig, test_size=0.2, shuffle=False)
x_sig_train, x_sig_val, y_sig_train, y_sig_val = train_test_split(x_sig_train_val, y_sig_train_val, test_size=0.375, shuffle=False)

# x_bkg_train_val, x_bkg_test, y_bkg_train_val, y_bkg_test = train_test_split(df_bkg, y_bkg, test_size=0.2, shuffle=False)
# x_bkg_train, x_bkg_val, y_bkg_train, y_bkg_val = train_test_split(x_bkg_train_val, y_bkg_train_val, test_size=0.375, shuffle=False)

x_singletop_train_val, x_singletop_test, y_singletop_train_val, y_singletop_test = train_test_split(df_singletop, y_singletop, test_size=0.2, shuffle=False)
x_singletop_train, x_singletop_val, y_singletop_train, y_singletop_val = train_test_split(x_singletop_train_val, y_singletop_train_val, test_size=0.375, shuffle=False)

x_ttbar_train_val, x_ttbar_test, y_ttbar_train_val, y_ttbar_test = train_test_split(df_ttbar, y_ttbar, test_size=0.2, shuffle=False)
x_ttbar_train, x_ttbar_val, y_ttbar_train, y_ttbar_val = train_test_split(x_ttbar_train_val, y_ttbar_train_val, test_size=0.375, shuffle=False)

logging.info("Creating dataFrames")
df_sig_train = pd.DataFrame({str(var): x_sig_train[var] for var in features})
df_sig_val = pd.DataFrame({str(var): x_sig_val[var] for var in features})
df_sig_test = pd.DataFrame({str(var): x_sig_test[var] for var in features})
# df_bkg_train = pd.DataFrame({str(var): x_bkg_train[var] for var in features})
# df_bkg_val = pd.DataFrame({str(var): x_bkg_val[var] for var in features})
# df_bkg_test = pd.DataFrame({str(var): x_bkg_test[var] for var in features})
df_singletop_train = pd.DataFrame({str(var): x_singletop_train[var] for var in features})
df_singletop_val = pd.DataFrame({str(var): x_singletop_val[var] for var in features})
df_singletop_test = pd.DataFrame({str(var): x_singletop_test[var] for var in features})
df_ttbar_train = pd.DataFrame({str(var): x_ttbar_train[var] for var in features})
df_ttbar_val = pd.DataFrame({str(var): x_ttbar_val[var] for var in features})
df_ttbar_test = pd.DataFrame({str(var): x_ttbar_test[var] for var in features})

df_sig_train["target"] = y_sig_train
df_sig_val["target"] = y_sig_val
df_sig_test["target"] = y_sig_test
# df_bkg_train["target"] = y_bkg_train
# df_bkg_val["target"] = y_bkg_val
# df_bkg_test["target"] = y_bkg_test
df_singletop_train["target"] = y_singletop_train
df_singletop_val["target"] = y_singletop_val
df_singletop_test["target"] = y_singletop_test
df_ttbar_train["target"] = y_ttbar_train
df_ttbar_val["target"] = y_ttbar_val
df_ttbar_test["target"] = y_ttbar_test

#save_path = "/data/atlas/atlasdata3/maggiechen/gnn_project/split_files"
logging.info("Saving h5 files")
df_sig_train.to_hdf(h5_path + "sig_train.h5", key="sig_train", mode="w")
df_sig_val.to_hdf(h5_path + "sig_val.h5", key="sig_val", mode="w")
df_sig_test.to_hdf(h5_path + "sig_test.h5", key="sig_test", mode="w")
# df_bkg_train.to_hdf(h5_path + "bkg_train.h5", key="bkg_train", mode="w")
# df_bkg_val.to_hdf(h5_path + "bkg_val.h5", key="bkg_val", mode="w")
# df_bkg_test.to_hdf(h5_path + "bkg_test.h5", key="bkg_test", mode="w")
df_singletop_train.to_hdf(h5_path + "singletop_train.h5", key="singletop_train", mode="w")
df_singletop_val.to_hdf(h5_path + "singletop_val.h5", key="singletop_val", mode="w")
df_singletop_test.to_hdf(h5_path + "singletop_test.h5", key="singletop_test", mode="w")
df_ttbar_train.to_hdf(h5_path + "ttbar_train.h5", key="ttbar_train", mode="w")
df_ttbar_val.to_hdf(h5_path + "ttbar_val.h5", key="ttbar_val", mode="w")
df_ttbar_test.to_hdf(h5_path + "ttbar_test.h5", key="ttbar_test", mode="w")
