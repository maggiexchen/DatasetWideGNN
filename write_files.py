"""Module to tranform root ntuples to h5 files, with correct weights, for signal and backgrounds."""
import os
import argparse
import logging
from glob import glob

import utils.misc as misc
import utils.plotting as plotting

import pandas as pd
import uproot

logging.getLogger().setLevel(logging.INFO)

parser = argparse.ArgumentParser(
    description="Reading ntuples command line options."
)

parser.add_argument(
    "--userconfig",
    "-u",
    type=str,
    required=True,
    help="Specify the config for the user",
)

parser.add_argument(
    "--plotEventWeights",
    "-p",
    action = "store_true",
    default = False,
    help="Specify the config for the user",
)

args = parser.parse_args()

user_config_path = args.userconfig
user_config = misc.load_config(user_config_path)
plot_event_weights = args.plotEventWeights

h5_path = user_config["kinematic_h5_path"]
ntuple_path = user_config["ntuple_path"]
os.makedirs(h5_path, exist_ok=True)

signal = user_config["signal"]
if user_config["signal_mass"] is not None:
    signal_mass = str(user_config["signal_mass"])
else:
    signal_mass = ""
backgrounds = user_config["backgrounds"]
cuts = user_config["cuts"]
cutstring = misc.get_cutstring(cuts)

logging.info("signal: %s", signal)
logging.info("backgrounds: %s", str(backgrounds))
logging.info("input ntuple path: %s", ntuple_path)
logging.info("output h5 data path: %s", h5_path)
os.makedirs(h5_path, exist_ok=True)

# load in input files
lumi_Run3 = 370.
logging.info('Importing and writing signal %s ...', str(signal))
if signal_mass is not None:
    print("signal mass ", signal_mass)
    signal_mass_str = "_mass" + str(signal_mass) + "*"
else:
    signal_mass_str = "_*"
signal_file_paths = glob(ntuple_path + "GNNTree_" + str(signal)
                         + signal_mass_str + ".root")
print(len(signal_file_paths), " signal files found ...")
signal_features = uproot.open(signal_file_paths[0]+":tree").keys()
data_list = []
weight_list = []

for signal_root_file in signal_file_paths:
    print("Reading ", signal_root_file)
    signal_file = uproot.open(signal_root_file+":tree")
    x_pd = signal_file.arrays(library="pd")
    data_list.append(x_pd)
    sig_initialWeights_arr = misc.get_hist_initial_weights(signal_root_file)
    weight_list.append(misc.calc_event_weight(x_pd, sig_initialWeights_arr, lumi_Run3))

df_sig = {str(signal):{}}
df_sig[signal] = pd.concat(data_list, ignore_index=True)
df_sig[signal]["target"] = [1]*len(df_sig[signal])
df_sig[signal]["eventWeight"] = pd.concat(weight_list, ignore_index=True)
print("Total ", signal, " events before cuts: ", len(df_sig[signal]))

if cuts is not None:
    df_sig[signal] = misc.cut_operation(df_sig[signal], cuts)
    print("Total ", signal, " events after cuts: ", len(df_sig[signal]))

df_sig[signal].to_hdf(h5_path + str(signal) + "_" + signal_mass + ".h5",
                      key=str(signal), mode="w")

logging.info('Importing and writing background ')
df_bkgs = {}

for background in backgrounds:
    logging.info("%s ...", str(background))
    df_bkgs[str(background)] = {}
    background_file_paths = glob(ntuple_path + "GNNTree_"+str(background)+"*.root")
    background_features = uproot.open(background_file_paths[0]+":tree").keys()
    feature_diff = [item for item in background_features if item not in signal_features]
    if feature_diff and feature_diff != ["nEvents"]:
        raise AssertionError(f"Signal and background features must be the same! \
                             Difference: {feature_diff}")
    data_list = []
    weight_list = []

    for background_root_file in background_file_paths:
        print("Reading ", background_root_file)
        background_file = uproot.open(background_root_file+":tree")
        x_pd = background_file.arrays(library="pd")
        data_list.append(x_pd)
        bkg_initialWeights_arr = misc.get_hist_initial_weights(background_root_file)
        weight_list.append(misc.calc_event_weight(x_pd, bkg_initialWeights_arr, lumi_Run3))

    df_bkgs[background] = pd.concat(data_list, ignore_index=True)
    df_bkgs[background]["target"] = [0]*len(df_bkgs[background])
    df_bkgs[background]["eventWeight"] = pd.concat(weight_list, ignore_index=True)
    print("Total ", background, " events before cuts: ", len(df_bkgs[background]))

    if cuts is not None:
        df_bkgs[background] = misc.cut_operation(df_bkgs[background], cuts)
        print("Total ", background, " events after cuts: ", len(df_bkgs[background]))

    df_bkgs[background].to_hdf(h5_path + str(background)+".h5", key=str(background), mode="w")

if plot_event_weights:

    logging.info("Plotting eventWeights ...")
    plotting.plot_event_weights(df_sig, signal, df_bkgs, backgrounds,
                                h5_path, signal_mass, cutstring)
