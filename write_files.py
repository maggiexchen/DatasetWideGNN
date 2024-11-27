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
os.makedirs(h5_path, exist_ok=True)

# TODO: assert. This should be "hhh" "LQ" or "stau"
signal = user_config["signal"]
if user_config["signal_mass"] is not None:
    signal_mass = str(user_config["signal_mass"])
else:
    signal_mass = ""
backgrounds = user_config["backgrounds"]
cuts = user_config["cuts"]
print("cuts ", cuts )

logging.info("signal: "+signal)
logging.info("backgrounds: "+str(backgrounds))
logging.info("input ntuple path: "+ntuple_path)
logging.info("output h5 data path: "+h5_path)

# load in input files
lumi_Run3 = 370
logging.info('Importing and writing signal '+str(signal)+' ...')
signal_file_path = ntuple_path + "GNNTree_"+str(signal)+"_"+signal_mass+".root"
signal_file = uproot.open(signal_file_path+":tree")
features = signal_file.keys()
df_sig = {str(signal):{}}
df_sig[signal] = signal_file.arrays(library="pd")
df_sig[signal]["target"] = [1]*len(df_sig[signal])
sig_initialWeights_arr = misc.get_histInitialWeights(signal_file_path)
df_sig[signal]["eventWeight"] = misc.calc_eventWeight(df_sig[signal], sig_initialWeights_arr, lumi_Run3)
print(signal, " event weights: ", df_sig[signal]["eventWeight"])
sig_cuts_condition = misc.cut_operation(df_sig[signal], cuts)
df_sig[signal] = df_sig[signal][sig_cuts_condition]
df_sig[signal].to_hdf(h5_path + str(signal)+".h5", key=str(signal), mode="w")

logging.info('Importing and writing background ')
df_bkgs = {}
for background in backgrounds:
    logging.info(str(background)+" ...")
    df_bkgs[str(background)] = {}
    background_file_path = ntuple_path + "GNNTree_"+str(background)+".root"
    background_file = uproot.open(background_file_path+":tree")
    df_bkgs[background] = background_file.arrays(library="pd")
    df_bkgs[background]["target"] = [0]*len(df_bkgs[background])
    bkgs_initialWeights_arr = misc.get_histInitialWeights(background_file_path)
    df_bkgs[background]["eventWeight"] = misc.calc_eventWeight(df_bkgs[background], bkgs_initialWeights_arr, lumi_Run3)
    print(background, " event weights: ", df_bkgs[background]["eventWeight"])
    bkg_cuts_condition = misc.cut_operation(df_bkgs[background], cuts)
    df_bkgs[background] = df_bkgs[background][bkg_cuts_condition]
    df_bkgs[background].to_hdf(h5_path + str(background)+".h5", key=str(background), mode="w")