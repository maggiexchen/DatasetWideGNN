import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
import utils.adj_mat as adj
import utils.misc as misc
import utils.normalisation as norm
import uproot

from embedding import EmbeddingNet

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset
torch.manual_seed(42)

import logging
logging.getLogger().setLevel(logging.INFO)
import argparse
import numpy as np
import pandas as pd
import json

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

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


device = torch.device('cpu')
args = GetParser()
user_config_path = args.userconfig
user_config = misc.load_config(user_config_path)
ntuple_path = user_config["ntuple_path"]
h5_path = user_config["h5_path"]
model_save_path = user_config["model_path"]
plot_path = user_config["plot_path"]
embedding_path = user_config["embedding_path"]
os.makedirs(embedding_path, exist_ok=True)
os.makedirs(h5_path, exist_ok=True)

signal = user_config["signal"]
if user_config["signal_mass"] is not None:
    signal_mass = str(user_config["signal_mass"])
else:
    signal_mass = ""
backgrounds = user_config["backgrounds"]
cuts = user_config["cuts"]

variable = user_config["variable"]
feature_dim = user_config["feature_dim"]
kinematics = misc.get_kinematics(variable, feature_dim)

# parameters of chosen model
embedding_dim = user_config["embedding_dim"]
margin = user_config["margin"]
penalty = user_config["penalty"]
radius = margin/2

# load the chosen model
logging.info("Loading trained model ...")
model = EmbeddingNet(input_dim=len(kinematics), embedding_dim=embedding_dim)
model.load_state_dict(torch.load(model_save_path + "embedding_"+str(embedding_dim)+"feats/EmbeddingNet_m"+str(margin)+"_r"+str(radius)+"_Lambda"+str(penalty)+"_model.pth")['model_state'])

# load the ntuples by signal/background type
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
if cuts is not None:
    df_sig[signal] = misc.cut_operation(df_sig[signal], cuts)

sig_scaler = StandardScaler()
standardised_sig_arr = torch.tensor(sig_scaler.fit_transform(df_sig[signal][kinematics]), dtype=torch.float32)
dataloader_sig = DataLoader(TensorDataset(standardised_sig_arr, torch.tensor(df_sig[signal]["target"].to_numpy(), dtype=torch.float32)), batch_size=128, shuffle=False)

sig_outputs = []
model.eval()
with torch.no_grad():
    for inputs, labels in dataloader_sig:
        sig_outputs.append(model(inputs))
sig_outputs = torch.cat(sig_outputs, dim=0)

# save the embedded features in h5 files
print(signal, " output shape ", sig_outputs.shape)
logging.info("Converting output tensor to numpy ...")
sig_outputs_numpy = sig_outputs.numpy()
for i in range(embedding_dim):
    df_sig[signal][r'feat_{i+1:02d}'] = sig_outputs.numpy()[:, i]

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
    if cuts is not None:
        df_bkgs[background] = misc.cut_operation(df_bkgs[background], cuts)

    bkg_scaler = StandardScaler()
    standardised_bkg_arr = torch.tensor(bkg_scaler.fit_transform(df_bkgs[background][kinematics]), dtype=torch.float32)
    dataloader_bkg = DataLoader(TensorDataset(standardised_bkg_arr, torch.tensor(df_bkgs[background]["target"].to_numpy(), dtype=torch.float32)), batch_size=128, shuffle=False)

    bkg_outputs = []
    model.eval()
    with torch.no_grad():
        for inputs, labels in dataloader_bkg:
            bkg_outputs.append(model(inputs))
    bkg_outputs = torch.cat(bkg_outputs, dim=0)

    # save the embedded features in h5 files
    print(background, " output shape ", bkg_outputs.shape)
    logging.info("Converting output tensor to numpy ...")
    bkg_outputs_numpy = bkg_outputs.numpy()
    for i in range(embedding_dim):
        df_bkgs[background][r'feat_{i+1:02d}'] = bkg_outputs.numpy()[:, i]

    df_bkgs[background].to_hdf(h5_path + str(background)+".h5", key=str(background), mode="w")


# # convert to torch tensor and standardise manually
# # load in to dataloader

# # data_loader = DataLoader(TensorDataset(full_x, full_y), batch_size=128, shuffle=False)

# # apply trained embedding model on full X
# logging.info("Evaluating full X with trained model ...")
# outputs = []
# model.eval()
# with torch.no_grad():
#     for inputs, labels in data_loader:
#         outputs.append(model(inputs))
# outputs = torch.cat(outputs, dim=0)

# sig_outputs = outputs[:sig_len, :]
# bkg_outputs = outputs[sig_len:, :]

# # save the embedded features in h5 files
# print("Output shape ", outputs.shape)
# logging.info("Converting output tensor to numpy ...")
# sig_outputs_numpy = sig_outputs.numpy()
# bkg_outputs_numpy = bkg_outputs.numpy()


# logging.info("Train val test split ...")
# # train:validation:test = 5:3:2
# x_sig_train_val, x_sig_test, y_sig_train_val, y_sig_test, wgts_sig_train_val, wgts_sig_test = train_test_split(sig_outputs, full_sig_labels, full_sig_wgts, test_size=0.2, shuffle=False)
# x_sig_train, x_sig_val, y_sig_train, y_sig_val, wgts_sig_train, wgts_sig_val = train_test_split(x_sig_train_val, y_sig_train_val, wgts_sig_train_val, test_size=0.375, shuffle=False)

# x_bkg_train_val, x_bkg_test, y_bkg_train_val, y_bkg_test, wgts_bkg_train_val, wgts_bkg_test = train_test_split(bkg_outputs, full_bkg_labels, full_bkg_wgts, test_size=0.2, shuffle=False)
# x_bkg_train, x_bkg_val, y_bkg_train, y_bkg_val, wgts_bkg_train, wgts_bkg_val = train_test_split(x_bkg_train_val, y_bkg_train_val, wgts_bkg_train_val, test_size=0.375, shuffle=False)


# df_sig_train = pd.DataFrame({f'feat_{i+1:02d}': x_sig_train[:, i] for i in range(embedding_dim)})
# df_sig_val = pd.DataFrame({f'feat_{i+1:02d}': x_sig_val[:, i] for i in range(embedding_dim)})
# df_sig_test = pd.DataFrame({f'feat_{i+1:02d}': x_sig_test[:, i] for i in range(embedding_dim)})
# df_bkg_train = pd.DataFrame({f'feat_{i+1:02d}': x_bkg_train[:, i] for i in range(embedding_dim)})
# df_bkg_val = pd.DataFrame({f'feat_{i+1:02d}': x_bkg_val[:, i] for i in range(embedding_dim)})
# df_bkg_test = pd.DataFrame({f'feat_{i+1:02d}': x_bkg_test[:, i] for i in range(embedding_dim)})

# df_sig_train["target"] = y_sig_train
# df_sig_val["target"] = y_sig_val
# df_sig_test["target"] = y_sig_test
# df_bkg_train["target"] = y_bkg_train
# df_bkg_val["target"] = y_bkg_val
# df_bkg_test["target"] = y_bkg_test

# df_sig_train["eventWeight"] = wgts_sig_train
# df_sig_val["eventWeight"] = wgts_sig_val
# df_sig_test["eventWeight"] = wgts_sig_test
# df_bkg_train["eventWeight"] = wgts_bkg_train
# df_bkg_val["eventWeight"] = wgts_bkg_val
# df_bkg_test["eventWeight"] = wgts_bkg_test

# # saving eventWeights


# logging.info("Saving h5 files")
# df_sig_train.to_hdf(embedding_path + "sig_train.h5", key="sig_train", mode="w")
# df_sig_val.to_hdf(embedding_path + "sig_val.h5", key="sig_val", mode="w")
# df_sig_test.to_hdf(embedding_path + "sig_test.h5", key="sig_test", mode="w")
# df_bkg_train.to_hdf(embedding_path + "bkg_train.h5", key="bkg_train", mode="w")
# df_bkg_val.to_hdf(embedding_path + "bkg_val.h5", key="bkg_val", mode="w")
# df_bkg_test.to_hdf(embedding_path + "bkg_test.h5", key="bkg_test", mode="w")


# # change config file so that these files can be found
# # use calc_distance.py to batch calculate distances in the embedded space
# # use linking_length.py to define linking length in the embedded space
