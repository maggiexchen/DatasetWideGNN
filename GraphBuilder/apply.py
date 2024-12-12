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
radius_name = margin/2
print("Connecting at radius ", radius, "with a training margin of ", margin)

# load the chosen model
logging.info("Loading trained model ...")
model = EmbeddingNet(input_dim=len(kinematics), embedding_dim=embedding_dim)
model.load_state_dict(torch.load(model_save_path + "embedding_"+str(embedding_dim)+"feats/EmbeddingNet_m"+str(margin)+"_r"+str(radius_name)+"_Lambda"+str(penalty)+"_model.pth")['model_state'])

# load the ntuples by signal/background type
# load in input files
lumi_Run3 = 370
logging.info('Importing and writing signal '+str(signal)+' ...')
signal_file_path = ntuple_path + "GNNTree_"+str(signal)+"_"+signal_mass+".root"
signal_file = uproot.open(signal_file_path+":tree")
features = signal_file.keys()
df_sig = {str(signal):{}}
df_sig[signal] = signal_file.arrays(library="pd")
if cuts is not None:
    df_sig[signal] = misc.cut_operation(df_sig[signal], cuts)
indices = [len(df_sig[signal])]
df_sig[signal]["target"] = [1]*len(df_sig[signal])
sig_initialWeights_arr = misc.get_histInitialWeights(signal_file_path)
df_sig[signal]["eventWeight"] = misc.calc_eventWeight(df_sig[signal], sig_initialWeights_arr, lumi_Run3)
tmp_df = df_sig[signal]
tmp_y = df_sig[signal]["target"]

logging.info('Importing and writing background ')
df_bkgs = {}
for background in backgrounds:
    logging.info(str(background)+" ...")
    df_bkgs[str(background)] = {}
    background_file_path = ntuple_path + "GNNTree_"+str(background)+".root"
    background_file = uproot.open(background_file_path+":tree")
    df_bkgs[background] = background_file.arrays(library="pd")
    if cuts is not None:
        df_bkgs[background] = misc.cut_operation(df_bkgs[background], cuts)
    df_bkgs[background]["target"] = [0]*len(df_bkgs[background])
    bkgs_initialWeights_arr = misc.get_histInitialWeights(background_file_path)
    df_bkgs[background]["eventWeight"] = misc.calc_eventWeight(df_bkgs[background], bkgs_initialWeights_arr, lumi_Run3)
    print(background, " event weights: ", df_bkgs[background]["eventWeight"])
    indices.append(len(df_bkgs[background]))
    
    tmp_df = pd.concat([tmp_df, df_bkgs[background]])
    tmp_y = pd.concat([tmp_y, df_bkgs[background]["target"]])

idx = [indices[0]]
for i in range(1, len(indices)):
    idx.append(idx[-1]+indices[i])
print("checking indices ", idx)
scaler = StandardScaler()
standardised_df = torch.tensor(scaler.fit_transform(tmp_df[kinematics], tmp_y), dtype=torch.float32)
data_loader = DataLoader(TensorDataset(standardised_df, torch.tensor(tmp_y.to_numpy(), dtype=torch.float32)), batch_size=128, shuffle=False)

outputs = []
model.eval()
with torch.no_grad():
    for inputs, labels in data_loader:
        outputs.append(model(inputs))
outputs = torch.cat(outputs, dim=0)


sig_outputs = outputs[:idx[0], :]
for i in range(embedding_dim):
    df_sig[signal][f'feat_{i+1:02d}'] = sig_outputs.numpy()[:, i]
    df_sig[signal].to_hdf(embedding_path + str(signal)+".h5", key=str(signal), mode="w")

for b, background in enumerate(backgrounds):
    bkg_outputs = outputs[idx[b]:idx[b+1]]
    for i in range(embedding_dim):
        df_bkgs[background][f'feat_{i+1:02d}'] = bkg_outputs.numpy()[:, i]
        df_bkgs[background].to_hdf(embedding_path + str(background)+".h5", key=str(background), mode="w")
