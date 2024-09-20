import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
import utils.adj_mat as adj
import utils.misc as misc
import utils.normalisation as norm

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
h5_path = user_config["h5_path"]
model_save_path = user_config["model_path"]
plot_path = user_config["plot_path"]
embedding_path = user_config["embedding_path"]
os.makedirs(embedding_path, exist_ok=True)

variable = user_config["variable"]
kinematics = misc.get_kinematics(variable, 1)
signal = user_config["signal"]

# parameters of chosen model
embedding_dim = user_config["embedding_dim"]
margin = user_config["margin"]
penalty = user_config["penalty"]
radius = margin/2

# load the chosen model
logging.info("Loading trained model ...")
model = EmbeddingNet(input_dim=len(kinematics), embedding_dim=embedding_dim)
model.load_state_dict(torch.load(model_save_path + "embedding_"+str(embedding_dim)+"feats/EmbeddingNet_m"+str(margin)+"_r"+str(radius)+"_Lambda"+str(penalty)+"_model.pth")['model_state'])

# load full X
logging.info('Importing full X...')
# normalised signal and background kinematics
train_sig, train_bkg, train_x, train_sig_wgts, train_bkg_wgts, train_sig_labels, train_bkg_labels = adj.data_loader(h5_path, plot_path, "train", kinematics, plot=False, signal=signal)
val_sig, val_bkg, val_x, val_sig_wgts, val_bkg_wgts, val_sig_labels, val_bkg_labels = adj.data_loader(h5_path, plot_path, "val", kinematics, plot=False, signal=signal)
test_sig, test_bkg, test_x, test_sig_wgts, test_bkg_wgts, test_sig_labels, test_bkg_labels = adj.data_loader(h5_path, plot_path, "test", kinematics, plot=False, signal=signal)

full_sig = torch.cat((train_sig, val_sig, test_sig), dim=0)
full_sig_labels = torch.cat((train_sig_labels, val_sig_labels, test_sig_labels))
full_sig_wgts = torch.cat((train_sig_wgts, val_sig_wgts, test_sig_wgts))

full_bkg = torch.cat((train_bkg, val_bkg, test_bkg), dim=0)
full_bkg_labels = torch.cat((train_bkg_labels, val_bkg_labels, test_bkg_labels))
full_bkg_wgts = torch.cat((train_bkg_wgts, val_bkg_wgts, test_bkg_wgts))

sig_len = len(full_sig)
bkg_len = len(full_bkg)

full_x = torch.cat((full_sig, full_bkg), dim=0).to(device)
del full_sig
del full_bkg

full_y = torch.cat((full_sig_labels, full_bkg_labels), dim=0).to(device)
full_y = full_y.float()
full_wgts = torch.cat((torch.cat((train_sig_wgts, val_sig_wgts, test_sig_wgts), dim=0), torch.cat((train_bkg_wgts, val_bkg_wgts, test_bkg_wgts), dim=0)), dim=0)#.cuda()

data_loader = DataLoader(TensorDataset(full_x, full_y), batch_size=128, shuffle=False)

# apply trained embedding model on full X
logging.info("Evaluating full X with trained model ...")
outputs = []
model.eval()
with torch.no_grad():
    for inputs, labels in data_loader:
        outputs.append(model(inputs))
outputs = torch.cat(outputs, dim=0)

sig_outputs = outputs[:sig_len, :]
bkg_outputs = outputs[sig_len:, :]

# save the embedded features in h5 files
print("Output shape ", outputs.shape)
logging.info("Converting output tensor to numpy ...")
sig_outputs_numpy = sig_outputs.numpy()
bkg_outputs_numpy = bkg_outputs.numpy()


logging.info("Train val test split ...")
# train:validation:test = 5:3:2
x_sig_train_val, x_sig_test, y_sig_train_val, y_sig_test, wgts_sig_train_val, wgts_sig_test = train_test_split(sig_outputs, full_sig_labels, full_sig_wgts, test_size=0.2, shuffle=False)
x_sig_train, x_sig_val, y_sig_train, y_sig_val, wgts_sig_train, wgts_sig_val = train_test_split(x_sig_train_val, y_sig_train_val, wgts_sig_train_val, test_size=0.375, shuffle=False)

x_bkg_train_val, x_bkg_test, y_bkg_train_val, y_bkg_test, wgts_bkg_train_val, wgts_bkg_test = train_test_split(bkg_outputs, full_bkg_labels, full_bkg_wgts, test_size=0.2, shuffle=False)
x_bkg_train, x_bkg_val, y_bkg_train, y_bkg_val, wgts_bkg_train, wgts_bkg_val = train_test_split(x_bkg_train_val, y_bkg_train_val, wgts_bkg_train_val, test_size=0.375, shuffle=False)


df_sig_train = pd.DataFrame({f'feat_{i+1:02d}': x_sig_train[:, i] for i in range(embedding_dim)})
df_sig_val = pd.DataFrame({f'feat_{i+1:02d}': x_sig_val[:, i] for i in range(embedding_dim)})
df_sig_test = pd.DataFrame({f'feat_{i+1:02d}': x_sig_test[:, i] for i in range(embedding_dim)})
df_bkg_train = pd.DataFrame({f'feat_{i+1:02d}': x_bkg_train[:, i] for i in range(embedding_dim)})
df_bkg_val = pd.DataFrame({f'feat_{i+1:02d}': x_bkg_val[:, i] for i in range(embedding_dim)})
df_bkg_test = pd.DataFrame({f'feat_{i+1:02d}': x_bkg_test[:, i] for i in range(embedding_dim)})

df_sig_train["target"] = y_sig_train
df_sig_val["target"] = y_sig_val
df_sig_test["target"] = y_sig_test
df_bkg_train["target"] = y_bkg_train
df_bkg_val["target"] = y_bkg_val
df_bkg_test["target"] = y_bkg_test

df_sig_train["eventWeight"] = wgts_sig_train
df_sig_val["eventWeight"] = wgts_sig_val
df_sig_test["eventWeight"] = wgts_sig_test
df_bkg_train["eventWeight"] = wgts_bkg_train
df_bkg_val["eventWeight"] = wgts_bkg_val
df_bkg_test["eventWeight"] = wgts_bkg_test

# saving eventWeights


logging.info("Saving h5 files")
df_sig_train.to_hdf(embedding_path + "sig_train.h5", key="sig_train", mode="w")
df_sig_val.to_hdf(embedding_path + "sig_val.h5", key="sig_val", mode="w")
df_sig_test.to_hdf(embedding_path + "sig_test.h5", key="sig_test", mode="w")
df_bkg_train.to_hdf(embedding_path + "bkg_train.h5", key="bkg_train", mode="w")
df_bkg_val.to_hdf(embedding_path + "bkg_val.h5", key="bkg_val", mode="w")
df_bkg_test.to_hdf(embedding_path + "bkg_test.h5", key="bkg_test", mode="w")


# change config file so that these files can be found
# use calc_distance.py to batch calculate distances in the embedded space
# use linking_length.py to define linking length in the embedded space
