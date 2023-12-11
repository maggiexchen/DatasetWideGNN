import pandas as pd
import uproot
import numpy
import h5py
import json
import math
import random
import os
os.environ["NUMEXPR_MAX_THREADS"] = "16"
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
import mplhep as hep
import logging
logging.getLogger().setLevel(logging.INFO)
import argparse
import utils.normalisation as norm
import utils.torch_distances as dis

import torch
import torch.nn as nn
import torch.nn.functional as F
#import torch_geometric
#import torch_geometric.nn as geom_nn

import time
st = time.time()


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
      "--eff",
      "-e",
      type=float,
      required=True,
      help="Specify sig-sig efficiency for the linking length",
  )

  parser.add_argument(
      "--ssbb",
      action="store_true",
      help="Specify linking length between sig-sig and bkg-bkg",
  )

  parser.add_argument(
      "--sssb",
      action="store_true",
      help="Specify linking length between sig-sig and sig-bkg",
  )

  args = parser.parse_args()
  return args

args = GetParser()

if args.variable == "mass":
    # mass-based kinematics
    kinematics = ["mH1","mH2","mH3","mHHH"]
elif args.variable == "angular":
    # angular kinematics
    kinematics = ["dRH1","dRH2","dRH3","meandRBB"]
elif args.variable == "shape":
    # event shape kinematics
    kinematics = ["sphere3dv2b","sphere3dv2btrans","aplan3dv2b","theta3dv2b"]
elif args.variable == "combined":
    kinematics = ["mH1","mH2","mH3","mHHH","dRH1","dRH2","dRH3","meandRBB","sphere3dv2b","sphere3dv2btrans","aplan3dv2b","theta3dv2b"]
else:
    print("bruh")

# load training data file and kinematics
logging.info('Importing signal and background files...')
file_path = "/data/atlas/atlasdata3/maggiechen/gnn_project/split_files/"
df_sig = pd.read_hdf(file_path+"sig_train.h5", key="sig_train")
df_sig = df_sig.sample(n=10)
df_sig_wgts = df_sig["eventWeight"]
sig_label = [1]*len(df_sig)
df_sig = df_sig[kinematics]
df_bkg = pd.read_hdf(file_path+"bkg_train.h5", key="bkg_train")
df_bkg = df_bkg.sample(n=10)
df_bkg_wgts = df_bkg["eventWeight"]
bkg_label = [0]*len(df_bkg)
df_bkg = df_bkg[kinematics]

logging.info("MAD scaling...")
# normalise kinematic values using MAD scaling
for var in kinematics:
    df_sig.loc[:, var], df_bkg.loc[:, var] = norm.MAD_norm(df_sig.loc[:, var], df_bkg.loc[:, var])

logging.info("Converting torch tensors...")
# convert pd dataframes to torch tensors
torch_sig = torch.tensor(df_sig.values, dtype=torch.float32)
torch_bkg = torch.tensor(df_bkg.values, dtype=torch.float32)
torch_sig_wgts = torch.tensor(df_sig_wgts.values, dtype=torch.float32)
torch_bkg_wgts = torch.tensor(df_bkg_wgts.values, dtype=torch.float32)
# concatenating signal and background events / weights
torch_all = torch.concat((torch_sig, torch_bkg), dim=0)
truth_labels = torch.tensor(numpy.concatenate((sig_label, bkg_label)), dtype=torch.float32)
torch_wgts = torch.concat((torch_sig_wgts, torch_bkg_wgts), dim=0)
print("Shape of signal + background tensor", torch_all.size())


# read in linking length calculated from sampled training data
sigsig_eff = args.eff
with open('/data/atlas/atlasdata3/maggiechen/gnn_project/linking_lengths/'+args.variable+"_"+args.distance+"_linking_length.json", 'r') as lfile:
    length_dict = json.load(lfile)
    eff = length_dict["sigsig_eff"]
    ss_bb_lengths = length_dict["ss_bb_length"]
    ss_sb_lengths = length_dict["ss_sb_length"]
if args.ssbb:
    linking_length = ss_bb_lengths[eff.index(sigsig_eff)]
elif args.sssb:
    linking_length = ss_sb_lengths[eff.index(sigsig_eff)]
else:
    print("Please specify distributions to generate the linking length (ssbb or sssb)!")

# calculate distances in chunks
logging.info('Calculating distances in batches...')
chunksize = 20000
nchunk = math.ceil(len(torch_all)/chunksize)

def create_adj_mat(a, length):
    return (a < length).float()


def create_node_wgts(a, b):
    a_col = a.view(-1,1)
    b_col = b.view(1,-1)
    outer = torch.matmul(a_col, b_col)
    return torch.transpose(outer, 0, 1)

# initialise adjacency matrix
adj_mat = torch.empty((0, len(torch_all)))
node_wgts = torch.empty((0, len(torch_wgts)))

# calculating distances and cutting with linking length in chunks 
for i in range(nchunk):
    print("Batch number ", i)
    # create subset of sig+bkg dataset
    torch_all_subset = torch_all[(i*chunksize):(i+1)*chunksize]
    torch_wgts_subset = torch_wgts[(i*chunksize):(i+1)*chunksize]
    # calculate distances
    if args.distance == "euclidean":
        distance_subset = dis.euclidean(torch_all, torch_all_subset)
    elif args.distance == "cityblock":
        distance_subset = dis.cityblock(torch_all, torch_all_subset)
    elif args.distance == "cosine":
        distance_subset = dis.cosine(torch_all, torch_all_subset)

    adj_mat_subset = create_adj_mat(distance_subset, linking_length)
    print("adj mat subset", adj_mat_subset.size())
    print("adj mat", adj_mat.size())
    adj_mat = torch.concat((adj_mat_subset, adj_mat), dim=0)
    
    # get calculate node weights by batches
    node_wgts_subset = create_node_wgts(torch_wgts, torch_wgts_subset)
    print("node wgts subset", node_wgts_subset.size())
    print("node wgts", node_wgts.size())
    node_wgts = torch.concat((node_wgts_subset, node_wgts), dim=0)

print(f"Time taken for adjacency matrix generation: {time.time() - st}")


# Big ass GNN class like an adult
class GCNLayer(nn.Module):
    # TODO add attention layer ...
    def __init__(self, dim_in, dim_out, node_weights):
        """
        dim_in: dimension of input node features
        dim_out: dimension of output features (for binary classification is 1)
        node_weights: weights (event weights) of the nodes (events)
        """
        super(GCNLayer, self).__init__()
        self.train_weight = nn.Parameter(torch.FloatTensor(dim_in, dim_out))
        self.train_bias = nn.Parameter(torch.FloatTensor(dim_out))
        self.node_weights = nn.Parameter(node_weights)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.train_weight.data)
        nn.init.zeros_(self.train_bias.data)

    def forward(self, x, adjacency_matrix):
        weighted_node_features = torch.mul(x, self.node_weights[:, None])
        output = torch.matmul(adjacency_matrix, torch.matmul(weighted_node_features, self.train_weight)) + self.train_bias
        #output = F.relu(output)
        return output

class GCNClassifier(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, node_weights):
        """
        input_sizes: dimension of input node features
        hidden_sizes: number of nodes in hidden graph layers as a list
        output_sizes: dimension of output features (for binary classification is 1)
        """
        super(GCNClassifier, self).__init__()
        # hidden layers
        self.hidden_layers = nn.ModuleList([GCNLayer(input_size, hidden_sizes[0], node_weights)])
        self.hidden_layers.append(nn.ReLU())
        for i in range(1, len(hidden_sizes)):
            self.hidden_layers.append(GCNLayer(hidden_sizes[i-1], hidden_sizes[i], node_weights))
            self.hidden_activation = nn.ReLU()
        # output layer
        self.output_layer = GCNLayer(hidden_sizes[-1], output_size, node_weights)
        self.output_activation = nn.Sigmoid()

    def forward(self, x, adjacency_matrix):
        for layer in self.hidden_layers:
            if isinstance(layer, GCNLayer):
                x = layer(x, adjacency_matrix)
                x = self.hidden_activation(x)
            else:
                x = layer(x)
                x = self.hidden_activation(x)
        output = self.output_layer(x, adjacency_matrix)
        output = self.output_activation(output)
        return output

input_size = len(kinematics)
hidden_sizes = [12, 12, 12]
LR = 0.001
epochs = 30

gcn_model = GCNClassifier(input_size=input_size, hidden_sizes=hidden_sizes, output_size=1, node_weights=torch_wgts)
train_loss = nn.CrossEntropyLoss()
optimiser = torch.optim.Adam(gcn_model.parameters(), lr=LR)

for epoch in range(epochs):
    outputs = gcn_model(torch_all, adj_mat)
    outputs = torch.sigmoid(outputs)
    loss = train_loss(outputs.squeeze(), truth_labels.squeeze())
    optimiser.zero_grad()
    loss.backward()
    optimiser.step()
    print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}')
print("Final predictions", outputs)
print("Truth labels", truth_labels)
