import pandas as pd
import uproot
import numpy
import h5py
import json
import math
import random
import torch
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
df_sig = pd.read_hdf(file_path+"sig_train.h5", key="sig_train")[kinematics]
df_bkg = pd.read_hdf(file_path+"bkg_train.h5", key="bkg_train")[kinematics]

logging.info("MAD scaling...")
# normalise kinematic values using MAD scaling
for var in kinematics:
    df_sig.loc[:, var], df_bkg.loc[:, var] = norm.MAD_norm(df_sig.loc[:, var], df_bkg.loc[:, var])

logging.info("Converting torch tensors...")
# convert pd dataframes to torch tensors
torch_sig = torch.tensor(df_sig.values, dtype=torch.float32)
torch_bkg = torch.tensor(df_bkg.values, dtype=torch.float32)

# concatenating signal and background events
torch_all = torch.concat((torch_sig, torch_bkg), dim=0)

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

# initialise adjacency matrix
adj_mat = torch.empty((0,len(torch_all)))

# calculating distances and cutting with linking length in chunks 
for i in range(nchunk):
    # create subset of sig+bkg dataset
    torch_all_subset = torch_all[(i*chunksize):(i+1)*chunksize]
    # calculate distances
    if args.distance == "euclidean":
        distance_subset = dis.euclidean(torch_all, torch_all_subset)
    elif args.distance == "cityblock":
        distance_subset = dis.cityblock(torch_all, torch_all_subset)
    elif args.distance == "cosine":
        distance_subset = dis.cosine(torch_all, torch_all_subset)

    adj_mat_subset = create_adj_mat(distance_subset, linking_length)
    adj_mat = torch.concat((adj_mat_subset, adj_mat), dim=0)

print(f"Time taken for adjacency matrix generation: {time.time() - st}" )

