import pandas as pd
import uproot
import numpy
import h5py
import json
import random
import tensorflow as tf
import os
os.environ["NUMEXPR_MAX_THREADS"] = "16"
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
import mplhep as hep
import logging
logging.getLogger().setLevel(logging.INFO)
import argparse
import normalisation as norm


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
    #kinematics = ["mH1","mH2","mH3","mHHH","mHcosTheta","meanmH","rmsmH","meanmBB","rmsmBB","meanPt","rmsPt","ht","massfraceta","massfracphi","massfracraw"]
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

logging.info("Converting tf tensors...")
# convert pd dataframes to tensors
tf_sig = tf.convert_to_tensor(df_sig, dtype_hint = 'float32')
tf_bkg = tf.convert_to_tensor(df_bkg, dtype_hint = 'float32')

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
# generate adjacency matrix chunks using linking length
# save adjecency matrix chunks