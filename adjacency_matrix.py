import pandas as pd
import uproot
import numpy
import h5py
import json
import math
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
import distances as dis
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

# concatenating signal and background events
tf_all = tf.concat([tf_sig, tf_bkg], axis=0)

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
chunksize = 10000
nchunk = math.ceil(len(tf_all)/chunksize)

# calculating distances and cutting with linking length in chunks 
for i in range(nchunk):
    # initialise adjacency matrix
    adj_mat = tf.reshape((), (0,len(kinematics)))
    # create subset of sig+bkg dataset
    tf_all_subset = tf_all[(i*chunksize):(i+1)*chunksize]
    # calculate distances
    if args.distance == "euclidean":
        distance_subset = dis.euclidean(tf_all, tf_all_subset)
    elif args.distance == "cityblock":
        distance_subset = dis.cityblock(tf_all, tf_all_subset)
    elif args.distance == "cosine":
        distance_subset = dis.cosine(tf_all, tf_all_subset)

    adj_mat_subset = tf.cast(distance_subset<linking_length, "float32")
    print(adj_mat_subset)
    
    print(f"--------> {i}: time taken so far: {time.time() - st}")

    # convert back to pd dataframes (honestly this is the bit that takes the longest and is the most unnecessary part)
    adj_mat_subset = pd.DataFrame(adj_mat_subset)
    # can also add event weights and labels etc. later
    # write adj_mat to file
    adj_mat_subset.to_hdf(f"adjacency_matrix/test__v{i}.h5", 'df')

print(f"-------->{i}: total time taken: {time.time() - st}" )