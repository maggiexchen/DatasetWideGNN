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
import tensorflow as tf
import distances as dis


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
      "--sample",
      "-s",
      action="store_true",
      help="Specify whether the datasets are sampled",
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

# median absolute deviation scaling
def MAD_norm(dist1, dist2):
    dist2_median = numpy.median(dist2)
    dist2_MAD = numpy.median(abs(dist2-dist2_median))
    norm_dist1 = (dist1 - dist2_median)/dist2_MAD
    norm_dist2 = (dist2 - dist2_median)/dist2_MAD

    return norm_dist1, norm_dist2

# load in input files
logging.info('Importing signal and background files...')
file_path = "/data/atlas/atlasdata3/maggiechen/gnn_project/split_files/"
df_sig = pd.read_hdf(file_path+"sig_train.h5", key="sig_train")
df_bkg = pd.read_hdf(file_path+"bkg_train.h5", key="bkg_train")

# randomly sample from the training datasets for linking length calculation if specified
if args.sample == True:
    logging.info("Sampling...")
    df_sig = df_sig.sample(n=1500)
    df_bkg = df_bkg.sample(n=3000)

logging.info("MAD scaling...")
# normalise kinematic values using MAD scaling
for var in kinematics:
    df_sig.loc[:, var], df_bkg.loc[:, var] = MAD_norm(df_sig.loc[:, var], df_bkg.loc[:, var])

# convert pandas dataframes to tf tensors
# only the kinematics used in distance calculation and weights need to be converted to tensors here for matrix multiplications

logging.info("Converting to tf tesnors...")
tf_sig = df_sig[kinematics]
tf_bkg = df_bkg[kinematics]
tf_sig = tf.convert_to_tensor(tf_sig, dtype_hint="float32")
tf_bkg = tf.convert_to_tensor(tf_bkg, dtype_hint="float32")
y_sig = df_sig["target"]
y_bkg = df_bkg["target"]

# mutliple events kinematics by the corresponding event weights and calcualte distances
logging.info('Getting MC event weights and calcualte weight matrix ...')
# The scale factor that scales 5b data down to the expected 6b yields, this is just taken as the ratio between 5b data/4b data for now
SF_4b5b = 0.07
sig_wgt = tf.convert_to_tensor(df_sig["eventWeight"], dtype_hint="float32")
bkg_wgt = tf.convert_to_tensor(df_bkg["eventWeight"]*SF_4b5b, dtype_hint="float32")
sigsig_wgt = tf.matmul(tf.reshape(sig_wgt, [-1,1]), tf.reshape(sig_wgt, [-1,1]), transpose_b=True)
sigbkg_wgt = tf.matmul(tf.reshape(sig_wgt, [-1,1]), tf.reshape(bkg_wgt, [-1,1]), transpose_b=True)
bkgbkg_wgt = tf.matmul(tf.reshape(bkg_wgt, [-1,1]), tf.reshape(bkg_wgt, [-1,1]), transpose_b=True)

# calculate distances
logging.info('Calculating distances...')
if args.distance == "euclidean":
    sigsig = dis.euclidean(tf_sig, tf_sig)
    sigbkg = dis.euclidean(tf_sig, tf_bkg)
    bkgbkg = dis.euclidean(tf_bkg, tf_bkg)

elif args.distance == "cityblock":
    sigsig = dis.cityblock(tf_sig, tf_sig)
    sigbkg = dis.cityblock(tf_sig, tf_bkg)
    bkgbkg = dis.cityblock(tf_bkg, tf_bkg)

elif args.distance == "cosine":
    sigsig = dis.cosine(tf_sig, tf_sig)
    sigbkg = dis.cosine(tf_sig, tf_bkg)
    bkgbkg = dis.cosine(tf_bkg, tf_bkg)

else:
    print("Specify a valid distance please!")

logging.info("Checking for NaNs in distances ... ")
print(tf.reduce_sum(tf.cast(tf.math.is_nan(sigsig), tf.int32)))
print(tf.reduce_sum(tf.cast(tf.math.is_nan(sigbkg), tf.int32)))
print(tf.reduce_sum(tf.cast(tf.math.is_nan(bkgbkg), tf.int32)))

# plot the (sampled) MAD-normed distances
logging.info("Converting distance and weight tensors to numpy arrays for saving and plotting ... ")
np_sigsig = sigsig.numpy().flatten()
np_sigbkg = sigbkg.numpy().flatten()
np_bkgbkg = bkgbkg.numpy().flatten()
np_sigsig_wgt = sigsig_wgt.numpy().flatten()
np_sigbkg_wgt = sigbkg_wgt.numpy().flatten()
np_bkgbkg_wgt = bkgbkg_wgt.numpy().flatten()

logging.info('Writing to h5...')
save_path = "/data/atlas/atlasdata3/maggiechen/gnn_project/distances/"

sigsig_file = save_path+str(args.variable)+"_"+str(args.distance)+"_sigsig_sampled_train.h5"
sigbkg_file = save_path+str(args.variable)+"_"+str(args.distance)+"_sigbkg_sampled_train.h5"
bkgbkg_file = save_path+str(args.variable)+"_"+str(args.distance)+"_bkgbkg_sampled_train.h5"
f_sigsig = h5py.File(sigsig_file, "w")
f_sigbkg = h5py.File(sigbkg_file, "w")
f_bkgbkg = h5py.File(bkgbkg_file, "w")

dtype = numpy.dtype([('distance', numpy.float32), ('weight', numpy.float32)])
sigsig_dset = f_sigsig.create_dataset("sigsig", shape=(len(np_sigsig),), dtype=dtype, chunks=True, compression="gzip")
sigbkg_dset = f_sigbkg.create_dataset("sigbkg", shape=(len(np_sigbkg),), dtype=dtype, chunks=True, compression="gzip")
bkgbkg_dset = f_bkgbkg.create_dataset("bkgbkg", shape=(len(np_bkgbkg),), dtype=dtype, chunks=True, compression="gzip")
# writing distances, and weights in chunks

sigsig_dset['distance'] = np_sigsig
sigsig_dset['weight'] = np_sigsig_wgt
sigbkg_dset['distance'] = np_sigbkg
sigbkg_dset['weight'] = np_sigbkg_wgt
bkgbkg_dset['distance'] = np_bkgbkg
bkgbkg_dset['weight'] = np_bkgbkg_wgt

f_sigsig.close()
f_sigbkg.close()
f_bkgbkg.close()

logging.info("Plotting ...")
nBins=100
if args.distance == "cityblock":
    x_max = 40
elif args.distance == "euclidean":
    x_max = 20
elif args.distance == "cosine":
    x_max = 2
else:
    print('Eh?')
binning=numpy.linspace(0,max(np_bkgbkg),nBins)
fig, ax = plt.subplots()
#binning = numpy.linspace(0,max(sigbkg),nBins)
ax.hist(np_sigsig, bins=binning, label="sig-sig", alpha=0.5, weights=np_sigsig_wgt, density=True)
ax.hist(np_sigbkg, bins=binning, label="sig-bkg", alpha=0.5, weights=np_sigbkg_wgt, density=True)
ax.hist(np_bkgbkg, bins=binning, label="bkg-bkg", alpha=0.5, weights=np_bkgbkg_wgt, density=True)
ax.legend(loc='upper right')
ax.set_xlabel(str(args.variable)+"_"+str(args.distance) +" distance", loc="right")
ax.set_ylabel("Normalised No. Events", loc="top")
fig.savefig("/data/atlas/atlasdata3/maggiechen/gnn_project/plots/MAD_norm_weighted/"+str(args.variable)+"/"+str(args.variable)+"_"+str(args.distance)+"_distances.pdf")
