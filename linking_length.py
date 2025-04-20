import pandas as pd
import numpy as np
import h5py
import json
import torch
import os
os.environ["NUMEXPR_MAX_THREADS"] = "16"
import matplotlib.pyplot as plt
import mplhep as hep
import logging
logging.getLogger().setLevel(logging.INFO)
import argparse
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, auc
import utils.normalisation as norm
import utils.misc as misc
import utils.plotting as plotting
import utils.performance as perf
import utils.graph_definition as graph_def
torch.manual_seed(42)

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
        "--MLconfig",
        "-c",
        type=str,
        required=True,
        help="Specify the config file for training",
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

variable = str(args.variable)
distance = str(args.distance)

user_config_path = args.userconfig
user_config = misc.load_config(user_config_path)
feature_h5_path = user_config["feature_h5_path"]
plot_path = user_config["plot_path"]
dist_path = user_config["dist_path"]
ll_path = user_config["ll_path"]

signal = user_config["signal"]
assert signal in ["hhh", "LQ", "stau"], f"Invalid signal type: {signal}"
signal_label, background_label = plotting.get_plot_labels(signal)

train_config_path = args.MLconfig
train_config = misc.load_config(train_config_path)
flip = train_config["flip"]

logging.info("variable set: "+variable)
logging.info("distance metric: "+distance)
logging.info("signal: "+signal)
logging.info("variable set: "+variable)
logging.info("input data path: "+feature_h5_path)
logging.info("input distances path: "+dist_path)
logging.info("output ll json path: "+ll_path)
logging.info("output plot path: "+plot_path)
logging.info("making a friend graph? " + str(flip))

logging.info("Loading sigsig distances in batches")
sigsig_distance, sigsig_wgt = misc.get_batched_distances(dist_path, variable, distance, "sigsig", sample=True)
logging.info("Loading sigbkg distances in batches")
sigbkg_distance, sigbkg_wgt = misc.get_batched_distances(dist_path, variable, distance, "sigbkg", sample=True)
logging.info("Loading bkgbkg distances in batches")
bkgbkg_distance, bkgbkg_wgt = misc.get_batched_distances(dist_path, variable, distance, "bkgbkg", sample=True)

# calculate ROC values for sigsig and bkgbkg
# Between sigsig (0) and bkgbkg (1)
# Need to normalise the distances to be between 0 and 1, e.g. using minmax normalisation
# This normalisation shouldn't change the shape and the relative scale of the distance distributions (can be checked in plots)

logging.info("Minmax normalising distances and plotting ...")
d_max = max(torch.max(sigbkg_distance), torch.max(bkgbkg_distance))
norm_sigsig = norm.minmax(sigsig_distance, 0, d_max)
norm_sigbkg = norm.minmax(sigbkg_distance, 0, d_max)
norm_bkgbkg = norm.minmax(bkgbkg_distance, 0, d_max)

plot_path = plot_path+"/"+variable+"/"
misc.create_dirs(plot_path)
plotting.plot_distances(norm_sigsig, norm_sigbkg, norm_bkgbkg, sigsig_wgt, sigbkg_wgt, bkgbkg_wgt, variable, distance, plot_path, label="minmaxNormed")

logging.info("Calculating and saving ROC to json ...")
# IN THE CASE WHERE MOST SIGSIG DISTANCES ARE SMALLER THAN BKGBKG DISTANCES (E.G. TRSM HHH SIGNALS)
# fpr here is the fraction of sigsig above a certain cut
# tpr here is the fraction of sig(bkg)bkg above a certain cut
# the actual tpr we want is the fraction of sigsig below a certain cut: (1-fpr)
# and the actual fpr is the fraction of sig(bkg)bkg below a certain cut: (1-tpr)

tpr_ss_bb, fpr_ss_bb, cut_ss_bb, roc_auc_ss_bb = perf.calc_ROC(norm_sigsig, norm_bkgbkg, sigsig_wgt, bkgbkg_wgt, flip=flip)
tpr_ss_sb, fpr_ss_sb, cut_ss_sb, roc_auc_ss_sb = perf.calc_ROC(norm_sigsig, norm_sigbkg, sigsig_wgt, sigbkg_wgt, flip=flip)

# saving roc and auc to json file
roc_dict = {"ss_bb_sig_cut": cut_ss_bb.tolist(),
            "tpr_ss_bb": tpr_ss_bb.tolist(),
            "fpr_ss_bb": fpr_ss_bb.tolist(),
            "ss_sb_sig_cut": cut_ss_sb.tolist(),
            "tpr_sb_bb": tpr_ss_bb.tolist(),
            "fpr_sb_bb": fpr_ss_bb.tolist()}
roc_path = plot_path+"/"+variable+"/ROC/"
misc.create_dirs(roc_path)
roc_name = roc_path+variable+"_"+distance+"_ROC.json"
with open(roc_name, "w") as outfile:
    json.dump(roc_dict, outfile)

# TODO: finer granularity for linking length scan?
# sigsig_eff = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
# eff_labels = ["10%", "20%", "30%", "40%", "50%", "60%", "70%", "80%", "90%"]
edge_frac = [0.1, 0.2, 0.3, 0.4, 0.5]
edge_frac_label = ["10%", "20%", "30%", "40%", "50%"]

ss_thresholds = graph_def.find_threshold_edge_frac(sigsig_distance, sigbkg_distance, bkgbkg_distance, edge_frac, flip=flip)

# saving linking lengths
length_dict = {"edge_frac": edge_frac, "length": ss_thresholds}
misc.create_dirs(ll_path)
ll_path = ll_path+""+variable+"_"+distance+"_linking_length.json"
with open(ll_path, "w") as lengthfile:
    json.dump(length_dict, lengthfile)

# plotting sig-sig and bkg-bkg distributions and the linking lengths
# TODO: moving plotting to utils
logging.info("Plotting linking lengths ...")
# plotting.plot_linking_length(sigsig_distance, sigbkg_distance, bkgbkg_distance, sigsig_wgt, sigbkg_wgt, bkgbkg_wgt, ss_thresholds, signal_label, background_label, plot_path, variable, distance, sigsig_eff)
plotting.plot_linking_length(sigsig_distance, sigbkg_distance, bkgbkg_distance, sigsig_wgt, sigbkg_wgt, bkgbkg_wgt, ss_thresholds, signal_label, background_label, plot_path, variable, distance, edge_frac)

# logging.info("Plotting ROC curves ...")
# plotting.plot_ROC(fpr_ss_bb, tpr_ss_bb, fpr_ss_sb, tpr_ss_sb, roc_auc_ss_bb, roc_auc_ss_sb, ss_bb_roc_cuts, ss_sb_roc_cuts, variable, distance, plot_path)
