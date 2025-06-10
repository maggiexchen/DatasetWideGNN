"""
Function to calculate linking lengths for a bunch of sig-sig efficiencies
"""
import logging
import argparse
import math
import json

import utils.normalisation as norm
import utils.misc as misc
import utils.plotting as plotting
import utils.performance as perf
import utils.graph_definition as graph_def
import utils.user_config as uconfig
import utils.ml_config as mlconfig


import numpy as np
import torch
from torcheval.metrics.functional import mean as mean_wgted

logging.getLogger().setLevel(logging.INFO)
torch.manual_seed(42)

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
    help="Specify the config for the user",
)

parser.add_argument(
    "--batchsize",
    "-b",
    type=int,
    default=10000,
    required=False,
    help="",
)

args = parser.parse_args()

user_config_path = args.userconfig
user = uconfig.UserConfig.from_yaml(user_config_path)

ml_config_path = args.MLconfig
ml = mlconfig.MLConfig.from_yaml(ml_config_path)

variable = str(args.variable)
distance = str(args.distance)
batch_size = args.batchsize

signal_label, background_label = plotting.get_plot_labels(user.signal)

if ml.targettarget_eff is not None and ml.edge_frac is not None:
    raise ValueError("edge_frac and targettarget_eff in ML config, pick just one!")
if ml.targettarget_eff is None and ml.edge_frac is None:
    raise ValueError("Neither edge_frac or sigsig_eff in ML config, pick one!")
do_edge_frac = True if ml.edge_frac is not None else False

do_same_class = True if variable == "embedding" else False
do_friend_graph = ml.friend_graph

logging.info("variable set: %s", variable)
logging.info("distance metric: %s", distance)
logging.info("variable set: %s", variable)
logging.info("making a friend graph? %s", str(ml.friend_graph))


logging.info("Loading sigbkg distances in batches")
sigbkg_distance, sigbkg_wgt, sigbkg_max = misc.get_batched_distances(user.dist_path,
                                                                     variable, distance,
                                                                     batch_size, "sigbkg",
                                                                     sample=True,
                                                                     cutstring=user.cutstring)
sigbkg_mean = torch.mean(sigbkg_distance)

logging.info("Loading bkgbkg distances in batches")
bkgbkg_distance, bkgbkg_wgt, bkgbkg_max = misc.get_batched_distances(user.dist_path,
                                                                     variable, distance,
                                                                     batch_size, "bkgbkg",
                                                                     sample=True,
                                                                     cutstring=user.cutstring)
bkgbkg_mean = mean_wgted(bkgbkg_distance, bkgbkg_wgt)

logging.info("Minmax normalising sigbkg/bkgbkg distances")
d_max = max(sigbkg_max, bkgbkg_max)
norm_sigbkg = norm.minmax(sigbkg_distance, 0, d_max)
norm_bkgbkg = norm.minmax(bkgbkg_distance, 0, d_max)

# initialise histograms
n_bins = 101
x_max = math.ceil(d_max)
binning = np.linspace(0., x_max, n_bins)
norm_binning = np.linspace(0., 1., n_bins)
sigbkg_hist = np.histogram(sigbkg_distance, bins=binning, range=(0, x_max),
                           weights=sigbkg_wgt, density=False)
bkgbkg_hist = np.histogram(bkgbkg_distance, bins=binning, range=(0, x_max),
                           weights=bkgbkg_wgt, density=False)
norm_sigbkg_hist = np.histogram(norm_sigbkg, bins=norm_binning, range=(0, 1),
                                weights=sigbkg_wgt, density=False)
norm_bkgbkg_hist = np.histogram(norm_bkgbkg, bins=norm_binning, range=(0, 1),
                                weights=bkgbkg_wgt, density=False)


logging.info("Loading sigsig distances in batches")
sigsig_distance, sigsig_wgt, sigsig_max = misc.get_batched_distances(user.dist_path,
                                                                     variable, distance,
                                                                     batch_size, "sigsig",
                                                                     sample=True,
                                                                     cutstring=user.cutstring)
sigsig_mean = mean_wgted(sigsig_distance, sigsig_wgt)

logging.info("Minmax normalising sigsig distances")
norm_sigsig = norm.minmax(sigsig_distance, 0, d_max)
sigsig_hist = np.histogram(sigsig_distance, bins=binning, range=(0, x_max),
                           weights=sigsig_wgt, density=False)
norm_sigsig_hist = np.histogram(norm_sigsig, bins=norm_binning, range=(0, 1),
                                weights=sigsig_wgt, density=False)


logging.info("Plotting ...")
plot_path = user.plot_path + "/" + variable + "/"
misc.create_dirs(plot_path)

plotting.plot_distances_hist(sigsig_hist, sigbkg_hist, bkgbkg_hist,
                             variable, distance, signal_label, background_label,
                             plot_path, standardised=False)

logging.info("Calculating and saving ROC to json ...")

plotting.plot_distances_hist(norm_sigsig_hist, norm_sigbkg_hist, norm_bkgbkg_hist,
                             variable, distance, signal_label, background_label,
                             plot_path, standardised=True)

############################
# Now work out if we have a case where sig-sig or bkg-bkg are friendlier
is_signal_closest = True
friend_species = "signal"
if sigsig_mean > bkgbkg_mean:
    is_signal_closest = False
    friend_species = "background"
if not do_same_class:
    logging.info("<sig-sig> = %s, <bkg-bkg> = %s",str(sigsig_mean), str(bkgbkg_mean))
    if not do_edge_frac:
        logging.info("The %s events are closest, measuring for their self-connection eff.",
                     friend_species)
else:
    if not do_edge_frac:
        logging.info("Doing same-class optimisation, measuring for sig-sig self-connection eff.")
if do_edge_frac:
    logging.info("Doing edge_frac optimisation, don't really care about relative distance sizes.")


# calculate ROC values for sigsig and bkgbkg Between sigsig (0) and bkgbkg (1)
# Need to normalise the distances to be between 0 and 1, e.g. using minmax normalisation
# This normalisation shouldn't change the shape and the relative scale
#    of the distance distributions (can be checked in plots)


# IN THE CASE WHERE MOST SIGSIG DISTANCES ARE SMALLER THAN BKGBKG DISTANCES
#   (E.G. TRSM HHH SIGNALS)
#   fpr here is the fraction of sigsig above a certain cut
#   tpr here is the fraction of sig(bkg)bkg above a certain cut
#   the actual tpr we want is the fraction of sigsig below a certain cut: (1-fpr)
#   and the actual fpr is the fraction of sig(bkg)bkg below a certain cut: (1-tpr)


tpr_ss_sb, fpr_ss_sb, cut_ss_sb, roc_auc_ss_sb = \
    perf.calc_roc(norm_sigsig, norm_sigbkg, sigsig_wgt, sigbkg_wgt,
                  is_target_closest=is_signal_closest)

tpr_bb_sb, fpr_bb_sb, cut_bb_sb, roc_auc_bb_sb = \
    perf.calc_roc(norm_bkgbkg, norm_sigbkg, bkgbkg_wgt, sigbkg_wgt,
                  is_target_closest=(not is_signal_closest))
del sigbkg_wgt
if not do_edge_frac:
    tpr_ss_bb, fpr_ss_bb, cut_ss_bb, roc_auc_ss_bb = \
        perf.calc_roc(norm_sigsig, norm_bkgbkg, sigsig_wgt, bkgbkg_wgt,
                      is_target_closest=is_signal_closest)
    tpr_bb_ss, fpr_bb_ss, cut_bb_ss, roc_auc_bb_ss = \
        perf.calc_roc(norm_bkgbkg, norm_sigsig, bkgbkg_wgt, sigsig_wgt,
                      is_target_closest=(not is_signal_closest))

del sigsig_wgt, bkgbkg_wgt
# saving roc and auc to json file
roc_dict = {
            "ss_sb_sig_cut": cut_ss_sb.tolist(),
            "tpr_ss_sb": tpr_ss_sb.tolist(),
            "fpr_ss_sb": fpr_ss_sb.tolist(),
            "bb_sb_bkg_cut": cut_bb_sb.tolist(),
            "tpr_bb_sb": tpr_bb_sb.tolist(),
            "fpr_bb_sb": fpr_bb_sb.tolist()}

if not do_edge_frac:
    roc_dict["bb_ss_bkg_cut"] = cut_bb_ss.tolist()
    roc_dict["tpr_bb_ss"] = tpr_bb_ss.tolist()
    roc_dict["fpr_bb_ss"] = fpr_bb_ss.tolist()
    roc_dict["ss_bb_bkg_cut"] = cut_ss_bb.tolist()
    roc_dict["tpr_ss_bb"] = tpr_ss_bb.tolist()
    roc_dict["fpr_ss_bb"] = fpr_ss_bb.tolist()

roc_path = plot_path + "/" + variable + "/ROC/"
misc.create_dirs(roc_path)
roc_name = roc_path + variable + "_" + distance + user.cutstring + "_ROC_" +\
           friend_species + "friends.json"
logging.info("saving ROC curve to %s", roc_name)
with open(roc_name, "w", encoding="utf-8") as outfile:
    json.dump(roc_dict, outfile)
del roc_dict

if do_edge_frac:

    logging.info("doing edge_frac threshold calculations")
    # doing edge-frac based calculations.

    del norm_sigsig, norm_sigbkg, norm_bkgbkg

    edge_frac = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
    linking_lengths = graph_def.find_threshold_edge_frac(sigsig_distance, sigbkg_distance,
                                                         bkgbkg_distance,
                                                         edge_frac, x_max,
                                                         do_friend_graph=do_friend_graph)


    length_dict = {"edge_frac": edge_frac, "length": linking_lengths}
    misc.create_dirs(user.ll_path)
    ll_path = user.ll_path + "edge_frac_" + variable + "_" +\
            distance + user.cutstring + "_linking_length.json"

    # plotting sig-sig and bkg-bkg distributions and the linking lengths
    logging.info("Plotting linking lengths ...")

    plotting.plot_linking_length_hist(sigsig_hist, sigbkg_hist, bkgbkg_hist, linking_lengths,
                                      signal_label, background_label, do_edge_frac, plot_path,
                                      variable, distance, edge_frac)

    logging.info("Plotting ROC curves ...")
    minmax_ll = norm.minmax(torch.tensor(linking_lengths), 0, d_max)
    plotting.plot_roc_edge_frac(fpr_ss_sb, tpr_ss_sb, fpr_bb_sb, tpr_bb_sb,
                                roc_auc_ss_sb, roc_auc_bb_sb, cut_ss_sb,
                                cut_bb_sb, minmax_ll.numpy(), variable, distance, plot_path)

else:

    logging.info("doing targettarget_eff threshold calculations")

    del sigbkg_distance, bkgbkg_distance, sigsig_distance

    # TODO: finer granularity for linking length scan?
    targettarget_eff = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    targettarget_eff_label = "sigsig_eff"

    ss_sb_roc_cuts = []
    ss_sb_thresholds = []

    ss_bb_roc_cuts = []
    ss_bb_thresholds = []

    bb_sb_roc_cuts = []
    bb_sb_thresholds = []

    bb_ss_roc_cuts = []
    bb_ss_thresholds = []

    targettarget_thresholds = []

    # finding the tpr, fpr and distance thresholds for each efficiency,
    #   then reverse minmax the distance threshold
    # note the ss_bb and ss_sb thresholds are the same as they are%s",
    #   just determined as the threshold for the given ss efficiency.
    for eff in targettarget_eff:
        if do_same_class:
            ss_sb_roc_cut, ss_sb_threshold = \
                graph_def.find_threshold(tpr_ss_sb, fpr_ss_sb, eff, cut_ss_sb,
                                         is_target_closest=is_signal_closest)
            ss_sb_roc_cuts.append(ss_sb_roc_cut)
            ss_sb_thresholds.append(norm.reverse_minmax(ss_sb_threshold, 0, d_max))
            bb_sb_roc_cut, bb_sb_threshold = \
                graph_def.find_threshold(tpr_bb_sb, fpr_bb_sb, eff, cut_bb_sb,
                                         is_target_closest=(not is_signal_closest))
            bb_sb_roc_cuts.append(bb_sb_roc_cut)
            targettarget_thresholds.append(norm.reverse_minmax(bb_sb_threshold, 0, d_max).item())
        else:
            if is_signal_closest:
                ss_sb_roc_cut, ss_sb_threshold = \
                    graph_def.find_threshold(tpr_ss_sb, fpr_ss_sb, eff, cut_ss_sb,
                                             is_target_closest=is_signal_closest)
                ss_sb_roc_cuts.append(ss_sb_roc_cut)
                ss_sb_thresholds.append(norm.reverse_minmax(ss_sb_threshold, 0, d_max))
                ss_bb_roc_cut, ss_bb_threshold = \
                    graph_def.find_threshold(tpr_ss_bb, fpr_ss_bb, eff, cut_ss_bb,
                                             is_target_closest=is_signal_closest)
                ss_bb_roc_cuts.append(ss_bb_roc_cut)
                targettarget_thresholds.append(norm.reverse_minmax(ss_bb_threshold,
                                                                   0, d_max).item())
            else:
                targettarget_eff_label = "bkgbkg_eff"
                bb_sb_roc_cut, bb_sb_threshold = \
                    graph_def.find_threshold(tpr_bb_sb, fpr_bb_sb, eff, cut_bb_sb,
                                             is_target_closest=(not is_signal_closest))
                bb_sb_roc_cuts.append(bb_sb_roc_cut)
                bb_sb_thresholds.append(norm.reverse_minmax(bb_sb_threshold, 0, d_max))
                bb_ss_roc_cut, bb_ss_threshold = \
                    graph_def.find_threshold(tpr_bb_ss, fpr_bb_ss, eff, cut_bb_ss,
                                             is_target_closest=(not is_signal_closest))
                bb_ss_roc_cuts.append(bb_ss_roc_cut)
                targettarget_thresholds.append(norm.reverse_minmax(bb_ss_threshold,
                                                                   0, d_max).item())

    # saving linking lengths
    length_dict = {"targettarget_eff": targettarget_eff, "length": targettarget_thresholds}
    misc.create_dirs(user.ll_path)
    ll_path = user.ll_path + "targettarget_eff_" + variable + "_" +\
        distance + user.cutstring + "_linking_length.json"

    # plotting sig-sig and bkg-bkg distributions and the linking lengths
    # TODO: moving plotting to utils
    logging.info("Plotting linking lengths ...")
    plotting.plot_linking_length_hist(sigsig_hist, sigbkg_hist, bkgbkg_hist,
                                      targettarget_thresholds, signal_label, background_label,
                                      do_edge_frac, plot_path, variable, distance, targettarget_eff,
                                      target_eff_label=targettarget_eff_label)

    logging.info("Plotting ROC curves ...")
    if do_same_class:
        plotting.plot_roc(fpr_ss_sb, tpr_ss_sb,
                          fpr_bb_sb, tpr_bb_sb,
                          roc_auc_ss_sb, roc_auc_bb_sb,
                          ss_sb_roc_cuts, bb_sb_roc_cuts,
                          variable, distance, plot_path)
    else:
        if is_signal_closest:
            plotting.plot_roc(fpr_ss_bb, tpr_ss_bb,
                              fpr_ss_sb, tpr_ss_sb,
                              roc_auc_ss_bb, roc_auc_ss_sb,
                              ss_bb_roc_cuts, ss_sb_roc_cuts,
                              variable, distance, plot_path)
        else:
            plotting.plot_roc(fpr_bb_ss, tpr_bb_ss,
                              fpr_bb_sb, tpr_bb_sb,
                              roc_auc_bb_sb, roc_auc_bb_sb,
                              bb_ss_roc_cuts, bb_sb_roc_cuts,
                              variable, distance, plot_path)


logging.info("saving ll json to: %s", ll_path)
with open(ll_path, "w", encoding="utf-8") as lengthfile:
    json.dump(length_dict, lengthfile)
