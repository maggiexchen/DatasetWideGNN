import pandas as pd
import uproot
import numpy
import h5py
import json
import random
import os
os.environ["NUMEXPR_MAX_THREADS"] = "16"
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
import mplhep as hep
import logging
logging.getLogger().setLevel(logging.INFO)
import argparse
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, auc
import utils.normalisation as norm 

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

  args = parser.parse_args()
  return args

args = GetParser()

file_path = "/data/atlas/atlasdata3/maggiechen/gnn_project/distances/"
variable = args.variable + "_"
distance = args.distance + "_"
sigsig_file = file_path + variable + distance + "sigsig_sampled_train.h5"
sigbkg_file = file_path + variable + distance + "sigbkg_sampled_train.h5"
bkgbkg_file = file_path + variable + distance + "bkgbkg_sampled_train.h5"
sigsig = h5py.File(sigsig_file,'r')
sigsig_distance = sigsig['sigsig']['distance']
sigsig_wgt = sigsig['sigsig']['weight']
sigbkg = h5py.File(sigbkg_file,'r')
sigbkg_distance = sigbkg['sigbkg']['distance']
sigbkg_wgt = sigbkg['sigbkg']['weight']
bkgbkg = h5py.File(bkgbkg_file,'r')
bkgbkg_distance = bkgbkg['bkgbkg']['distance']
bkgbkg_wgt = bkgbkg['bkgbkg']['weight']

print(min(bkgbkg_distance), max(bkgbkg_distance))

# calculate ROC values for sigsig and bkgbkg
# Between sigsig (0) and bkgbkg (1)
# Need to normalise the distances to be between 0 and 1, e.g. using minmax normalisation

logging.info("Minmax normalising distances and plotting ...")
d_max = max(max(sigbkg_distance), max(bkgbkg_distance))
norm_sigsig = norm.minmax(sigsig_distance, 0, d_max)
norm_sigbkg = norm.minmax(sigbkg_distance, 0, d_max)
norm_bkgbkg = norm.minmax(bkgbkg_distance, 0, d_max)

nBins = 100
fig, ax = plt.subplots()
binning = numpy.linspace(0,max(norm_bkgbkg),nBins)
ax.hist(norm_sigsig, bins=binning, label="sig-sig", weights=sigsig_wgt, alpha=0.5, density=True)
ax.hist(norm_sigbkg, bins=binning, label="sig-bkg", weights=sigbkg_wgt, alpha=0.5, density=True)
ax.hist(norm_bkgbkg, bins=binning, label="bkg-bkg", weights=bkgbkg_wgt, alpha=0.5, density=True)
ax.legend(loc='upper right')
ax.set_xlabel(str(variable) + str(distance) +" distance", loc="right")
ax.set_ylabel("Normalised No. Events", loc="top")
fig.savefig("/data/atlas/atlasdata3/maggiechen/gnn_project/plots/MAD_norm_weighted/minmax_normed/"+str(variable)+str(distance)+"minmax_distances.pdf")

logging.info("Calculating and saving ROC to json ...")
# fpr here is the fraction of sigsig above a certain cut
# tpr here is the fraction of sig(bkg)bkg above a certain cut
# the actual tpr we want is the fraction of sigsig below a certain cut: (1-fpr)
# and the actual fpr is the fraction of sig(bkg)bkg below a certain cut: (1-tpr)

def calc_ROC(sig, bkg, sig_wgt, bkg_wgt):
    y_sig = [0]*len(sig)
    y_bkg = [1]*len(bkg)
    x_combined = numpy.concatenate((sig, bkg))
    y_combined = numpy.concatenate((y_sig, y_bkg))
    wgt_combined = numpy.concatenate((sig_wgt, bkg_wgt))

    fpr, tpr, cut = roc_curve(y_combined, x_combined, sample_weight=wgt_combined)
    auc = roc_auc_score(y_combined, x_combined, sample_weight=wgt_combined)

    true_tpr = 1-fpr
    true_fpr = 1-tpr

    return true_tpr, true_fpr, cut, auc

tpr_ss_bb, fpr_ss_bb, cut_ss_bb, roc_auc_ss_bb = calc_ROC(norm_sigsig, norm_bkgbkg, sigsig_wgt, bkgbkg_wgt)
tpr_ss_sb, fpr_ss_sb, cut_ss_sb, roc_auc_ss_sb = calc_ROC(norm_sigsig, norm_sigbkg, sigsig_wgt, sigbkg_wgt)

# saving roc and auc to json file
roc_dict = {"ss_bb_sig_cut": cut_ss_bb.tolist(), "tpr_ss_bb": tpr_ss_bb.tolist(), "fpr_ss_bb": fpr_ss_bb.tolist(), "ss_sb_sig_cut": cut_ss_sb.tolist(), "tpr_sb_bb": tpr_ss_bb.tolist(), "fpr_sb_bb": fpr_ss_bb.tolist()}
with open("/data/atlas/atlasdata3/maggiechen/gnn_project/plots/MAD_norm_weighted/ROC/"+str(variable)+str(distance)+"roc.json", "w") as outfile:
  json.dump(roc_dict, outfile)

# pick sig-sig efficiencies at 0.7, 0.8, 0.9
def find_threshold(tpr, fpr, eff, cut):
    # the tpr here is in reverse order with the discriminant cut, so it's <=
    tpr_index = numpy.argmax(tpr <= eff)
    return [tpr[tpr_index], fpr[tpr_index]], cut[tpr_index]

sigsig_eff = [0.7, 0.8, 0.9]
eff_labels = ["70%", "80%", "90%"]
ss_bb_roc_cuts = []
ss_bb_thresholds = []
ss_sb_roc_cuts = []
ss_sb_thresholds = []

# finding the tpr, fpr and distance thresholds for each efficiency, then reverse minmax the distance threshold
for eff in sigsig_eff:
    ss_bb_roc_cut, ss_bb_threshold = find_threshold(tpr_ss_bb, fpr_ss_bb, eff, cut_ss_bb)
    ss_bb_roc_cuts.append(ss_bb_roc_cut)
    ss_bb_thresholds.append(norm.reverse_minmax(ss_bb_threshold, 0, d_max))
    ss_sb_roc_cut, ss_sb_threshold = find_threshold(tpr_ss_sb,fpr_ss_sb, eff, cut_ss_sb)
    ss_sb_roc_cuts.append(ss_sb_roc_cut)
    ss_sb_thresholds.append(norm.reverse_minmax(ss_sb_threshold, 0 ,d_max))

# saving linking lengths
length_dict = {"sigsig_eff": sigsig_eff, "ss_bb_length": ss_bb_thresholds, "ss_sb_length": ss_sb_thresholds}
with open("/data/atlas/atlasdata3/maggiechen/gnn_project/linking_lengths/"+str(variable)+str(distance)+"linking_length.json", "w") as lengthfile:
  json.dump(length_dict, lengthfile)

logging.info("Plotting distance with linking lengths selected from ROC ...")
nBins = 100
# plotting sig-sig and bkg-bkg distributions and the linking lengths
fig, ax = plt.subplots()
binning = numpy.linspace(0,18,nBins)
ax.hist(sigsig_distance, bins=binning, label="sig-sig", weights=sigsig_wgt, alpha=0.5, density=True, color="steelblue")
ax.hist(sigbkg_distance, bins=binning, label="sig-bkg", weights=sigbkg_wgt, alpha=0.5, density=True, color="darkorange")
ax.hist(bkgbkg_distance, bins=binning, label="bkg-bkg", weights=bkgbkg_wgt, alpha=0.5, density=True, color="forestgreen")
ax.text(0.04, 0.93, "ATLAS", fontweight="bold", fontstyle="italic", verticalalignment="bottom", size=10, transform=ax.transAxes)
ax.text(0.14, 0.93, "Internal", verticalalignment="bottom", size=10, transform=ax.transAxes)
ax.text(0.04, 0.88, r"$\sqrt{s}=13$ TeV, 5b data", verticalalignment="bottom", size=10, transform=ax.transAxes)
ax.text(0.04, 0.83, r"6b resonant TRSM signals", verticalalignment="bottom", size=10, transform=ax.transAxes)
y_min, y_max = ax.get_ylim()
for i, eff in enumerate(sigsig_eff):
    ax.axvline(x=ss_bb_thresholds[i], ymax=0.75+i*0.02, linestyle="--", color="red")
    ax.text(x=ss_bb_thresholds[i], y=0.75+i*0.02, transform=ax.get_xaxis_text1_transform(0)[0], s=eff_labels[i], ha='center', va='bottom', fontsize=7)
ax.legend(loc='upper right')
ax.set_ylim(y_min, y_max*1.2)
ax.set_xlabel(str(variable) + str(distance) +"distance", loc="right")
ax.set_ylabel("Normalised No. Pairs", loc="top")
fig.savefig("/data/atlas/atlasdata3/maggiechen/gnn_project/plots/MAD_norm_weighted/linking_lengths_ss_bb/"+str(variable)+str(distance)+"ss_bb_linking_lengths.pdf", transparent=True)

# plotting sig-sig and sig-bkg distributions and the linking lengths
fig, ax = plt.subplots()
binning = numpy.linspace(0,max(bkgbkg_distance),nBins)
ax.hist(sigsig_distance, bins=binning, label="sig-sig", weights=sigsig_wgt, alpha=0.5, density=True, color="steelblue")
ax.hist(sigbkg_distance, bins=binning, label="sig-bkg", weights=sigbkg_wgt, alpha=0.5, density=True, color="darkorange")
ax.text(0.04, 0.93, "ATLAS", fontweight="bold", fontstyle="italic", verticalalignment="bottom", size=10, transform=ax.transAxes)
ax.text(0.14, 0.93, "Internal", verticalalignment="bottom", size=10, transform=ax.transAxes)
ax.text(0.04, 0.88, r"$\sqrt{s}=13$ TeV, 5b data", verticalalignment="bottom", size=10, transform=ax.transAxes)
ax.text(0.04, 0.83, r"6b resonant TRSM signals", verticalalignment="bottom", size=10, transform=ax.transAxes)
y_min, y_max = ax.get_ylim()
for i, eff in enumerate(sigsig_eff):
    ax.axvline(x=ss_bb_thresholds[i], ymax=0.75+i*0.02, linestyle="--", color="red")
    ax.text(x=ss_bb_thresholds[i], y=0.75+i*0.02, transform=ax.get_xaxis_text1_transform(0)[0], s=eff_labels[i], ha='center', va='bottom', fontsize=9)
ax.legend(loc='upper right')
ax.set_ylim(y_min, y_max*1.2)
ax.set_xlabel(str(variable) + str(distance) +"distance", loc="right")
ax.set_ylabel("Normalised No. Pairs", loc="top")
fig.savefig("/data/atlas/atlasdata3/maggiechen/gnn_project/plots/MAD_norm_weighted/linking_lengths_ss_sb/"+str(variable)+str(distance)+"ss_sb_linking_lengths.pdf", transparent=True)


logging.info("Plotting ROC curves ...")
fig, ax = plt.subplots()
plt.style.use(hep.style.ROOT)
plt.plot(fpr_ss_bb, tpr_ss_bb, label='sig-sig bkg-bkg ROC curve (AUC = {:.3f})'.format(roc_auc_ss_bb))
plt.plot(fpr_ss_sb, tpr_ss_sb, label='sig-sig sig-bkg ROC curve (AUC = {:.3f})'.format(roc_auc_ss_sb))
plt.scatter(numpy.array(ss_bb_roc_cuts)[:,1], numpy.array(ss_bb_roc_cuts)[:,0], marker='x', s=50, label="linking lengths",color="red")
plt.scatter(numpy.array(ss_sb_roc_cuts)[:,1], numpy.array(ss_bb_roc_cuts)[:,0], marker='x', s=50,color="red")
plt.legend()
ymin, ymax = plt.ylim()
plt.ylim(ymin, ymax*1.2)
plt.xlim(0,1)
plt.xlabel("Sig(bkg)-bkg Efficiency")
plt.ylabel("Sig-sig Efficiency")
plot_name = "/data/atlas/atlasdata3/maggiechen/gnn_project/plots/MAD_norm_weighted/ROC/"+str(variable)+str(distance)+"ROC.pdf"
fig.savefig(plot_name, transparent=True)