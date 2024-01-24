import pandas as pd
import numpy as np
import h5py
import json
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
      "--path",
      "-p",
      type=str,
      required=False,
      help="Specify the path to store all the input/output data and results",
  )


  args = parser.parse_args()
  return args

args = GetParser()

variable = str(args.variable)
distance = str(args.distance)

path = "/data/atlas/atlasdata3/maggiechen/gnn_project/"
if args.path:
    path = args.path
    if path[-1]!="/": path += "/"

logging.info("variable set: "+variable)
logging.info("distance metric: "+distance)
logging.info("input/output path: "+path)

file_path = path+"distances/"
sigsig_file, sigbkg_file, bkgbkg_file = misc.get_h5_paths(file_path, variable, distance)
sigsig = h5py.File(sigsig_file,'r')
sigsig_distance = sigsig['sigsig']['distance']
sigsig_wgt = sigsig['sigsig']['weight']
sigbkg = h5py.File(sigbkg_file,'r')
sigbkg_distance = sigbkg['sigbkg']['distance']
sigbkg_wgt = sigbkg['sigbkg']['weight']
bkgbkg = h5py.File(bkgbkg_file,'r')
bkgbkg_distance = bkgbkg['bkgbkg']['distance']
bkgbkg_wgt = bkgbkg['bkgbkg']['weight']

logging.info("Min bgbg distance = "+str(min(bkgbkg_distance))+", Max bgbg distance = "+str(max(bkgbkg_distance)))

# calculate ROC values for sigsig and bkgbkg
# Between sigsig (0) and bkgbkg (1)
# Need to normalise the distances to be between 0 and 1, e.g. using minmax normalisation

logging.info("Minmax normalising distances and plotting ...")
d_max = max(max(sigbkg_distance), max(bkgbkg_distance))
norm_sigsig = norm.minmax(sigsig_distance, 0, d_max)
norm_sigbkg = norm.minmax(sigbkg_distance, 0, d_max)
norm_bkgbkg = norm.minmax(bkgbkg_distance, 0, d_max)

plot_path = path+"plots/MAD_norm_weighted/"+variable+"/"
misc.create_dirs(plot_path)
plotting.plot_distances(norm_sigsig, norm_sigbkg, norm_bkgbkg, sigsig_wgt, sigbkg_wgt, bkgbkg_wgt, variable, distance, plot_path, label="minmaxNormed")

logging.info("Calculating and saving ROC to json ...")
# fpr here is the fraction of sigsig above a certain cut
# tpr here is the fraction of sig(bkg)bkg above a certain cut
# the actual tpr we want is the fraction of sigsig below a certain cut: (1-fpr)
# and the actual fpr is the fraction of sig(bkg)bkg below a certain cut: (1-tpr)

def calc_ROC(sig, bkg, sig_wgt, bkg_wgt):
    y_sig = [0]*len(sig)
    y_bkg = [1]*len(bkg)
    x_combined = np.concatenate((sig, bkg))
    y_combined = np.concatenate((y_sig, y_bkg))
    wgt_combined = np.concatenate((sig_wgt, bkg_wgt))

    fpr, tpr, cut = roc_curve(y_combined, x_combined, sample_weight=wgt_combined)
    auc = roc_auc_score(y_combined, x_combined, sample_weight=wgt_combined)

    true_tpr = 1-fpr
    true_fpr = 1-tpr

    return true_tpr, true_fpr, cut, auc

tpr_ss_bb, fpr_ss_bb, cut_ss_bb, roc_auc_ss_bb = calc_ROC(norm_sigsig, norm_bkgbkg, sigsig_wgt, bkgbkg_wgt)
tpr_ss_sb, fpr_ss_sb, cut_ss_sb, roc_auc_ss_sb = calc_ROC(norm_sigsig, norm_sigbkg, sigsig_wgt, sigbkg_wgt)

# saving roc and auc to json file
roc_dict = {"ss_bb_sig_cut": cut_ss_bb.tolist(),
            "tpr_ss_bb": tpr_ss_bb.tolist(),
            "fpr_ss_bb": fpr_ss_bb.tolist(),
            "ss_sb_sig_cut": cut_ss_sb.tolist(),
            "tpr_sb_bb": tpr_ss_bb.tolist(),
            "fpr_sb_bb": fpr_ss_bb.tolist()}
roc_path = path+"plots/MAD_norm_weighted/ROC/"+variable+"_"+distance+"_ROC.json"
misc.create_dirs(roc_path)
with open(roc_path, "w") as outfile:
  json.dump(roc_dict, outfile)

# pick sig-sig efficiencies at 0.7, 0.8, 0.9
def find_threshold(tpr, fpr, eff, cut):
    # the tpr here is in reverse order with the discriminant cut, so it's <=
    tpr_index = np.argmax(tpr <= eff)
    return [tpr[tpr_index], fpr[tpr_index]], cut[tpr_index]

sigsig_eff = [0.6, 0.7, 0.8, 0.9]
eff_labels = ["60%", "70%", "80%", "90%"]
ss_sb_roc_cuts = []
ss_sb_thresholds = []
ss_bb_roc_cuts = []
ss_thresholds = []

# finding the tpr, fpr and distance thresholds for each efficiency, then reverse minmax the distance threshold
# note the ss_bb and ss_sb thresholds are the same as they are just determined as the threshold for the given ss efficiency.
for eff in sigsig_eff:
    ss_sb_roc_cut, ss_sb_threshold = find_threshold(tpr_ss_sb,fpr_ss_sb, eff, cut_ss_sb)
    ss_sb_roc_cuts.append(ss_sb_roc_cut)
    ss_sb_thresholds.append(norm.reverse_minmax(ss_sb_threshold, 0 ,d_max))
    ss_bb_roc_cut, ss_bb_threshold = find_threshold(tpr_ss_bb,fpr_ss_bb, eff, cut_ss_bb)
    ss_bb_roc_cuts.append(ss_bb_roc_cut)
    ss_thresholds.append(norm.reverse_minmax(ss_bb_threshold, 0 ,d_max))

# saving linking lengths
length_dict = {"sigsig_eff": sigsig_eff, "length": ss_thresholds}
ll_path = path+"linking_lengths/"+variable+"_"+distance+"_linking_length.json"
misc.create_dirs(ll_path)
with open(ll_path, "w") as lengthfile:
  json.dump(length_dict, lengthfile)

logging.info("Plotting distance with linking lengths selected from ROC ...")
nBins = 100

# plotting sig-sig and bkg-bkg distributions and the linking lengths
fig, ax = plt.subplots()
binning = np.linspace(0,18,nBins)
ax.hist(sigsig_distance, bins=binning, label="sig-sig", weights=sigsig_wgt, alpha=0.5, density=True, color="steelblue")
ax.hist(sigbkg_distance, bins=binning, label="sig-bkg", weights=sigbkg_wgt, alpha=0.5, density=True, color="darkorange")
ax.hist(bkgbkg_distance, bins=binning, label="bkg-bkg", weights=bkgbkg_wgt, alpha=0.5, density=True, color="forestgreen")
ax.text(0.04, 0.93, "ATLAS", fontweight="bold", fontstyle="italic", verticalalignment="bottom", size=10, transform=ax.transAxes)
ax.text(0.14, 0.93, "Internal", verticalalignment="bottom", size=10, transform=ax.transAxes)
ax.text(0.04, 0.88, r"$\sqrt{s}=13$ TeV, 5b data", verticalalignment="bottom", size=10, transform=ax.transAxes)
ax.text(0.04, 0.83, r"6b resonant TRSM signals", verticalalignment="bottom", size=10, transform=ax.transAxes)
y_min, y_max = ax.get_ylim()
for i, eff in enumerate(sigsig_eff):
    ax.axvline(x=ss_thresholds[i], ymax=0.7+i*0.02, linestyle="--", color="red")
    ax.text(x=ss_thresholds[i], y=0.75+i*0.02, transform=ax.get_xaxis_text1_transform(0)[0], s=eff_labels[i], ha='center', va='bottom', fontsize=7)
ax.legend(loc='upper right')
ax.set_ylim(y_min, y_max*1.2)
ax.set_xlabel(variable + distance +"distance", loc="right")
ax.set_ylabel("Normalised # event pairs / bin", loc="top")
ssbb_path = path+"plots/MAD_norm_weighted/linking_lengths/"
misc.create_dirs(ssbb_path)
fig.savefig(ssbb_path+"/"+variable+"_"+distance+"_linking_lengths.pdf", transparent=True)

logging.info("Plotting ROC curves ...")
fig, ax = plt.subplots()
plt.style.use(hep.style.ROOT)
plt.plot(fpr_ss_bb, tpr_ss_bb, label='sig-sig bkg-bkg ROC curve (AUC = {:.3f})'.format(roc_auc_ss_bb))
plt.plot(fpr_ss_sb, tpr_ss_sb, label='sig-sig sig-bkg ROC curve (AUC = {:.3f})'.format(roc_auc_ss_sb))
plt.scatter(np.array(ss_bb_roc_cuts)[:,1], np.array(ss_bb_roc_cuts)[:,0], marker='x', s=50, label="linking lengths",color="red")
plt.scatter(np.array(ss_sb_roc_cuts)[:,1], np.array(ss_sb_roc_cuts)[:,0], marker='x', s=50, color="red")
plt.legend(loc="lower right", fontsize="11")
ymin, ymax = plt.ylim()
plt.ylim(0.,1.)
plt.xlim(0.,1.)
plt.xlabel("sig(bkg)-bkg Efficiency")
plt.ylabel("sig-sig Efficiency")
plot_name = path+"plots/MAD_norm_weighted/ROC/"
misc.create_dirs(plot_name)
fig.savefig(plot_name+"/"+variable+"_"+distance+"_ROC.pdf", transparent=True)
