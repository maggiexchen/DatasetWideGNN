import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
import utils.misc as misc

def plot_distances(ss, sb, bb, ss_wgt, sb_wgt, bb_wgt, var, distance, path, label=""):
  '''
  Todo add docstring
  '''
  # bin/range
  x_max = max(max(bb),max(max(ss),max(sb)))
  n_bins=100
  binning=np.linspace(0,x_max,n_bins)
  # plot
  fig, ax = plt.subplots()
  ax.hist(ss, bins=binning, label="sig-sig", alpha=0.5, weights=ss_wgt, density=True)
  ax.hist(sb, bins=binning, label="sig-bkg", alpha=0.5, weights=sb_wgt, density=True)
  ax.hist(bb, bins=binning, label="bkg-bkg", alpha=0.5, weights=bb_wgt, density=True)
  # aesthesics
  ax.legend(loc='upper right')
  ax.set_xlabel(var+"_"+distance +" distance", loc="right")
  ax.set_ylabel("Normalised # event pairs / bin", loc="top")
  # save
  if label!="": label = "_"+label
  exts = [".pdf", ".eps", ".png"]
  for ext in exts:
    fig.savefig(path+"/"+var+"_"+distance+label+"_distances"+ext)
  return 0

def plot_kinematic_hists(df_sig, df_bkg, var, file_path):
  '''
  Todo add dosctring
  '''
  # plot
  fig, ax = plt.subplots()
  binning = np.linspace(min(df_bkg.loc[:, var]),max(df_bkg.loc[:, var]), 50)
  ys, xs, _ = ax.hist(df_sig.loc[:, var], bins=binning, label="MAD-normed sig (6b TRSM)", alpha=0.3, density=True, color="steelblue")
  yb, xb, _ = ax.hist(df_bkg.loc[:, var], bins=binning, label="MAD-normed bkg (5b data)", alpha=0.3, density=True, color="red")
  # aesthetics
  hep.atlas.label(ax=ax, data=False, label="Internal", lumi="129")
  ax.legend(loc='upper right')
  ax.set_ylim([0.01, 1.2*max(max(ys),max(yb))])
  ax.set_xlabel(str(var), loc="right")
  ax.set_ylabel("Normalised No. Events", loc="top")
  # save
  save_path = file_path+"/training_kinematics/"
  misc.create_dirs(save_path)
  exts = [".pdf", ".eps", ".png"]
  for ext in exts:
    fig.savefig(save_path+"/"+var+ext, transparent=True)
  return 0

