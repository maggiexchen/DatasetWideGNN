import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
import torch
import utils.normalisation as norm
import utils.misc as misc
import numpy

def plot_distances(ss, sb, bb, ss_wgt, sb_wgt, bb_wgt, var, distance, path, label=""):
    """
    Function to plot distances for sig-sig, sig-bkg and bkg-bkg on one figure.

    Args:
        ss (numpy.ndarray): array of the distances for each pair of sig-sig events
        sb (numpy.ndarray): array of the distances for each pair of sig-bkg events
        bb (numpy.ndarray): array of the distances for each pair of bkg-bkg events
        ss_wgt (numpy.ndarray): array of the product of eventWeights for each pair of sig-sig events
        sb_wgt (numpy.ndarray): array of the product of eventWeights for each pair of sig-bkg events
        bb_wgt (numpy.ndarray): array of the product of eventWeights for each pair of bkg-bkg events
        var (str): Kinematic variable
        distance (str): Distance metric (euclidean, cosine, cityblock)
        path (str): base dir to store output
        label (str): extra str for filename

    Returns:
            void
    """
    print(type(ss))
    print(type(ss_wgt))
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
    exts = [".pdf"]
    for ext in exts:
        fig.savefig(path+"/"+var+"_"+distance+label+"_distances"+ext)

    return 0

def plot_kinematic_hists(df_sig, df_bkg, var, file_path):
    """
    Function to plot the histogram of a kinematic variable for signal and background on one figure.

    Args:
        df_sig (pandas.dataframe): dataframe of kinematics for set of signal events 
        df_bkg (pandas.dataframe): dataframe of kinematics for set of background events 
        var (str): name of kinematic variable to plot

    Returns:
        void
    """
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
    exts = [".pdf"]
    for ext in exts:
        fig.savefig(save_path+"/"+var+ext, transparent=True)

    return 0

def plot_conv_kinematics(adj_mat, sig, bkg, kinematics, file_path):
    x = torch.cat((sig, bkg), dim=0)
    conv_x = torch.matmul(adj_mat, x)
    conv_x_numpy = conv_x.detach().numpy()

    for v, var in enumerate(kinematics):
        conv_x_numpy[:,v] = norm.standardise(conv_x_numpy[:, v])
    conv_sig = conv_x_numpy[: len(sig)]
    conv_bkg = conv_x_numpy[len(sig):]

    misc.create_dirs(file_path+"/")
    for v, var in enumerate(kinematics):
        fig, ax = plt.subplots()
        binning = numpy.linspace(min(conv_sig[:,v]),max(conv_sig[:,v]), 50)
        ax.hist(conv_sig[:,v], bins=binning, label="Signal", alpha=0.3, density=True, color="red")
        ax.hist(conv_bkg[:,v], bins=binning, label="Background", alpha=0.3, density=True, color="steelblue")
        ax.text(0.04, 0.93, "ATLAS", fontweight="bold", fontstyle="italic", verticalalignment="bottom", size=10, transform=ax.transAxes)
        ax.text(0.14, 0.93, "Internal", verticalalignment="bottom", size=10, transform=ax.transAxes)
        ax.text(0.04, 0.88, r"$\sqrt{s}=13$ TeV, 5b data", verticalalignment="bottom", size=10, transform=ax.transAxes)
        ax.text(0.04, 0.83, r"6b resonant TRSM signals", verticalalignment="bottom", size=10, transform=ax.transAxes)
        ax.text(0.04, 0.78, r"Standardised to (mean, std) = (0, 1)", verticalalignment="bottom", size=10, transform=ax.transAxes)
        ax.text(0.04, 0.73, r"Linking length at sig-sig eff 0.8", verticalalignment="bottom", size=10, transform=ax.transAxes)
        ax.legend(loc='upper right')
        ymin, ymax = ax.get_ylim()
        ax.set_ylim((ymin, ymax*1.4))
        ax.set_xlabel(str(var)+" [GeV]", loc="right")
        ax.set_ylabel("Normalised No. Events", loc="top")
        fig.savefig(file_path+"/"+var+".pdf", transparent=True)

def plot_conv_conv_kinematics(adj_mat, sig, bkg, kinematics, file_path):
    x = torch.cat((sig, bkg), dim=0)
    conv_conv_x = torch.matmul(adj_mat, torch.matmul(adj_mat, x))
    conv_conv_x_numpy = conv_conv_x.detach().numpy()

    for v, var in enumerate(kinematics):
        conv_conv_x_numpy[:,v] = norm.standardise(conv_conv_x_numpy[:, v])
    conv_conv_sig = conv_conv_x_numpy[: len(sig)]
    conv_conv_bkg = conv_conv_x_numpy[len(sig):]
    
    misc.create_dirs(file_path+"/")
    for v, var in enumerate(kinematics):
        fig, ax = plt.subplots()
        binning = numpy.linspace(min(conv_conv_sig[:,v]),max(conv_conv_sig[:,v]), 50)
        ax.hist(conv_conv_sig[:,v], bins=binning, label="Signal", alpha=0.3, density=True, color="red")
        ax.hist(conv_conv_bkg[:,v], bins=binning, label="Background", alpha=0.3, density=True, color="steelblue")
        ax.text(0.04, 0.93, "ATLAS", fontweight="bold", fontstyle="italic", verticalalignment="bottom", size=10, transform=ax.transAxes)
        ax.text(0.14, 0.93, "Internal", verticalalignment="bottom", size=10, transform=ax.transAxes)
        ax.text(0.04, 0.88, r"$\sqrt{s}=13$ TeV, 5b data", verticalalignment="bottom", size=10, transform=ax.transAxes)
        ax.text(0.04, 0.83, r"6b resonant TRSM signals", verticalalignment="bottom", size=10, transform=ax.transAxes)
        ax.text(0.04, 0.78, r"Standardised to (mean, std) = (0, 1)", verticalalignment="bottom", size=10, transform=ax.transAxes)
        ax.text(0.04, 0.73, r"Linking length at sig-sig eff 0.8", verticalalignment="bottom", size=10, transform=ax.transAxes)
        ax.legend(loc='upper right')
        ymin, ymax = ax.get_ylim()
        ax.set_ylim((ymin, ymax*1.4))
        ax.set_xlabel(str(var)+" [GeV]", loc="right")
        ax.set_ylabel("Normalised No. Events", loc="top")
        fig.savefig(file_path+"/"+var+".pdf", transparent=True)

def plot_centrality(centrality, sig, bkg, file_path, eff):
    degree_centrality = centrality / (len(sig)+len(bkg))
    norm_centrality = norm.standardise(centrality.detach().numpy())
    norm_degree_centrality = norm.standardise(degree_centrality.detach().numpy())
    misc.create_dirs(file_path+"/centrality/")
    
    fig, ax = plt.subplots()
    ax.hist(centrality[: len(sig)], bins=50, label="Signal", alpha=0.3, density=True, color="red")
    ax.hist(centrality[len(sig):], bins=50, label="Signal", alpha=0.3, density=True, color="steelblue")
    ax.text(0.04, 0.93, "ATLAS", fontweight="bold", fontstyle="italic", verticalalignment="bottom", size=10, transform=ax.transAxes)
    ax.text(0.14, 0.93, "Internal", verticalalignment="bottom", size=10, transform=ax.transAxes)
    ax.text(0.04, 0.88, r"$\sqrt{s}=13$ TeV, 5b data", verticalalignment="bottom", size=10, transform=ax.transAxes)
    ax.text(0.04, 0.83, r"6b resonant TRSM signals", verticalalignment="bottom", size=10, transform=ax.transAxes)
    ax.text(0.04, 0.78, r"Linking length at sig-sig eff " + str(eff), verticalalignment="bottom", size=10, transform=ax.transAxes)
    ax.legend(loc='upper right')
    ymin, ymax = ax.get_ylim()
    ax.set_ylim((ymin, ymax*1.4))
    ax.set_xlabel("Centrality", loc="right")
    ax.set_ylabel("Normalised No. Events", loc="top")
    fig.savefig(file_path+"/centrality/centrality_sigsig_eff_"+str(eff)+".pdf", transparent=True)

    fig, ax = plt.subplots()
    ax.hist(degree_centrality[: len(sig)], bins=50, label="Signal", alpha=0.3, density=True, color="red")
    ax.hist(degree_centrality[len(sig):], bins=50, label="Signal", alpha=0.3, density=True, color="steelblue")
    ax.text(0.04, 0.93, "ATLAS", fontweight="bold", fontstyle="italic", verticalalignment="bottom", size=10, transform=ax.transAxes)
    ax.text(0.14, 0.93, "Internal", verticalalignment="bottom", size=10, transform=ax.transAxes)
    ax.text(0.04, 0.88, r"$\sqrt{s}=13$ TeV, 5b data", verticalalignment="bottom", size=10, transform=ax.transAxes)
    ax.text(0.04, 0.83, r"6b resonant TRSM signals", verticalalignment="bottom", size=10, transform=ax.transAxes)
    ax.text(0.04, 0.78, r"Linking length at sig-sig eff " + str(eff), verticalalignment="bottom", size=10, transform=ax.transAxes)
    ax.legend(loc='upper right')
    ymin, ymax = ax.get_ylim()
    ax.set_ylim((ymin, ymax*1.4))
    ax.set_xlabel("Degree Centrality", loc="right")
    ax.set_ylabel("Normalised No. Events", loc="top")
    fig.savefig(file_path+"/centrality/degree_centrality_sigsig_eff_"+str(eff)+".pdf", transparent=True)

    fig, ax = plt.subplots()
    ax.hist(norm_centrality[: len(sig)], bins=50, label="Signal", alpha=0.3, density=True, color="red")
    ax.hist(norm_centrality[len(sig):], bins=50, label="Signal", alpha=0.3, density=True, color="steelblue")
    ax.text(0.04, 0.93, "ATLAS", fontweight="bold", fontstyle="italic", verticalalignment="bottom", size=10, transform=ax.transAxes)
    ax.text(0.14, 0.93, "Internal", verticalalignment="bottom", size=10, transform=ax.transAxes)
    ax.text(0.04, 0.88, r"$\sqrt{s}=13$ TeV, 5b data", verticalalignment="bottom", size=10, transform=ax.transAxes)
    ax.text(0.04, 0.83, r"6b resonant TRSM signals", verticalalignment="bottom", size=10, transform=ax.transAxes)
    ax.text(0.04, 0.78, r"Standardised to (mean, std) = (0, 1)", verticalalignment="bottom", size=10, transform=ax.transAxes)
    ax.text(0.04, 0.73, r"Linking length at sig-sig eff " + str(eff), verticalalignment="bottom", size=10, transform=ax.transAxes)
    ax.legend(loc='upper right')
    ymin, ymax = ax.get_ylim()
    ax.set_ylim((ymin, ymax*1.4))
    ax.set_xlabel("Centrality", loc="right")
    ax.set_ylabel("Normalised No. Events", loc="top")
    fig.savefig(file_path+"/centrality/norm_centrality_sigsig_eff_"+str(eff)+".pdf", transparent=True)

    fig, ax = plt.subplots()
    ax.hist(norm_degree_centrality[: len(sig)], bins=50, label="Signal", alpha=0.3, density=True, color="red")
    ax.hist(norm_degree_centrality[len(sig):], bins=50, label="Signal", alpha=0.3, density=True, color="steelblue")
    ax.text(0.04, 0.93, "ATLAS", fontweight="bold", fontstyle="italic", verticalalignment="bottom", size=10, transform=ax.transAxes)
    ax.text(0.14, 0.93, "Internal", verticalalignment="bottom", size=10, transform=ax.transAxes)
    ax.text(0.04, 0.88, r"$\sqrt{s}=13$ TeV, 5b data", verticalalignment="bottom", size=10, transform=ax.transAxes)
    ax.text(0.04, 0.83, r"6b resonant TRSM signals", verticalalignment="bottom", size=10, transform=ax.transAxes)
    ax.text(0.04, 0.78, r"Standardised to (mean, std) = (0, 1)", verticalalignment="bottom", size=10, transform=ax.transAxes)
    ax.text(0.04, 0.73, r"Linking length at sig-sig eff " + str(eff), verticalalignment="bottom", size=10, transform=ax.transAxes)
    ax.legend(loc='upper right')
    ymin, ymax = ax.get_ylim()
    ax.set_ylim((ymin, ymax*1.4))
    ax.set_xlabel("Degree Centrality", loc="right")
    ax.set_ylabel("Normalised No. Events", loc="top")
    fig.savefig(file_path+"/centrality/norm_degree_centrality_sigsig_eff_"+str(eff)+".pdf", transparent=True)



