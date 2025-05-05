"""Functions for plotting"""
import math

import utils.normalisation as norm
import utils.misc as misc
import utils.variables as var_config
from var_config import var_dict

#from scipy import stats
#from scipy.stats import entropy
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
import torch

def save_fig(fig, save_str):
    """
    Function to save a figure for a set of file extensions.

    Args:
        fig (mpl.pyplot): figure to be saved
        save_str (str): full path+filename to save
    """
    exts = [".pdf"]
    for ext in exts:
        tmp_save_str = save_str + ext
        print(f"saving plot to {tmp_save_str}")
        fig.savefig(tmp_save_str)

    return


def get_x_label(var):
    """
    Function to get the LaTeX formatted variable labels
    Args:
        var (str): name of var as appears in ntuples/h5

    Returns:
        (str) variable label for plot.

    """
#    var_dict = {'xsec': "cross section [fb]", 'nEvents': "# events",
#                'genWeight': "generator weight",
#                'bjet1eta': r"$\eta(b_{1})$", 'bjet2eta': r"$\eta(b_{2})$",
#                'bjet1phi': r"$\phi(b_{1})$", 'bjet2phi': r"$\phi(b_{2})$",
#                'bjet1pt': r"$p_T(b_{1})$ [GeV]", 'bjet2pt': r"$p_T(b_{2})$ [GeV]",
#                'lep1eta': r"$\eta(l_{1})$", 'lep2eta': r"$\eta(l_{2})$",
#                'lep1phi': r"$\phi(l_{1})$", 'lep2phi': r"$\phi(l_{2})$",
#                'lep1pt': r"$p_T(l_{1})$ [GeV]", 'lep2pt': r"$p_T(l_{2})$ [GeV]",
#                'njets': r"$n_{jets}$", 'nbjets': r"$\n_{b}$",
#                'met': r"$p_{T}^{miss}$ [GeV]", 'metphi': r"$\phi^{miss}$",
#                'metsigHt': r"$p_{T}^{miss}/H_{T}~[\sqrt{GeV}]$",
#                'sumptllbb': r"$H_{T}$ [GeV]", 'sumptllbbMET': r"$H_{T} + p_{T}^{miss}$ [GeV]",
#                'mt2': r"$M_{T2}$ [GeV]",
#                'mindPhiMETl': r"$min(\Delta\phi(p_{T}^{miss},l))$",
#                'maxdPhiMETl': r"$max(\Delta\phi(p_{T}^{miss},l))$",
#                'mindPhiMETb': r"$min(\Delta\phi(p_{T}^{miss},b))$",
#                'maxdPhiMETb': r"$max(\Delta\phi(p_T}^{miss},b))$",
#                'avedPhiMETl': r"$<(\Delta\phi(p_{T}^{miss},l))>$",
#                'avedPhiMETb': r"$<(\Delta\phi(p_{T}^{miss},b))>$",
#                'mtl1': r"$m_{T}(l_{1})$ [GeV]", 'mtl2': r"$m_{T}(l_{2})$ [GeV]",
#                'mtlb1': r"$m_{T}(l, b)-close$ [GeV]", 'mtlb2': r"$m_{T}(l,b)-far$ [GeV]",
#                'mtlmin': r"$min(m_{T}(l))$ [GeV]", 'mtlbmin': r"$min(m_{T}(l,b))$ [GeV]",
#                'summtlb': r"$\Sigma(m_{T}(l,b))$ [GeV]",
#                'summtl': r"$\Sigma(m_{T}(p_{T}^{miss},l))$ [GeV]",
#                'dPhil1MET': r"$\Delta\phi(p_{T}^{miss},l_{1})$",
#                'dPhil2MET': r"$\Delta\phi(p_{T}^{miss},l_{2})",
#                'dPhib1MET': r"$\Delta\phi(p_{T}^{miss},b_{1})$",
#                'dPhib2MET': r"$\Delta\phi(p_{T}^{miss},b_{2})$",
#                'dRl1b1': r"$\Delta R(l_{1}, b_{1})$", 'dRl1b2': r"$\Delta R(l_{1}, b_{2})$",
#                'dRl2b1': r"$\Delta R(l_{2}, b_{1})$", 'dRl2b2': r"$\Delta R(l_{2}, b_{2})$",
#                'sumdRlb': r"$\Sigma(\Delta R(l,b))$", 'mindRlb': r"$min(\Delta R(l,b))$",
#                'invsumdRlb': r"$1/Sigma(\Delta R(l,b))$", 'invmindRlb': r"$1/min(\Delta R(l,b))",
#                'mH1': r'$m(H_{1})$ [GeV]', 'mH2': r'$m(H_{2})$ [GeV]', 'mH3': r'$m(H_{3})$ [GeV]',
#                'mHHH': r'$m(HHH)$ [GeV]',
#                'dRH1': r'$\Delta R(H_{1})$', 'dRH2': r'$\Delta R(H_{2})$',
#                'dRH3': r'$\Delta R(H_{3})$', 'meandRBB': r'$<\Delta R(jj)>$',
#                'sphere3dv2b': r'Sphericity$_{6j}$',
#                'sphere3dv2btrans': 'Transverse Sphericity$_{6j}$',
#                'aplan3dv2b': r'Aplanarity$_{6jets}$', 'theta3dv2b': r'$\theta_{6jets}$',
#                'feat_01': 'Feature 01', 'feat_02': 'Feature 02', 'feat_03': 'Feature 03',
#                'feat_04': 'Feature 04', 'feat_05': 'Feature 05', 'feat_06': 'Feature 06',
#                'feat_07': 'Feature 07', 'feat_08': 'Feature 08', 'feat_09': 'Feature 09',
#                'feat_10': 'Feature 10', 'feat_11': 'Feature 11', 'feat_12': 'Feature 12',
#                'feat_13': 'Feature 13', 'feat_14': 'Feature 14', 'feat_15': 'Feature 15',
#                'feat_16': 'Feature 16', 'feat_17': 'Feature 17', 'feat_18': 'Feature 18',
#                'feat_19': 'Feature 19', 'feat_20': 'Feature 20', 'feat_21': 'Feature 21'
#                }
    if var in var_dict:
        return r"{}".format(var_dict[var])
    else:
        print(f"{var} not in var dict for x label LaTeX formatting, will use var as-is")
        return var


def get_plot_labels(signal_type, signal_mass = None):
    """
    Args:
    signal_type (str): the type of signal being plotted,
      specified in the user config (hhh, LQ or stau)

    Returns:
    The signal and background labels used in plots

    """
    if signal_type == "hhh":
        signal = "6b resonant TRSM HHH signal"
        background = "Data-driven QCD background estimate (5b data)"
    elif "LQ" in signal_type:
        signal = "Leptoquark signal"
        if signal_mass is not None:
            signal += f" ({signal_mass} GeV)"
        background = r"$t\bar{t}$ and Single top backgrounds"
    elif signal_type == "stau":
        signal = "StauStau signal"
        background = r"$W$ jets, $Z\rightarrow ll$ jets, Diboson (0$l$, 1$l$, 2$l$, 3$l$, 4$l$),\
                       Triboson, Higgs, Single top, $t\bar{t}V$, $t\bar{t}$"
    else:
        raise ValueError("Signal type is either hhh, LQ or stau")

    return signal, background


def add_text(ax, text, do_atlas=False, startx=0.04, starty=0.93):
    """
    Function to add text to figures

    Args:
        ax (mpl axis): the axis of the figure to draw the text to.
        text (list[str]): the list of lines of text you want to write.
        doATLAS (bool): flag to add ATLAS Internal labelling.
        startx (float): leftmost point to align text to, as a fraction of the axis width.
        starty (float): topmost point to align text to, as a fraction of the axis height.
    """
    jump = 0.05
    if do_atlas:
        ax.text(startx, starty, "ATLAS", fontweight="bold", fontstyle="italic",\
                verticalalignment="bottom", size=10, transform=ax.transAxes)
        ax.text(startx + 0.1, starty, "Internal", verticalalignment="bottom",\
                size=10, transform=ax.transAxes)
    for i,t in enumerate(text):
        atlasdrop = jump if do_atlas else 0.0
        ax.text(startx, starty-jump*i-atlasdrop, t, verticalalignment="bottom",\
                size=10, transform=ax.transAxes)

    return


def plot_distances(sigsig, sigbkg, bkgbkg, sigsig_wgt, sigbkg_wgt, bkgbkg_wgt,
                   var, distance, path, label="", standardised=False):
    """
    Function to plot distances for sig-sig, sig-bkg and bkg-bkg on one figure.

    Args:
        sigsig (numpy.ndarray): array of the distances for each pair of sig-sig events
        sigbkg (numpy.ndarray): array of the distances for each pair of sig-bkg events
        bkgbkg (numpy.ndarray): array of the distances for each pair of bkg-bkg events
        sigsig_wgt (numpy.ndarray): array of product of eventWeights for each sig-sig event pair
        sigbkg_wgt (numpy.ndarray): array of product of eventWeights for each sig-bkg event pair
        bkgbkg_wgt (numpy.ndarray): array of product of eventWeights for each bkg-bkg event pair
        var (str): Kinematic variable
        distance (str): Distance metric (euclidean, cosine, cityblock)
        path (str): base dir to store output
        label (str): extra str for filename
    """
    # bin/range
    x_max = max(bkgbkg.max(), sigsig.max(), sigbkg.max())
    n_bins=100
    binning=np.linspace(0,x_max,n_bins)

    # plot
    fig, ax = plt.subplots()

    # plot text if needed
    plot_text = []
    if standardised:
        plot_text.append("Min-Max Standardised")
    add_text(ax, plot_text)
    print(sigsig_wgt, sigbkg_wgt, bkgbkg_wgt)
    # draw histograms
    ax.hist(sigsig, bins=binning, label="sig-sig", alpha=0.5, weights=sigsig_wgt, density=True)
    ax.hist(sigbkg, bins=binning, label="sig-bkg", alpha=0.5, weights=sigbkg_wgt, density=True)
    ax.hist(bkgbkg, bins=binning, label="bkg-bkg", alpha=0.5, weights=bkgbkg_wgt, density=True)
    # aesthesics
    ax.legend(loc='upper right')
    ax.set_xlabel(var+" "+distance +" distance", loc="right")
    ax.set_ylabel("Normalised event pairs / bin", loc="top")
    # save
    if label != "":
        label = "_" + label
    if standardised:
        label = label + "_minmaxNormed"

    save_fig(fig, path + "/" + var + "_" + distance + label + "_distances")

    return


def plot_kinematic_hists(df_sig, df_bkg, sig_label, bkg_label, var,
                         file_path, standardised=True, normalise=True,
                         log_scale=True, sig_wgts=None, bkg_wgts=None, text="", ex=""):
    """
    Function to plot the histogram of a kinematic variable for signal and background on one figure.

    Args:
        df_sig (pandas.dataframe): dataframe of kinematics for set of signal events 
        df_bkg (pandas.dataframe): dataframe of kinematics for set of background events 
        var (str): name of kinematic variable to plot
        file_path (str): where to save the plots to.
        standardised (bool): whether the variable distributions were standardised
            to have a mean of 0 and stdev of 1.
        normalise (bool): whether to normalise the distributions to have unit area
            (remember "density" normalises by binwidth as well as height).
        log_scale (bool): whether to use a log scale on the y-axis. 
        sig_wgts (pandas.series): dataframe of event weights for set of signal events 
        bkg_wgts (pandas.series): dataframe of event weights for set of background events 
        text (str): text to add to plot e.g. describing cuts placed.

    """
    # plot
    fig, ax = plt.subplots()

    plot_text = [r"$\sqrt{s}=13.6$ TeV, 370 fb$^{-1}$", sig_label, bkg_label]
    if standardised:
        plot_text.append("Standardised to (mean, std) = (0, 1)")
    if text!="":
        plot_text.append(text)
    add_text(ax, plot_text)

    xlabel = r"{}".format(get_x_label(var))

    xmin = math.floor(min(df_sig.loc[:, var].min(), df_bkg.loc[:, var].min()))
    xmax = math.ceil(max(df_sig.loc[:, var].max(), df_bkg.loc[:, var].max()))

    # don't bother plotting the variables that are all 1 value and just for sanity checking,
    #    e.g. btag should always be 1.
    if xmin == xmax:
        print("trying to plot "+var+" that just has the same value for everything, skipping")
        return
    binning = np.linspace(xmin, xmax, 25) #rounding to nearest integer, nicer in most cases

    ys, _, _ = ax.hist(df_sig.loc[:, var], bins=binning, label="Signal", alpha=0.3,
                        color="red", density=normalise, weights=sig_wgts)
    yb, _, _ = ax.hist(df_bkg.loc[:, var], bins=binning, label="Background", alpha=0.3,
                        color="steelblue", density=normalise, weights=bkg_wgts)

    if log_scale:
        ax.set_yscale("log")
        if normalise:
            ax.set_ylim([0.1*(np.min((ys, yb))+0.00001), 5*np.max((ys, yb))])
        else:
            ax.set_ylim([0.01, 5*np.max((ys, yb))])
    else:
        ax.set_ylim([0.8*np.min((ys, yb)), 1.2*np.max((ys, yb))])
    ax.legend(loc='upper right')

    ax.set_xlabel(xlabel, loc="right")
    ylabel = "Normalised Events" if normalise else "Events / Bin"
    ax.set_ylabel(ylabel, loc="top")

    # save
    if ((len(ex) > 0) and (ex[0] != "_")):
        ex = "_"+ex
    save_path = file_path+"/kinematics/"
    setting_label = ""
    if standardised:
        setting_label += "_standardised"
    if normalise:
        setting_label += "_normalised"
    if ex != "":
        setting_label += ex
    misc.create_dirs(save_path)
    fig.tight_layout()

    save_fig(fig, save_path + var + setting_label)

    return


def plot_conv_kinematics(adj_mat, x, len_sig, kinematics, kinematic_labels,\
                         signal_type, eff, file_path, normalisation, standardise=True,\
                         nconv=1, edge_wgts=False, cutstring=""):
    """
    Function to plot the histograms of a list of kinematics variable for signal and
      background on one figure, after a set number of convolutions.

    Args:
        adj_mat (torch.tensor(float32)): the adjacency matrix.
        sig (torch.tensor): kinematics for set of signal events
        bkg (torch.tensor): kinematics for set of background events
        kinematics (list[str]): list of names of kinematic variables to plot
        eff (float): sig-sig eff used for linking length for plot text
        file_path (str): where to save the plots to.
        normalisation (str): which way to normalise the Adj. ("D_inv", or "D_half_inv")
        standardise (bool): whether or not to standardise the variable distributions.
        nconv (int): how many times you want to convolute the kinematics.
        edge_wgts (bool): whether or not to include edge weights in convolution
    """
    if normalisation == "D_inv":
        d_inv = torch.inverse(torch.diag(torch.sum(adj_mat, dim=1)))
        adj_mat = torch.matmul(d_inv, adj_mat)
        diagonal = adj_mat.diagonal()
        diagonal.fill_(1)
    elif normalisation == "D_half_inv":
        d_half_inv = torch.diag(torch.rsqrt(torch.sum(adj_mat, dim=1)))
        # print("Half inv degree matrix ", D_half_inv)
        adj_mat = torch.matmul(d_half_inv, torch.matmul(adj_mat, d_half_inv))
        # print("Degree normed adj mat ", adj_mat)
        diagonal = adj_mat.diagonal()
        diagonal.fill_(1)
    else:
        print("Please specify a normalisation that is either D_inv or D_half_inv")

    convcount = 0
    post_conv_x = x
    label = ""
    while convcount < nconv:
        #conv_conv_x = torch.matmul(adj_mat, torch.matmul(adj_mat, x))
        post_conv_x = torch.matmul(adj_mat, post_conv_x)
        convcount += 1
    label = label + "conv"+str(nconv)+"_"
    post_conv_x_numpy = post_conv_x.detach().cpu().numpy()

    if standardise:
        for v, var in enumerate(kinematics):
            post_conv_x_numpy[:,v] = norm.standardise(post_conv_x_numpy[:, v])
        plot_name = "/standardised_"+label
    else:
        plot_name = "/"+label
    post_conv_sig = post_conv_x_numpy[: len_sig]
    post_conv_bkg = post_conv_x_numpy[len_sig:]

    # plot standardised convoluted kinematics
    if edge_wgts:
        conv_plot_path = file_path+"/conv_kinematics_with_edge_wgts/"
    else:
        conv_plot_path = file_path+"/conv_kinematics"
    misc.create_dirs(conv_plot_path)
    print("Saving to ", conv_plot_path)
    signal_label, background_label = get_plot_labels(signal_type)

    plot_text = [r"$\sqrt{s}=13.6$ TeV, 370 fb$^{-1}$", signal_label, background_label,
                 cutstring,
                 r"Linking length at "+str(eff)+" edge fraction",
                 "After "+str(nconv)+" convolutions",
                ]
    if standardise:
        plot_text.append("Standardised to (mean, std) = (0, 1)")

    for v, var in enumerate(kinematics):
        fig, ax = plt.subplots()
        binning = np.linspace(min(post_conv_bkg[:,v]),max(post_conv_bkg[:,v]), 50)
        ax.hist(post_conv_sig[:,v], bins=binning, label="Signal", alpha=0.3,
                density=True, color="red")
        ax.hist(post_conv_bkg[:,v], bins=binning, label="Background", alpha=0.3,
                density=True, color="steelblue")
        add_text(ax, plot_text)
        ax.legend(loc='upper right')
        ymin, ymax = ax.get_ylim()
        ax.set_ylim((ymin, ymax*1.4))
        ax.set_xlabel("\n"+str(kinematic_labels[v]), loc="right")
        ax.set_ylabel("Normalised No. Events", loc="top")
        fig.tight_layout()
        save_fig(fig, conv_plot_path + plot_name + var)

    return


def plot_centrality(centrality, sig, bkg, file_path, eff):
    """
    Function to plot degree centrality

    Args:
        centrality (torch.tensor): centralities for all the s+b events
        sig (numpy array): array of signal events
        bkg (numpy array): array of background events
        file_path (str): directory to store plot
        eff (float): sigsig-eff used for LL
        
    """
    degree_centrality = centrality.detach().cpu().numpy() / (len(sig)+len(bkg))
    norm_centrality = norm.standardise(centrality.detach().cpu().numpy())
    norm_degree_centrality = norm.standardise(degree_centrality)
    misc.create_dirs(file_path+"/centrality/")

    toplot = {
        "centrality": {"data": centrality, "xlabel":
                       "Centrality",
                       "extratext": ""},
        "degree_centrality": {"data": degree_centrality,
                              "xlabel": "Degree Centrality",
                              "extratext": ""},
        "norm_centrality": {"data": norm_centrality,
                            "xlabel": "Centrality",
                            "extratext": "Standardised to (mean, std) = (0, 1)"},
        "norm_degree_centrality": {"data": norm_degree_centrality,
                                   "xlabel": "Degree Centrality",
                                   "extratext": "Standardised to (mean, std) = (0, 1)"},
    }

    for plot, setup in toplot.items():
        fig, ax = plt.subplots()
        ax.hist(setup["data"][: len(sig)].detach().cpu().numpy(), bins=50, label="Signal",
                alpha=0.3, density=True, color="red")
        ax.hist(setup["data"][len(sig):].detach().cpu().numpy(), bins=50, label="Background",
                alpha=0.3, density=True, color="steelblue")
        text = [r"$\sqrt{s}=13$ TeV, 5b data", r"6b resonant TRSM signals",
                r"Linking length at edge fraction "+str(eff)]
        if setup["extratext"] != "":
            text.append(setup["extratext"])
        add_text(ax, text)
        ax.legend(loc='upper right')
        ymin, ymax = ax.get_ylim()
        ax.set_ylim((ymin, ymax*1.4))
        ax.set_xlabel(setup["xlabel"], loc="right")
        ax.set_ylabel("Normalised No. Events", loc="top")
        # save
        save_path = file_path+"/"+plot
        misc.create_dirs(save_path)
        save_fig(fig, save_path + "/" + plot + "_sigsig_eff_" + str(eff).replace(".","p"))

    return


def plot_linking_length(sigsig, sigbkg, bkgbkg, sigsig_wgt, sigbkg_wgt, bkgbkg_wgt,
                        sigsig_thresholds, sig_label, bkg_label, plot_path,
                        variable, distance, sigsig_eff):
    """
    Function to plot distances for sig-sig, sig-bkg and bkg-bkg on one figure.

    Args:
        sigsig (numpy.ndarray): array of the distances for each pair of sig-sig events
        sigbkg (numpy.ndarray): array of the distances for each pair of sig-bkg events
        bkgbkg (numpy.ndarray): array of the distances for each pair of bkg-bkg events
        sigsig_wgt (numpy.ndarray): array of product of eventWeights for each sig-sig event pair
        sigbkg_wgt (numpy.ndarray): array of product of eventWeights for each sig-bkg event pair
        bkgbkg_wgt (numpy.ndarray): array of product of eventWeights for each bkg-bkg event pair
        sig_sig thresholds (list(float)): list of distances for each eff. value
        sig_label (str): signal label for text
        bkg_label (str): background label for text
        plot_path (str): base dir to store plot
        variable (str): Kinematic variableset
        distance (str): Distance metric (euclidean, cosine, cityblock)
        sigsig_eff (list(float)): list of eff. values to plot
    """
    fig, ax = plt.subplots()
    n_bins = 100
    binning = np.linspace(0, torch.max(torch.cat((sigsig, sigbkg, bkgbkg))), n_bins)
    ax.hist(sigsig, bins=binning, label="sig-sig", weights=sigsig_wgt,
            alpha=0.5, density=True, color="steelblue")
    ax.hist(sigbkg, bins=binning, label="sig-bkg", weights=sigbkg_wgt,
            alpha=0.5, density=True, color="darkorange")
    ax.hist(bkgbkg, bins=binning, label="bkg-bkg", weights=bkgbkg_wgt,
            alpha=0.5, density=True, color="forestgreen")
    ax.text(0.04, 0.88, "Signal - " + sig_label, verticalalignment="bottom",
            size=10, transform=ax.transAxes)
    ax.text(0.04, 0.83, "Background - " + bkg_label, verticalalignment="bottom",
            size=10, transform=ax.transAxes)
    y_min, y_max = ax.get_ylim()
    for i, eff in enumerate(sigsig_eff):
        eff_label=str(eff*100)+"%"
        ax.axvline(x=sigsig_thresholds[i], ymax=0.6+i*0.02, linestyle="--", color="red")
        ax.text(x=sigsig_thresholds[i], y=0.63+i*0.022,
                transform=ax.get_xaxis_text1_transform(0)[0], s=eff_label,
                ha='center', va='bottom', fontsize=7)
    ax.legend(loc='upper right')
    ax.set_ylim(y_min, y_max*1.2)
    ax.set_xlabel(variable + " " + distance +" distance", loc="right")
    ax.set_ylabel("Normalised # event pairs / bin", loc="top")
    sigsigbkgbkg_path = plot_path+"linking_lengths/"
    misc.create_dirs(sigsigbkgbkg_path)
    save_fig(fig, sigsigbkgbkg_path + "/" + variable + "_" + distance + "_linking_lengths")

    return


def plot_roc(fpr_ss_sb, tpr_ss_sb, fpr_bb_sb, tpr_bb_sb, roc_auc_ss_sb, roc_auc_bb_sb,
             ss_sb_roc_cuts, bb_sb_roc_cuts, variable, distance, plot_path):
    """
    Function to plot ROC curve of distance cut efficiencies.

    Args:
        fpr_ss_sb (numpy array): array of false positive rates for ss-sb separation
        tpr_ss_sb (numpy array): array of true positive rates for ss-sb separation
        fpr_bb_sb (numpy array): array of false positive rates for bb-sb separation
        tpr_bb_sb (numpy array): array of true positive rates for bb-sb separation
        roc_auc_ss_sb (float): AUC value for ss-sb separation ROC
        roc_auc_bb_sb (float): AUC value for bb-sb separation ROC
        ss_sb_roc_cuts (numpy array): array of ss-sb values to plot for defined thresholds
        bb_sb_roc_cuts (numpy array): array of ss-sb values to plot for defined thresholds
        variable (str): variable set used
        distance (str): distance metric choice
        plot_path (str): directory of where to save plot
    """
    fig, _ = plt.subplots()
    plt.style.use(hep.style.ROOT)
    plt.plot(fpr_ss_sb, tpr_ss_sb, label='sig-sig sig-bkg (AUC = {roc_auc_ss_sb:.3f})')
    plt.plot(fpr_bb_sb, tpr_bb_sb, label='bkg-bkg sig-bkg (AUC = {roc_auc_bb_sb:.3f})')
    plt.scatter(np.array(ss_sb_roc_cuts)[:,1], np.array(ss_sb_roc_cuts)[:,0], marker='x',
                s=50, label="linking lengths",color="red")
    plt.scatter(np.array(bb_sb_roc_cuts)[:,1], np.array(bb_sb_roc_cuts)[:,0], marker='x',
                s=50, color="red")
    plt.legend(loc="lower right", fontsize="11")
    plt.ylim(0.,1.)
    plt.xlim(0.,1.)
    plt.xlabel("sig-bkg (efficiency")
    plt.ylabel("sig-sig (bkg-bkg) efficiency")
    plot_dir = plot_path+"ROC/"
    misc.create_dirs(plot_dir)
    save_fig(fig, plot_dir + "/" + variable + "_" + distance + "_ROC")

    return

def plot_roc_edge_frac(fpr_ss_sb, tpr_ss_sb, fpr_bb_sb, tpr_bb_sb, roc_auc_ss_sb, roc_auc_bb_sb,
                       ss_sb_roc_cuts, bb_sb_roc_cuts, linking_lengths,
                       variable, distance, plot_path):
    """
    Function to plot ROC curve of distance cut edge fractions.

    Args:
        fpr_ss_sb (numpy array): array of false positive edge_fracs for ss-sb separation
        tpr_ss_sb (numpy array): array of true positive edge_fracs for ss-sb separation
        fpr_bb_sb (numpy array): array of false positive edge_fracs for bb-sb separation
        tpr_bb_sb (numpy array): array of true positive edge_fracs for bb-sb separation
        roc_auc_ss_sb (float): AUC value for ss-sb separation ROC
        roc_auc_bb_sb (float): AUC value for bb-sb separation ROC
        ss_sb_roc_cuts (numpy array): array of ss-sb values to plot for defined thresholds
        bb_sb_roc_cuts (numpy array): array of ss-sb values to plot for defined thresholds
        linking_lengths (list(floats)): list of LL values to plot
        variable (str): variable set used
        distance (str): distance metric choice
        plot_path (str): directory of where to save plot
    """
    fig, _ = plt.subplots()
    plt.style.use(hep.style.ROOT)
    plt.plot(fpr_ss_sb, tpr_ss_sb, label='sig-sig sig-bkg (AUC = {roc_auc_ss_sb:.3f})')
    plt.plot(fpr_bb_sb, tpr_bb_sb, label='bkg-bkg sig-bkg (AUC = {roc_auc_bb_sb:.3f})')
    ss_sb_ind = []
    bb_sb_ind = []
    for l in linking_lengths:
        ss_sb_ind.append(np.argmin(abs(ss_sb_roc_cuts-l)))
        bb_sb_ind.append(np.argmin(abs(bb_sb_roc_cuts-l)))
    plt.scatter(np.array(fpr_ss_sb)[ss_sb_ind], np.array(tpr_ss_sb)[ss_sb_ind],
                marker='x', s=50, label="linking lengths",color="red")
    plt.scatter(np.array(fpr_bb_sb)[bb_sb_ind], np.array(tpr_bb_sb)[bb_sb_ind],
                marker='x', s=50, color="red")
    plt.legend(loc="lower right", fontsize="11")
    plt.ylim(0.,1.)
    plt.xlim(0.,1.)
    plt.xlabel("sig-bkg efficiency")
    plt.ylabel("Same class efficiency")
    plot_dir = plot_path+"ROC/"
    misc.create_dirs(plot_dir)
    save_fig(fig, plot_dir + "/" + variable + "_" + distance + "_ROC")

    return


def plot_full_distances_hist(sigsig_hists, sigbkg_hists, bkgbkg_hists,
                             var, distance, path, label="", standardised=False):
    """
    Function to plot distances for sig-sig, sig-bkg and bkg-bkg on one figure.

    Args:
        sigsig_hists (list(numpy.hist)): list of distance hists for each batch of sig-sig events
        sigbkg_hists (list(numpy.hist)): list of distance hists for each batch of sig-bkg events
        bkgbkg_hists (list(numpy.hist)): list of distance hists for each batch of bkg-bkg events
        var (str): Kinematic variable
        distance (str): Distance metric (euclidean, cosine, cityblock)
        path (str): base dir to store output
        label (str): extra str for filename
        standardised (bool): whether the distances have been min-max normed
    """

    # plot text if needed
    plot_text = []
    if standardised:
        plot_text.append("Min-Max Standardised")

    # load and sum histograms
    n_bins = sigsig_hists[0][0].shape[0]
    bins = sigsig_hists[0][1]
    bin_centres = [ bins[b] + 0.5*(bins[b+1]-bins[b]) for b in range(0,n_bins) ]
    # todo sanity check the bins are the same for all the histograms...
    sigsig_hist_total = np.zeros(n_bins)
    sigbkg_hist_total = np.zeros(n_bins)
    bkgbkg_hist_total = np.zeros(n_bins)
    for sigsig_hist in sigsig_hists:
        sigsig_hist_total = np.add(sigsig_hist[0], sigsig_hist_total)
    for sigbkg_hist in sigbkg_hists:
        sigbkg_hist_total = np.add(sigbkg_hist[0], sigbkg_hist_total)
    for bkgbkg_hist in bkgbkg_hists:
        bkgbkg_hist_total = np.add(bkgbkg_hist[0], bkgbkg_hist_total)
    # plot
    fig, ax = plt.subplots()

    add_text(ax, plot_text)
    ax.hist(bin_centres, bins=bins, label="sig-sig", alpha=0.5,
            weights=sigsig_hist_total, density=True)
    ax.hist(bin_centres, bins=bins, label="sig-bkg", alpha=0.5,
            weights=sigbkg_hist_total, density=True)
    ax.hist(bin_centres, bins=bins, label="bkg-bkg", alpha=0.5,
            weights=bkgbkg_hist_total, density=True)
    # aesthesics
    ax.legend(loc='upper right')
    ax.set_xlabel(var+" "+distance +" distance", loc="right")
    ax.set_ylabel("Normalised # event pairs / bin", loc="top")
    # save
    if label!="":
        label = "_" + label
    if standardised:
        label = label + "_minmaxNormed"
    fig.tight_layout()
    save_fig(fig, path + "/" + var + "_" + distance + label + "_distances")

    return

def plot_distances_hist(sigsig_hist, sigbkg_hist, bkgbkg_hist, var, distance,
                        sig_label, bkg_label, path, label="", standardised=False):
    """
    Function to plot distances for sig-sig, sig-bkg and bkg-bkg on one figure.

    Args:
        sigsig_hist (numpy.hist): histogram ofthe distances for each pair of sig-sig events
        sigbkg_hist (numpy.hist): histogram ofthe distances for each pair of sig-bkg events
        bkgbkg_hist (numpy.hist): histogram ofthe distances for each pair of bkg-bkg events
        var (str): Kinematic variable
        distance (str): Distance metric (euclidean, cosine, cityblock)
        sig_label (str): label for signal to draw on plot
        bkg_label (str): label for background to draw on plot
        path (str): base dir to store output
        label (str): extra str for filename
        standardised (bool): whether the distances have been min-max normed
    """

    # plot text if needed
    plot_text = [r"$\sqrt{s}=13.6$ TeV, 370 fb$^{-1}$", sig_label, bkg_label]
    if standardised:
        plot_text.append("Min-Max Standardised")

    # plot
    fig, ax = plt.subplots()

    add_text(ax, plot_text)
    bins = sigsig_hist[1]
    n_bins = sigsig_hist[0].shape[0]
    bin_centres = [ bins[b] + 0.5*(bins[b+1]-bins[b]) for b in range(0,n_bins) ]
    ax.hist(bin_centres, bins=sigsig_hist[1], label="sig-sig", alpha=0.5,
            weights=sigsig_hist[0], density=True, color="steelblue")
    ax.hist(bin_centres, bins=sigbkg_hist[1], label="sig-bkg", alpha=0.5,
            weights=sigbkg_hist[0], density=True, color="darkorange")
    ax.hist(bin_centres, bins=bkgbkg_hist[1], label="bkg-bkg", alpha=0.5,
            weights=bkgbkg_hist[0], density=True, color="forestgreen")

    # aesthetics
    ax.legend(loc='upper right')
    ax.set_xlabel(var+" "+distance +" distance", loc="right")
    ax.set_ylabel("Normalised # event pairs / bin", loc="top")
    # save
    if label!="":
        label = "_"+label
    if standardised:
        label = label + "_minmaxNormed"
    fig.tight_layout()
    save_fig(fig, path + "/" + var + "_" + distance + label + "_distances")

    return

def plot_event_weights(df_sig, signal, df_bkgs, backgrounds, h5_path, signal_mass="", cutstring=""):
    """
    Function to plot event weight distribution

    Args:
        df_sig (torch.tensor): dataframe of signal events
        df_bkgs (list(torch.tensor)): list of dataframes for each background
        backgrouds (list(str)): list of backgrounds
        plot_path (str): directory to save plot in
    """
    fig, ax = plt.subplots()
    _, _, _ = ax.hist(df_sig[signal]["eventWeight"], histtype="step", bins=100, label=signal)

    for background in backgrounds:
        ax.hist(df_bkgs[background]["eventWeight"], histtype="step", bins=100, label=background)

    ax.legend(loc='upper right', fontsize=9)
    ax.set_yscale("log")
    ax.set_xlabel("Event weight", loc="right")
    ax.set_ylabel("No. Events", loc="top")
    save_path = h5_path + "/eventweight_check" + signal
    if signal_mass != "":
        save_path = save_path + "_" + signal_mass
    if cutstring != "":
        save_path = save_path + "_" + cutstring
    save_fig(fig, save_path)

    return


def plot_linking_length_hist(sigsig_hist, sigbkg_hist, bkgbkg_hist,
                             sigsig_thresholds,sig_label, bkg_label, plot_path,
                             variable, distance, sigsig_eff, standardised=False):
    """
    Function to plot distances for sig-sig, sig-bkg and bkg-bkg on one figure.

    Args:
        sigsig_hist (numpy.ndarray): sig-sig distance histogram
        sigbkg_hist (numpy.ndarray): sig-bkg distance histogram
        bkgbkg_hist (numpy.ndarray): bkg-bkg distance histogram
        sig_sig thresholds (list(float)): list of distances for each eff. value
        sig_label (str): signal label for text
        bkg_label (str): background label for text
        plot_path (str): base dir to store plot
        variable (str): Kinematic variableset
        distance (str): Distance metric (euclidean, cosine, cityblock)
        sigsig_eff (list(float)): list of eff. values to plot
        standardised (bool): whether to indicate the distance metric was
            standardised to 0-1 (default false)
    """
    # plot text if needed
    plot_text = [r"$\sqrt{s}=13.6$ TeV, 370 fb$^{-1}$", sig_label, bkg_label]
    if standardised:
        plot_text.append("Min-Max Standardised")

    fig, ax = plt.subplots()

    add_text(ax, plot_text)
    bins = sigsig_hist[1]
    n_bins = sigsig_hist[0].shape[0]
    bin_centres = [bins[b] + 0.5*(bins[b+1]-bins[b]) for b in range(0,n_bins)]
    ax.hist(bin_centres, bins=sigsig_hist[1], label="sig-sig", alpha=0.5,
            weights=sigsig_hist[0], density=True, color="steelblue")
    ax.hist(bin_centres, bins=sigbkg_hist[1], label="sig-bkg", alpha=0.5,
            weights=sigbkg_hist[0], density=True, color="darkorange")
    ax.hist(bin_centres, bins=bkgbkg_hist[1], label="bkg-bkg", alpha=0.5,
            weights=bkgbkg_hist[0], density=True, color="forestgreen")
    y_min, y_max = ax.get_ylim()
    for i, eff in enumerate(sigsig_eff):
        eff_label = str(eff*100)+ "%"
        ax.axvline(x=sigsig_thresholds[i], ymax=0.6 + i*0.02, linestyle="--", color="red")
        ax.text(x=sigsig_thresholds[i], y=0.63 + i*0.022,
                transform=ax.get_xaxis_text1_transform(0)[0], s=eff_label,
                ha='center', va='bottom', fontsize=7)
    ax.legend(loc='upper right')
    ax.set_ylim(y_min, y_max*1.2)
    ax.set_xlabel(variable + " " + distance + " distance", loc="right")
    ax.set_ylabel("Normalised # event pairs / bin", loc="top")
    ll_path = plot_path+"linking_lengths/"
    misc.create_dirs(ll_path)
    fig.tight_layout()
    save_fig(fig, ll_path + "/" + variable + "_" + distance + "_linking_lengths")

    return
