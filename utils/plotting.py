import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
import torch
import utils.normalisation as norm
import utils.misc as misc
import numpy
from scipy import stats
from scipy.stats import entropy

def get_plot_labels(signal_type):
    """
    Args:
    signal_type (str): the type of signal being plotted, specified in the user config (hhh, LQ or stau)
    
    Returns:
    The signal and background labels used in plots

    """
    if signal_type == "hhh":
        signal = "6b resonant TRSM HHH signal"
        background = "Data-driven QCD background estimate (5b data)"
    elif signal_type == "LQ":
        signal = r"Leptoquark signal ($m_{LQ}$ = 1 TeV)"
        background = r"$t\bar{t}$ and Single top backgrounds"
    elif signal_type == "stau":
        signal = "StauStau signal"
        background = r"$W$ jets, $Z\rightarrow ll$ jets, Diboson (0$l$, 1$l$, 2$l$, 3$l$, 4$l$), Triboson, Higgs, Single top, $t\bar{t}V$, $t\bar{t}$"
    return signal, background


def add_text(ax, text, doATLAS=False, startx=0.04, starty=0.93):
    """
    Function to add text to figures
    
    Args:
        ax (mpl axis): the axis of the figure to draw the text to.
        text (list[str]): the list of lines of text you want to write.
        doATLAS (bool): flag to add ATLAS Internal labelling.
        startx (float): leftmost point the text will align too, as a fraction of the axis width.
        starty (float): topmost point thee text will align to, as a fraction of the axis height.
    """
    jump = 0.05
    if doATLAS:
        ax.text(startx, starty, "ATLAS", fontweight="bold", fontstyle="italic", verticalalignment="bottom", size=10, transform=ax.transAxes)
        ax.text(startx + 0.1, starty, "Internal", verticalalignment="bottom", size=10, transform=ax.transAxes)
    for i,t in enumerate(text):
        atlasdrop = jump if doATLAS else 0.0
        ax.text(startx, starty-jump*i-atlasdrop, t, verticalalignment="bottom", size=10, transform=ax.transAxes)

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
    ax.set_xlabel(var+" "+distance +" distance", loc="right")
    ax.set_ylabel("Normalised event pairs / bin", loc="top")
    # ax.set_xlim((0, 1650))
    # ax.set_yscale('log')
    # save
    if label!="": label = "_"+label
    exts = [".pdf"]
    for ext in exts:
        fig.tight_layout()
        fig.savefig(path+"/"+var+"_"+distance+label+"_distances"+ext, transparent=True)

    return 0

def plot_kinematic_hists(df_sig, df_bkg, sig_label, bkg_label, var, var_label, file_path, standardise=True):
    """
    Function to plot the histogram of a kinematic variable for signal and background on one figure.

    Args:
        df_sig (pandas.dataframe): dataframe of kinematics for set of signal events 
        df_bkg (pandas.dataframe): dataframe of kinematics for set of background events 
        var (str): name of kinematic variable to plot
        file_path (str): where to save the plots to.
        standardise (bool): whether or not to standardise the variable distributions.

    Returns:
        void
    """
    # plot
    fig, ax = plt.subplots()
    if standardise:
        add_text(ax, [r"$\sqrt{s}=13$ TeV, 370 fb$^{-1}$", sig_label, bkg_label, r"$E_T^{miss}$ > 200 GeV ", r"Standardised to (mean, std) = (0, 1)"])
        plot_name = "/standardised_"
        binning = np.linspace(min(min(df_sig.loc[:, var]), min(df_bkg.loc[:, var])), max(max(df_sig.loc[:, var]), max(df_bkg.loc[:, var])), 50)
        bool_density = True
    else:
        add_text(ax, [r"$\sqrt{s}=13$ TeV, 370 fb$^{-1}$", sig_label, bkg_label, r"$E_T^{miss}$ > 200 GeV "])
        plot_name = "/"
        binning = np.linspace(min(min(df_sig.loc[:, var]), min(df_bkg.loc[:, var])), max(max(df_sig.loc[:, var]), max(df_bkg.loc[:, var])), 50)
        bool_density = True
    yb, xb, _ = ax.hist(df_bkg.loc[:, var], bins=binning, label="Background", alpha=0.3, color="steelblue", density=bool_density, weights=df_bkg["eventWeight"])
    ys, xs, _ = ax.hist(df_sig.loc[:, var], bins=binning, label="Signal", alpha=0.3, color="red", density=bool_density, weights=df_sig["eventWeight"])
    ax.legend(loc='upper right')
    ax.set_ylim([0, 1.2*max(max(ys),max(yb))])
    ax.set_xlabel("\n"+str(var_label), loc="right")
    if bool_density:
        ax.set_ylabel("Normalised Events / Bin", loc="top")
    else:
        ax.set_ylabel("Events / Bin", loc="top")
    # save
    save_path = file_path+"/training_kinematics/"
    misc.create_dirs(save_path)
    exts = [".pdf"]
    for ext in exts:
        fig.tight_layout()
        fig.savefig(save_path+plot_name+var+ext, transparent=True)
    return 0

def plot_conv_kinematics(adj_mat, x, len_sig, len_bkg, kinematics, kinematic_labels, signal_type, eff, file_path, normalisation, standardise=True, nconv=1, edge_wgts=False ):
    """
    Function to plot the histograms of a list of kinematics variable for signal and background on one figure, after a set number of convolutions.

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

    Returns:
        void
    """
    if normalisation == "D_inv":
        D_inv = torch.inverse(torch.diag(torch.sum(adj_mat, dim=1)))
        adj_mat = torch.matmul(D_inv, adj_mat)
        diagonal = adj_mat.diagonal()
        diagonal.fill_(1)
    elif normalisation == "D_half_inv":
        D_half_inv = torch.diag(torch.rsqrt(torch.sum(adj_mat, dim=1)))
        # print("Half inv degree matrix ", D_half_inv)
        adj_mat = torch.matmul(D_half_inv, torch.matmul(adj_mat, D_half_inv))
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

    for v, var in enumerate(kinematics):
        fig, ax = plt.subplots()
        binning = numpy.linspace(min(post_conv_bkg[:,v]),max(post_conv_bkg[:,v]), 50)
        ax.hist(post_conv_sig[:,v], bins=binning, label="Signal", alpha=0.3, density=True, color="red")
        ax.hist(post_conv_bkg[:,v], bins=binning, label="Background", alpha=0.3, density=True, color="steelblue")
        if standardise:
            add_text(ax, [r"$\sqrt{s}=13$ TeV, 370 fb$^{-1}$", signal_label, background_label, r"$E_T^{miss}$ > 200 GeV ", r"Linking length at "+str(eff)+" edge fraction", "After "+str(nconv)+" convolutions", r"Standardised to (mean, std) = (0, 1)"])
        else:
            add_text(ax, [r"$\sqrt{s}=13$ TeV, 370 fb$^{-1}$", signal_label, background_label, r"$E_T^{miss}$ > 200 GeV ", r"Linking length at "+str(eff)+" edge fraction", "After "+str(nconv)+" convolutions"])
        ax.legend(loc='upper right')
        ymin, ymax = ax.get_ylim()
        ax.set_ylim((ymin, ymax*1.4))
        ax.set_xlabel("\n"+str(kinematic_labels[v]), loc="right")
        ax.set_ylabel("Normalised No. Events", loc="top")
        fig.tight_layout()
        fig.savefig(conv_plot_path+plot_name+var+".pdf", transparent=True)


def plot_centrality(centrality, sig, bkg, file_path, eff):
    degree_centrality = centrality.detach().cpu().numpy() / (len(sig)+len(bkg))
    norm_centrality = norm.standardise(centrality.detach().cpu().numpy())
    norm_degree_centrality = norm.standardise(degree_centrality)
    misc.create_dirs(file_path+"/centrality/")
    
    toplot = {
        "centrality": { "data": centrality, "xlabel": "Centrality", "extratext": ""},
        "degree_centrality": { "data": degree_centrality, "xlabel": "Degree Centrality", "extratext": ""},
        "norm_centrality": { "data": norm_centrality, "xlabel": "Centrality", "extratext": "Standardised to (mean, std) = (0, 1)"},
        "norm_degree_centrality": { "data": norm_degree_centrality, "xlabel": "Degree Centrality", "extratext": "Standardised to (mean, std) = (0, 1)"},
    }

    for plot, setup in toplot.items():
        fig, ax = plt.subplots()
        ax.hist(setup["data"][: len(sig)].detach().cpu().numpy(), bins=50, label="Signal", alpha=0.3, density=True, color="red")
        ax.hist(setup["data"][len(sig):].detach().cpu().numpy(), bins=50, label="Background", alpha=0.3, density=True, color="steelblue")
        text = [r"$\sqrt{s}=13$ TeV, 5b data", r"6b resonant TRSM signals", r"Linking length at edge fraction "+str(eff)]
        if setup["extratext"] != "": text.append(setup["extratext"])
        add_text(ax, text)
        ax.legend(loc='upper right')
        ymin, ymax = ax.get_ylim()
        ax.set_ylim((ymin, ymax*1.4))
        ax.set_xlabel(setup["xlabel"], loc="right")
        ax.set_ylabel("Normalised No. Events", loc="top")
        # save
        save_path = file_path+"/"+plot
        misc.create_dirs(save_path)
        print("Saving to ", save_path+"/"+plot)
        fig.savefig(save_path+"/"+plot+"_sigsig_eff_"+str(eff)+".pdf", transparent=True)

def plot_linking_length(sigsig, sigbkg, bkgbkg, sigsig_wgt, sigbkg_wgt, bkgbkg_wgt, ss_thresholds, sig_label, bkg_label, plot_path, variable, distance, sigsig_eff):
    fig, ax = plt.subplots()
    nBins = 50
    binning = np.linspace(0, torch.max(torch.cat((sigsig, sigbkg, bkgbkg))), nBins)
    sigsig_hist, bin_edges = numpy.histogram(sigsig, bins=binning, weights=sigsig_wgt, density=True)
    sigbkg_hist, _ = numpy.histogram(sigbkg, bins=binning, weights=sigbkg_wgt, density=True)
    bkgbkg_hist, _ = numpy.histogram(bkgbkg, bins=binning, weights=bkgbkg_wgt, density=True)
    bin_center = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    sigsig_mode = bin_center[numpy.argmax(sigsig_hist)]
    sigbkg_mode = bin_center[numpy.argmax(sigbkg_hist)]
    bkgbkg_mode = bin_center[numpy.argmax(bkgbkg_hist)]
    KL_sigsig_sigbkg = entropy(sigsig_hist+1e-10, sigbkg_hist+1e-10)
    KL_bkgbkg_sigbkg = entropy(bkgbkg_hist+1e-10, sigbkg_hist+1e-10)
    KL_sigsig_bkgbkg = entropy(sigsig_hist+1e-10, bkgbkg_hist+1e-10)
    ax.hist(sigsig, bin_edges, label=f"sig-sig (mode: {sigsig_mode:.3g})", color="steelblue", alpha=0.5, weights=sigbkg_wgt, density=True)
    ax.hist(sigbkg, bin_edges, label=f"sig-bkg (mode: {sigbkg_mode:.3g})", color="darkorange", alpha=0.5, weights=sigbkg_wgt, density=True)
    ax.hist(bkgbkg, bin_edges, label=f"bkg-bkg (mode: {bkgbkg_mode:.3g})", color="forestgreen", alpha=0.5, weights=sigbkg_wgt, density=True)
    add_text(ax, [sig_label, bkg_label, r"$KL_{sigsig, sigbkg}$: " + f"{KL_sigsig_sigbkg:.3g}", r"$KL_{bkgbkg, sigbkg}$: " + f"{KL_bkgbkg_sigbkg:.3g}", r"$KL_{sigsig, bkgbkg}$: " + f"{KL_sigsig_bkgbkg:.3g}"])
    y_min, y_max = ax.get_ylim()
    x_min, x_max = ax.get_xlim()
    for i, eff in enumerate(sigsig_eff):
        # eff_label=str(eff*100)+"%"
        eff_label = str(eff)
        ax.axvline(x=ss_thresholds[i], ymax=0.6+i*0.02, linestyle="--", color="red")
        ax.text(x=ss_thresholds[i], y=0.6+i*0.022, transform=ax.get_xaxis_text1_transform(0)[0], s=eff_label, ha='center', va='bottom', fontsize=7)
    ax.legend(loc='upper right')
    ax.set_ylim(y_min, y_max*1.3)
    kinematic_label = misc.get_kinematics_labels(variable)
    ax.set_xlabel(kinematic_label + " " + distance +" distance", loc="right")
    ax.set_ylabel("Normalised # event pairs / bin", loc="top")
    ssbb_path = plot_path+"linking_lengths/"
    misc.create_dirs(ssbb_path)
    fig.tight_layout()
    fig.savefig(ssbb_path+"/"+variable+"_"+distance+"_linking_lengths.pdf", transparent=True)

def plot_ROC(fpr_ss_sb, tpr_ss_sb, fpr_bb_sb, tpr_bb_sb, roc_auc_ss_sb, roc_auc_bb_sb, ss_sb_roc_cuts, bb_sb_roc_cuts, variable, distance, plot_path):
    fig, ax = plt.subplots()
    plt.style.use(hep.style.ROOT)
    plt.plot(fpr_ss_sb, tpr_ss_sb, label='sig-sig sig-bkg ROC curve (AUC = {:.3f})'.format(roc_auc_ss_sb))
    plt.plot(fpr_bb_sb, tpr_bb_sb, label='bkg-bkg sig-bkg ROC curve (AUC = {:.3f})'.format(roc_auc_bb_sb))
    plt.scatter(np.array(ss_sb_roc_cuts)[:,1], np.array(ss_sb_roc_cuts)[:,0], marker='x', s=50, label="linking lengths",color="red")
    plt.scatter(np.array(bb_sb_roc_cuts)[:,1], np.array(bb_sb_roc_cuts)[:,0], marker='x', s=50, color="red")
    plt.legend(loc="lower right", fontsize="11")
    ymin, ymax = plt.ylim()
    plt.ylim(0.,1.)
    plt.xlim(0.,1.)
    plt.xlabel("sig-bkg (efficiency")
    plt.ylabel("sig-sig (bkg-bkg) efficiency")
    plot_dir = plot_path+"ROC/"
    misc.create_dirs(plot_dir)
    fig.savefig(plot_dir+"/"+variable+"_"+distance+"_ROC.pdf", transparent=True)

def plot_ROC_edge_frac(fpr_ss_sb, tpr_ss_sb, fpr_bb_sb, tpr_bb_sb, roc_auc_ss_sb, roc_auc_bb_sb, ss_sb_roc_cuts, bb_sb_roc_cuts, linking_lengths, variable, distance, plot_path):
    fig, ax = plt.subplots()
    plt.style.use(hep.style.ROOT)
    plt.plot(fpr_ss_sb, tpr_ss_sb, label='sig-sig sig-bkg ROC curve (AUC = {:.3f})'.format(roc_auc_ss_sb))
    plt.plot(fpr_bb_sb, tpr_bb_sb, label='bkg-bkg sig-bkg ROC curve (AUC = {:.3f})'.format(roc_auc_bb_sb))
    ss_sb_ind = []
    bb_sb_ind = []
    for l in linking_lengths:
        ss_sb_ind.append(numpy.argmin(abs(ss_sb_roc_cuts-l)))
        bb_sb_ind.append(numpy.argmin(abs(bb_sb_roc_cuts-l)))
    plt.scatter(np.array(fpr_ss_sb)[ss_sb_ind], np.array(tpr_ss_sb)[ss_sb_ind], marker='x', s=50, label="linking lengths",color="red")
    plt.scatter(np.array(fpr_bb_sb)[bb_sb_ind], np.array(tpr_bb_sb)[bb_sb_ind], marker='x', s=50, color="red")
    plt.legend(loc="lower right", fontsize="11")
    ymin, ymax = plt.ylim()
    plt.ylim(0.,1.)
    plt.xlim(0.,1.)
    plt.xlabel("sig-bkg efficiency")
    plt.ylabel("Same class efficiency")
    plot_dir = plot_path+"ROC/"
    misc.create_dirs(plot_dir)
    fig.savefig(plot_dir+"/"+variable+"_"+distance+"_ROC.pdf", transparent=True)
