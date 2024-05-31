import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
import torch
import utils.normalisation as norm
import utils.misc as misc
import numpy

def add_text(ax, text, doATLAS=True, startx=0.04, starty=0.93):
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
    ax.set_ylabel("Normalised # event pairs / bin", loc="top")
    # save
    if label!="": label = "_"+label
    exts = [".pdf"]
    for ext in exts:
        fig.tight_layout()
        fig.savefig(path+"/"+var+"_"+distance+label+"_distances"+ext)

    return 0

def plot_kinematic_hists(df_sig, df_bkg, var, file_path, standardise=True):
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
    binning = np.linspace(min(min(df_sig.loc[:, var]), min(df_bkg.loc[:, var])), max(max(df_sig.loc[:, var]), max(df_bkg.loc[:, var])), 50)
    ys, xs, _ = ax.hist(df_sig.loc[:, var], bins=binning, label="Signal (6b TRSM)", alpha=0.3, color="red")
    yb, xb, _ = ax.hist(df_bkg.loc[:, var], bins=binning, label="Background (5b data)", alpha=0.3, color="steelblue")
    add_text(ax, [r"$\sqrt{s}=13$ TeV, 5b data", r"6b resonant TRSM signals"])
    if standardise:
        ax.text(0.04, 0.78, r"Standardised to (mean, std) = (0, 1)", verticalalignment="bottom", size=10, transform=ax.transAxes)
        plot_name = "/standardised_"
    else:
        plot_name = "/"
    # aesthetics
    #hep.atlas.label(ax=ax, data=False, label="Internal", lumi="129")
    ax.legend(loc='upper right')
    ax.set_ylim([0.01, 1.2*max(max(ys),max(yb))])
    ax.set_xlabel("\n"+str(var) + " [GeV]", loc="right")
    ax.set_ylabel("No. Events", loc="top")
    # save
    save_path = file_path+"/training_kinematics/"
    misc.create_dirs(save_path)
    exts = [".pdf"]
    for ext in exts:
        fig.tight_layout()
        fig.savefig(save_path+plot_name+var+ext, transparent=True)

    return 0

def plot_conv_kinematics(adj_mat, D, sig, bkg, kinematics, eff, file_path, normalisation, standardise=True, nconv=1):
    """
    Function to plot the histograms of a list of kinematics variable for signal and background on one figure, after a set number of convolutions.

    Args:
        adj_mat (torch.tensor(float32)): the adjacency matrix.
        D (torch.tensor(float32)): the Degree matrix to normalise to.
        sig (torch.tensor): kinematics for set of signal events 
        bkg (torch.tensor): kinematics for set of background events 
        kinematics (list[str]): list of names of kinematic variables to plot
        eff (float): sig-sig eff used for linking length for plot text
        file_path (str): where to save the plots to.
        normalisation (str): which way to normalise the Adj. ("D_inv", or "D_half_inv")
        standardise (bool): whether or not to standardise the variable distributions.
        nconv (int): how many times you want to convolute the kinematics.

    Returns:
        void
    """
    if normalisation == "D_inv":
        D_inv = torch.inverse(torch.diag(D))
        adj_mat = torch.matmul(D_inv, adj_mat)
        diagonal = adj_mat.diagonal()
        diagonal.fill_(1)
    elif normalisation == "D_half_inv":
        D_half_inv = torch.diag(torch.rsqrt(D))
        adj_mat = torch.matmul(D_half_inv, torch.matmul(adj_mat, D_half_inv))
        diagonal = adj_mat.diagonal()
        diagonal.fill_(1)
    else: 
        print("Please specify a normalisation that is either D_inv or D_half_inv")

    x = torch.cat((sig, bkg), dim=0).cuda()
    convcount = 0
    post_conv_x = x
    label = ""
    while convcount < nconv:
        #conv_conv_x = torch.matmul(adj_mat, torch.matmul(adj_mat, x))
        post_conv_x = torch.matmul(adj_mat, post_conv_x)
        label = label + "conv_"
        convcount ++1
    post_conv_x_numpy = post_conv_x.detach().cpu().numpy()
    
    if standardise:
        for v, var in enumerate(kinematics):
            post_conv_x_numpy[:,v] = norm.standardise(post_conv_x_numpy[:, v])
        plot_name = "/standardised_"+label
    else:
        plot_name = "/"+label
    post_conv_sig = post_conv_x_numpy[: len(sig)]
    post_conv_bkg = post_conv_x_numpy[len(sig):]
    
    # plot standardised convoluted kinematics
    misc.create_dirs(file_path+"/")
    for v, var in enumerate(kinematics):
        fig, ax = plt.subplots()
        binning = numpy.linspace(min(post_conv_bkg[:,v]),max(post_conv_bkg[:,v]), 50)
        ax.hist(post_conv_sig[:,v], bins=binning, label="Signal", alpha=0.3, density=True, color="red")
        ax.hist(post_conv_bkg[:,v], bins=binning, label="Background", alpha=0.3, density=True, color="steelblue")
        add_text(ax, [r"$\sqrt{s}=13$ TeV, 5b data", r"6b resonant TRSM signals", r"Linking length at sig-sig eff "+str(eff), "After "+nconv+" convolutions"])
        if standardise:
            ax.text(0.04, 0.68, r"Standardised to (mean, std) = (0, 1)", verticalalignment="bottom", size=10, transform=ax.transAxes)
        ax.legend(loc='upper right')
        ymin, ymax = ax.get_ylim()
        ax.set_ylim((ymin, ymax*1.4))
        ax.set_xlabel("\n"+str(var)+" [GeV]", loc="right")
        ax.set_ylabel("Normalised No. Events", loc="top")
        fig.tight_layout()
        fig.savefig(file_path+plot_name+var+".pdf", transparent=True)


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
        text = [r"$\sqrt{s}=13$ TeV, 5b data", r"6b resonant TRSM signals", r"Linking length at sig-sig eff "+str(eff)]
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
        fig.savefig(save_path+"/"+plot+"_sigsig_eff_"+str(eff)+".pdf", transparent=True)
