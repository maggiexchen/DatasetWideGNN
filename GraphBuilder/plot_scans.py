import os
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
import matplotlib.pyplot as plt
import json
import utils.misc as misc
import argparse

def GetParser():
    """Argument parser for reading Ntuples script."""
    parser = argparse.ArgumentParser(
        description="Reading Ntuples command line options."
    )

    parser.add_argument(
        "--userconfig",
        "-u",
        type=str,
        required=True,
        help="Specify the config for the user e.g. paths to store all the input/output data and results, signal model to look at",
    )

    parser.add_argument(
        "--plot",
        "-p",
        type=str,
        required=True,
        help="Specify which type of plot to make (margin, penalty)",
    )

    args = parser.parse_args()
    return args

args = GetParser()
user_config_path = args.userconfig
user_config = misc.load_config(user_config_path)
plot_path = user_config["plot_path"]
model_path = user_config["model_path"]
margins = [0.1, 0.5, 1.0, 2.0]
penalties = [1, 5, 10, 20]
dims = [2, 4, 12]
dims_colours = ["dodgerblue", "firebrick", "darkorchid", "forestgreen", "darkorange"]
pen_colours = ["thistle", "plum", "mediumorchid", "darkviolet", "blueviolet", "rebeccapurple", "indigo"]
linestyles = ["-", "--", "dotted"]

if args.plot == "margin":
    eff_fig, eff_ax = plt.subplots(figsize=(11, 9))
    pur_fig, pur_ax = plt.subplots(figsize=(11, 9))
    for i, dim in enumerate(dims):
        same_class_eff = []
        same_class_pur = []
        sigsig_eff = []
        sigsig_pur = []
        bkgbkg_eff = []
        bkgbkg_pur = []

        for margin in margins: 
            penalty = 1
            radius = margin/2
            json_file = model_path + "embedding_"+str(dim)+"feats/EmbeddingNet_m"+str(margin)+"_r"+str(radius)+"_Lambda"+str(penalty)+"_dict.json"
            with open(json_file, 'r') as file:
                graph_dict = json.load(file)

            same_class_eff.append(graph_dict["same_class_eff"])
            same_class_pur.append(graph_dict["same_class_purity"])
            sigsig_eff.append(graph_dict["sigsig_eff"])
            sigsig_pur.append(graph_dict["sigsig_purity"])
            bkgbkg_eff.append(graph_dict["bkgbkg_eff"])
            bkgbkg_pur.append(graph_dict["bkgbkg_purity"])
        
        eff_ax.plot(margins, same_class_eff, linestyle=linestyles[0], marker='o', color=dims_colours[i])
        eff_ax.plot(margins, sigsig_eff, linestyle=linestyles[1], marker='o', color=dims_colours[i])
        eff_ax.plot(margins, bkgbkg_eff, linestyle=linestyles[2], marker='o', color=dims_colours[i])
        
        linestyle_leg = [plt.Line2D([0], [0], color="black", linestyle=l) for l in linestyles]
        colour_leg = [plt.Line2D([0], [0], lw=2, linestyle="-", marker='o', color=c) for c in dims_colours]
        
        eff_ax.set_xlabel("Margin", loc="right", fontsize=16)
        eff_ax.set_ylabel("Efficiency", loc="top", fontsize=16)
        linestyle_legend = eff_ax.legend(handles=linestyle_leg, labels=["Same class", "Sig-sig", "Bkg-bkg"], fontsize=16, frameon=False, loc="upper center")
        colour_legend = eff_ax.legend(handles=colour_leg, labels=[str(d)+" embeddings" for d in dims], fontsize=16, frameon=False, loc="upper right")
        eff_ax.add_artist(linestyle_legend)
        eff_ax.set_yscale("log")
        eff_ax.set_ylim((0.7, 1.05))
        os.makedirs(plot_path, exist_ok=True)
        eff_fig.savefig(plot_path+"eff_vs_margin.pdf")

        pur_ax.plot(margins, same_class_pur, linestyle=linestyles[0], marker='^', color=dims_colours[i])
        pur_ax.plot(margins, sigsig_pur, linestyle=linestyles[1], marker='^', color=dims_colours[i])
        pur_ax.plot(margins, bkgbkg_pur, linestyle=linestyles[2], marker='^', color=dims_colours[i])

        pur_ax.set_xlabel("Margin", loc="right", fontsize=16)
        pur_ax.set_ylabel("Purity", loc="top", fontsize=16)
        linestyle_legend = pur_ax.legend(handles=linestyle_leg, labels=["Same class", "Sig-sig", "Bkg-bkg"], fontsize=16, frameon=False, loc="upper center")
        colour_legend = pur_ax.legend(handles=colour_leg, labels=[str(d)+" embeddings" for d in dims], fontsize=16, frameon=False, loc="upper right")
        pur_ax.add_artist(linestyle_legend)
        pur_ax.set_yscale("log")
        pur_ax.set_ylim((0.3, 1.4))
        os.makedirs(plot_path, exist_ok=True)
        pur_fig.savefig(plot_path+"purity_vs_margin.pdf")

if args.plot == "penalty":
    eff_fig, eff_ax = plt.subplots(figsize=(11, 9))
    pur_fig, pur_ax = plt.subplots(figsize=(11, 9))
    dist_fig, dist_ax = plt.subplots(figsize=(11, 9))
    avg_sigsig_dist = []
    avg_bkgbkg_dist = []
    avg_sigbkg_dist = []
    sigsig_eff = []
    bkgbkg_eff = []
    # sigbkg_eff = []
    sigsig_pur = []
    bkgbkg_pur = []
    # sigbkg_pur = []
    for i, pen in enumerate(penalties):
        dim = 2
        margin = 0.1
        radius = margin/2
        json_file = model_path + "embedding_"+str(dim)+"feats/EmbeddingNet_m"+str(margin)+"_r"+str(radius)+"_Lambda"+str(pen)+"_dict.json"
        with open(json_file, 'r') as file:
            graph_dict = json.load(file)
        sigsig_eff.append(graph_dict["sigsig_eff"])
        sigsig_pur.append(graph_dict["sigsig_purity"])
        bkgbkg_eff.append(graph_dict["bkgbkg_eff"])
        bkgbkg_pur.append(graph_dict["bkgbkg_purity"])
        # sigbkg_eff.append(graph_dict["sigbkg_eff"])
        # sigbkg_pur.append(graph_dict["sigbkg_purity"])
        avg_sigsig_dist.append(graph_dict["avg_sigsig_dist"])
        avg_bkgbkg_dist.append(graph_dict["avg_bkgbkg_dist"])
        avg_sigbkg_dist.append(graph_dict["avg_sigbkg_dist"])
    eff_ax.plot(penalties, sigsig_eff,marker='o', label="Sig-sig")
    eff_ax.plot(penalties, bkgbkg_eff,marker='o', label="Bkg-bkg")
    # eff_ax.plot(penalties, sigbkg_eff, linestyle=linestyles[2], marker='o', color=pen_colours[i])

    pur_ax.plot(penalties, sigsig_pur, marker='^', label="Sig-sig")
    pur_ax.plot(penalties, bkgbkg_pur, marker='^', label="Bkg-bkg")
    # pur_ax.plot(penalties, sigbkg_pur, linestyle=linestyles[2], marker='^', color=pen_colours[i])

    dist_ax.plot(penalties, avg_sigsig_dist, marker='P', label="Sig-sig")
    dist_ax.plot(penalties, avg_bkgbkg_dist, marker='P', label="Bkg-bkg")
    dist_ax.plot(penalties, avg_sigbkg_dist, marker='P', label="Sig-bkg")


    # linestyle_leg = [plt.Line2D([0], [0], color="black", linestyle=l) for l in linestyles]
    # colour_leg = [plt.Line2D([0], [0], lw=2, linestyle="-", marker='o', color=c) for c in pen_colours]
    # eff_ax.set_xlabel(r"Penalty $\lambda$", loc="right", fontsize=16)
    # eff_ax.set_ylabel("Efficiency", loc="top", fontsize=16)
    # linestyle_legend = eff_ax.legend(handles=linestyle_leg, labels=["Sig-sig", "Bkg-bkg"], fontsize=16, frameon=False, loc="upper center")
    # colour_legend = eff_ax.legend(handles=colour_leg, labels="2 embeddings", fontsize=16, frameon=False, loc="upper right")
    # eff_ax.add_artist(linestyle_legend)
    eff_ax.set_yscale("log")
    eff_ax.legend(loc="upper right", fontsize=16)

    pur_ax.set_xlabel(r"Penalty $\lambda$", loc="right", fontsize=16)
    pur_ax.set_ylabel("Purity", loc="top", fontsize=16)
    pur_ax.legend(loc="upper right", fontsize=16)
    # linestyle_legend = pur_ax.legend(handles=linestyle_leg, labels=["Sig-sig", "Bkg-bkg"], fontsize=16, frameon=False, loc="upper center")
    # colour_legend = pur_ax.legend(handles=colour_leg, labels="2 embeddings", fontsize=16, frameon=False, loc="upper right")
    # pur_ax.add_artist(linestyle_legend)
    pur_ax.set_yscale("log")

    dist_ax.set_xlabel(r"Penalty $\lambda$", loc="right", fontsize=16)
    dist_ax.set_ylabel("Average distance", loc="top", fontsize=16)
    dist_ax.legend(loc="upper right", fontsize=16)
    # linestyle_legend = dist_ax.legend(handles=linestyle_leg, labels=["Sig-sig", "Bkg-bkg", "Sig-bkg"], fontsize=16, frameon=False, loc="upper center")
    # colour_legend = dist_ax.legend(handles=colour_leg, labels="2 embeddings", fontsize=16, frameon=False, loc="upper right")
    # dist_ax.add_artist(linestyle_legend)

    os.makedirs(plot_path, exist_ok=True)
    eff_fig.savefig(plot_path+"eff_vs_penalty.pdf")
    pur_fig.savefig(plot_path+"purity_vs_penalty.pdf")
    dist_fig.savefig(plot_path+"distance_vs_penalty.pdf")
    








            

