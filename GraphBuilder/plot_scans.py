import os
import matplotlib.pyplot as plt
import json
import utils.misc as misc

user_config = misc.load_config(user_config_path)
model_path = user_config["model_path"]
margins = [0.1, 0.5, 1.0, 1.5, 2]
penalties = [1, 2, 3, 4]
dims = [2, 3, 4, 10, 24]
dims_colours = ["dodgerblue", "firebrick", "darkorchid", "forestgreen", "darkorange"]
linestyles = ["-", "--", "dotted"]

eff_fig, eff_ax = plt.subplots(figsize=(11, 9))
pur_fig, pur_ax = plt.subplots(figsize=(11, 9))
plot_path = "scan_plots/"

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



            

