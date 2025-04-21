import os
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import json
import utils.misc as misc
import argparse
from glob import glob
import numpy

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
        "--scan",
        "-s",
        action="store_true",
        help="Plot scan over model parameters, need to then specify which parameter to fix in these scans",
    )
    
    parser.add_argument(
        "--fixed",
        "-f",
        type=str,
        required=False,
        help="Specify which parameter to fix (margin, penalty, dim)",
    )

    parser.add_argument(
        "--corr",
        "-c",
        action="store_true",
        help="Plot correlation between parameters and metrics"
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
dims = [2, 6, 12, 18]
parameter_dict = {"margin": {"value": margins, "label": "Margin"}, 
                  "penalty": {"value": penalties, "label": "Penalty"},
                   "dim": {"value": dims, "label": "Dimension"}}
colours = ["dodgerblue", "firebrick", "darkorchid", "forestgreen", "darkorange", "plum"]
gradient_colours = ["thistle", "plum", "mediumorchid", "darkviolet", "blueviolet", "rebeccapurple", "indigo"]
linestyles = ["-", "--", "dotted"]
s_type = ["Sig-sig", "Bkg-bkg", "Sig-bkg"]
parameters = ["margin", "penalty", "dim"]


file_list = glob(model_path + "/*/*.json")
data = []
for file in file_list:
    with open(file, 'r') as f:
        data.append(json.load(f))
df = pd.DataFrame(data)
print(df.columns)
max_pur_ind = numpy.argmax(df['same_class_purity'])
max_eff_ind = numpy.argmax(df['same_class_eff'])
print("Maximum same class purity ", df['same_class_purity'][max_pur_ind])
print("=== Dim: ", df['embedding_dim'][max_pur_ind])
print("=== Margin: ", df['loss_margin'][max_pur_ind])
print("=== Penalty: ", df["loss_penalty"][max_pur_ind])

print("Maximum same class efficiency ", df['same_class_purity'][max_eff_ind])
print("=== Dim: ", df['embedding_dim'][max_eff_ind])
print("=== Margin: ", df['loss_margin'][max_eff_ind])
print("=== Penalty: ", df["loss_penalty"][max_eff_ind])



def assign_variable(parameters, values, assignment):
    for i, parameter in enumerate(parameters):
        if parameter == assignment:
            variable = values[i]
    return variable


if args.scan:
    if args.fixed == False:
        print("Need to specify a pamaeter to fix")
    else:
        if args.fixed not in parameters:
            print("Need to specify a valid parameter to fix")
        else:
            parameters_to_plot = [s for s in parameters if s != args.fixed]

        print("Fixing ", args.fixed, ", scanning over", parameters_to_plot)

    for i, i_value in enumerate(parameter_dict[args.fixed]["value"]):
        label_i = parameter_dict[args.fixed]["label"]
        for parameter_j in parameters_to_plot:
            eff_fig, eff_ax = plt.subplots(figsize=(11, 9))
            pur_fig, pur_ax = plt.subplots(figsize=(11, 9))
            dist_fig, dist_ax = plt.subplots(figsize=(11, 9))
            label_j = parameter_dict[parameter_j]["label"]
            parameter_k = parameters_to_plot[1] if parameter_j == parameters_to_plot[0] else parameters_to_plot[0]
            label_k = parameter_dict[parameter_k]["label"]
            for j, j_value in enumerate(parameter_dict[parameter_j]["value"]):
                same_class_eff = []
                same_class_pur = []
                sigsig_eff = []
                sigsig_pur = []
                bkgbkg_eff = []
                bkgbkg_pur = []
                avg_sigsig_dist = []
                avg_bkgbkg_dist = []
                avg_sigbkg_dist = []
                for k, k_value in enumerate(parameter_dict[parameter_k]["value"]):
                    dim = assign_variable([args.fixed, parameter_j, parameter_k], [i_value, j_value, k_value], "dim")
                    margin = assign_variable([args.fixed, parameter_j, parameter_k], [i_value, j_value, k_value], "margin")
                    penalty = assign_variable([args.fixed, parameter_j, parameter_k], [i_value, j_value, k_value], "penalty")
                    radius = margin/2
                    json_file = model_path + "embedding_"+str(dim)+"feats/EmbeddingNet_m"+str(margin)+"_r"+str(radius)+"_Lambda"+str(penalty)+"_dict.json"
                    with open(json_file, 'r') as file:
                        graph_dict = json.load(file)
                    same_class_eff.append(graph_dict["same_class_eff"])
                    same_class_pur.append(graph_dict["same_class_purity"])
                    avg_sigsig_dist.append(graph_dict["avg_sigsig_dist"])
                    avg_bkgbkg_dist.append(graph_dict["avg_bkgbkg_dist"])
                    avg_sigbkg_dist.append(graph_dict["avg_sigbkg_dist"])
                eff_ax.plot(parameter_dict[parameter_k]["value"], same_class_eff, linestyle="-", marker='o', color=colours[j], label=label_j + ": " + str(j_value))
                pur_ax.plot(parameter_dict[parameter_k]["value"], same_class_pur, linestyle="-", marker='o', color=colours[j], label=label_j + ": " + str(j_value))
                for d, dist in enumerate([avg_sigsig_dist, avg_bkgbkg_dist, avg_sigbkg_dist]):
                    dist_ax.plot(parameter_dict[parameter_k]["value"], dist, linestyle=linestyles[d], marker='o', color=colours[j])
                linestyle_leg = [plt.Line2D([0], [0], color="black", linestyle=l) for l in linestyles]
                colour_leg = [plt.Line2D([0], [0], lw=2, linestyle="-", marker='o', color=colours[v], label=label_j+": "+str(value)) for v, value in enumerate(parameter_dict[parameter_j]["value"])]
                linestyle_legend = dist_ax.legend(handles=linestyle_leg, labels=s_type, fontsize=16, frameon=False, loc="upper center")
                colour_legend = dist_ax.legend(handles=colour_leg, labels=[label_j + ": " + str(value) for value in parameter_dict[parameter_j]["value"]], fontsize=16, frameon=False, loc="upper left")
                dist_ax.add_artist(linestyle_legend) 

                eff_ax.legend(loc="best", fontsize=16)
                eff_ax.set_xlabel(label_k, loc="right", fontsize=16)
                eff_ax.set_ylabel("Same-class efficiency", loc="top", fontsize=16)
                eff_ax.set_yscale("log")
                os.makedirs(plot_path+"/eff/", exist_ok=True)
                eff_fig.savefig(plot_path+"/eff/"+"eff_vs_"+parameter_j+"_"+parameter_k+"_"+args.fixed+str(i_value)+".pdf")

                pur_ax.legend(loc="best", fontsize=16)
                pur_ax.set_xlabel(label_k, loc="right", fontsize=16)
                pur_ax.set_ylabel("Same-class purity", loc="top", fontsize=16)
                pur_ax.set_yscale("log")
                os.makedirs(plot_path+"/pur/", exist_ok=True)
                pur_fig.savefig(plot_path+"/pur/"+"pur_vs_"+parameter_j+"_"+parameter_k+"_"+args.fixed+str(i_value)+".pdf")

                dist_ax.set_xlabel(label_k, loc="right", fontsize=16)
                dist_ax.set_ylabel("Average distance", loc="top", fontsize=16)
                dist_ax.set_yscale("log")
                os.makedirs(plot_path+"/dist/", exist_ok=True)
                dist_fig.savefig(plot_path+"/dist/"+"dist_vs_"+parameter_j+"_"+parameter_k+"_"+args.fixed+str(i_value)+".pdf")
            plt.close()

if args.corr:
    print("Plotting correlation map")
    tick_labels = ["Margin", "Penalty", "Radius", "Dimension", 
                   "Avg. sig-sig dist", "Avg. sig-bkg dist", "Avg. bkg-bkg dist",
                   "Edge fraction", "Same class efficiency", "Same class purity", "Sig-sig efficiency", "Sig-sig purity", "Bkg-bkg efficiency", "Bkg-bkg purity"]
    correlation_matrix = df.corr()
    
    corr_fig, corr_ax = plt.subplots(figsize=(11, 9), constrained_layout=True)
    sns.heatmap(correlation_matrix, annot=True, cbar=True, xticklabels=tick_labels, yticklabels=tick_labels, cmap="coolwarm")
    os.makedirs(plot_path+"/corr/", exist_ok=True)
    corr_plot_name = plot_path+"/corr/corr.pdf"
    corr_fig.savefig(corr_plot_name)

    