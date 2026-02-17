import os
import sys
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import json
import re
import argparse
from glob import glob
import numpy
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
import utils.misc as misc

def GetParser():
    """Argument parser for reading Ntuples script."""
    parser = argparse.ArgumentParser(
        description="Reading Ntuples command line options."
    )

    parser.add_argument(
        "--model",
        "-m",
        required=True,
        help="Specify the type of model scanned (DNN, GCN or Graph)",
    )

    parser.add_argument(
        "--input_variable",
        "-i",
        type=str,
        required=True,
        help="Sepcify the type of kinematic variables used in model training (LQ_LowLevel, LQ_HighLevel)"
    )
    parser.add_argument(
        "--distance_variable",
        "-dv",
        type=str,
        required=False,
        help="In the case of a GNN model, specify the type of kinematic variables used to calculate distances"
    )
    parser.add_argument(
        "--distance",
        "-d",
        type=str,
        required=False,
        help="In the case of a GNN model, sepcify the type of distance metric used"
    )
    parser.add_argument(
        "--edge_frac",
        "-e",
        type=float,
        required=False,
        help="In the case of a GNN model, specify the edge fraction of the graph"
    )

    args = parser.parse_args()
    return args

args = GetParser()
model_type = args.model
input_variable = args.input_variable

if model_type == "DNN":
    opt_path = os.path.join(parent_dir, "hyperparameter_optimisation", model_type+"_Inputs_"+input_variable)
    user_config_path = os.path.join(opt_path, "metadata", "user_Maggie_DNN_scan.yaml")
    scan_script = os.path.join(opt_path, "metadata", "train_DNN.sh")
    param_dict = {
        "BATCHSIZE": int,
        "DROPOUT": int,
        "LR": float,
        "LR_PATIENCE": int,
        "MLP_HIDDEN_NODES": int,
        "MLP_LAYERS": int,
    }
elif (model_type == "GCN") or (model_type == "Graph"):
    distance_var = args.distance_variable
    distance = args.distance
    edge_frac = args.edge_frac
    opt_path = os.path.join(parent_dir, "hyperparameter_optimisation", model_type+"_"+distance+"_"+distance_var+"_Inputs_"+input_variable+"_EdgeFrac_"+str(edge_frac))
    user_config_path = os.path.join(opt_path, "metadata", "user_Maggie_GNN_scan.yaml")
    scan_script = os.path.join(opt_path, "metadata", "train_GNN.sh")
    param_dict = {
        "BATCHSIZE": int,
        "DROPOUT": int,
        "LR": float,
        "LR_PATIENCE": int,
        "MLP_HIDDEN_NODES": int,
        "MLP_LAYERS": int,
        "GNN_HIDDEN_NODES": int,
        "GNN_LAYERS": int,
        "NEIGHBOURS": int,
        "NEIGHBOURS_LAYERS": int,
    }
else:
    raise ValueError("Please specify a valid model type (DNN, GCN, Graph)")

param_values = {}

with open(scan_script, "r") as f:
    for line in f:
        line = line.strip()

        m = re.match(r"(\w+)=\(([^)]*)\)", line)
        if not m:
            continue
        name, values = m.groups()

        if name not in param_dict:
            continue

        caster = param_dict[name]
        param_values[name] = [caster(v) for v in values.split()]

user_config = misc.load_config(user_config_path)
plot_path = user_config["plot_path"]
model_path = user_config["model_path"]

perf_file_list = glob(model_path + "/*/*/performance.json")
metadata_file_list = glob(model_path + "/*/*/metadata.json")
assert len(perf_file_list) == len(metadata_file_list), "Unequal number of performance files and metadata files"
print("Found ", len(perf_file_list), " models")

performance = []
metadata = []
for file in perf_file_list:
    with open(file, 'r') as f:
        performance.append(json.load(f))
for file in metadata_file_list:
    with open(file, 'r') as f:
        metadata.append(json.load(f))
df_perf = pd.DataFrame(performance)
df_metadata = pd.DataFrame(metadata)
metrics = df_perf.columns
parameters = df_metadata.columns

df_perf["min train loss"] = df_perf["train_loss"].apply(min)
df_perf["min val loss"] = df_perf["val_loss"].apply(min)

max_val_auc_ind = numpy.argmax(df_perf["val_auc"])
print("Maximum val AUC found: ", df_perf["val_auc"][max_val_auc_ind])
print("Model parameters that yield the maximum val AUC: \n", df_metadata.iloc[max_val_auc_ind])

df_parameters = pd.DataFrame({
    "Batch size": df_metadata["batch_size"],
    "Dropout rate": df_metadata["dropout_rates"].apply(lambda x: x[0]),
    "LR": df_metadata["learning_rate"],
    "LR patience": df_metadata["learning_rate_patience"],
    "MLP hidden nodes": df_metadata["hidden_sizes_mlp"].apply(lambda x: x[0]),
    "MLP hidden layers": df_metadata["hidden_sizes_mlp"].apply(len)
})

if (model_type == "GCN") or (model_type == "Graph"):
    df_parameters["GNN hidden nodes"] = df_metadata["hidden_sizes_gcn"].apply(lambda x: x[0])
    df_parameters["GNN hidden layers"] = df_metadata["hidden_sizes_gcn"].apply(len)
    df_parameters["neighbours sampled"] = df_metadata["neighbour_sampling"].apply(lambda x: x[0])
    df_parameters["neighbours layers"] = df_metadata["neighbours_sampling"].apply(len)

print("Plotting correlation map")
df_corr = pd.concat([df_parameters, df_perf["train_auc"], df_perf["val_auc"]], axis=1).corr()
metric_cols = ["train_auc", "val_auc"]
param_cols = df_parameters.columns
corr_fig, corr_ax = plt.subplots(figsize=(16, 4), constrained_layout=True)
sns.heatmap(df_corr.loc[metric_cols, param_cols], annot=True, cbar=True, cmap="coolwarm", cbar_kws={'label': 'Correlation'})
cbar = corr_ax.collections[0].colorbar
cbar.ax.yaxis.label.set_rotation(270)
cbar.ax.yaxis.labelpad = 20
cbar.ax.yaxis.label.set_fontsize(16)
corr_ax.tick_params(axis='x', labelsize=16)  # Set font size for x-axis tick labels
corr_ax.tick_params(axis='y', labelsize=16)
corr_plot_name = plot_path+"scan_correlation.pdf"
corr_fig.savefig(corr_plot_name)
