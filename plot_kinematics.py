"""Module to plot kinematics from h5 files."""
import logging
import argparse

import utils.normalisation as norm
import utils.misc as misc
import utils.plotting as plotting

import torch
import pandas as pd

torch.manual_seed(42)
logging.getLogger().setLevel(logging.INFO)

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
    "--userconfig",
    "-u",
    type=str,
    required=True,
    help="Specify the config for the user.",
)

args = parser.parse_args()

user_config_path = args.userconfig
user_config = misc.load_config(user_config_path)

signal = str(user_config["signal"])
signal_mass = str(user_config["signal_mass"])
feature_dim = user_config["feature_dim"]
assert signal in ["hhh", "LQ", "stau"], f"Invalid signal type: {signal}"
cuts = user_config["cuts"]
cutstring = misc.get_cutstring(cuts)
variable = str(args.variable)

feature_h5_path = user_config["feature_h5_path"]
plot_path = user_config["plot_path"] + "/" + signal + "/"
dist_path = user_config["dist_path"]

logging.info("signal: %s", signal)
logging.info("variable set: %s", variable)
logging.info("input data path: %s", feature_h5_path)
logging.info("input distances path: %s", dist_path)
logging.info("output plot path: %s", plot_path)
kinematics = misc.get_kinematics(variable, feature_dim)

# load in input files
logging.info('Importing signal and background files...')
if signal == "hhh":
    SF_4b5b = 0.07 # placeholder value for HHH data-driven background.

signal_label, background_label = plotting.get_plot_labels(signal)
bkg_types = misc.get_background_types(signal)
df_bkg = pd.DataFrame()

if signal == "stau":
    logging.info("Loading stau signal sample(s) ...")
    camps = ["mc20a", "mc20d","mc20e"]
    df_sig = pd.DataFrame()
    for camp in camps:
        df_sig_camp = pd.read_hdf(feature_h5_path + "/StauStau_" + camp + ".h5")
        df_sig_camp = misc.sig_mass_point(df_sig_camp, mass_points = ['100_50'])
        df_sig_camp = misc.stau_selections(df_sig_camp)
        df_sig = pd.concat([df_sig, df_sig_camp], ignore_index=True, axis=0)
    for bkg in bkg_types:
        print(f"loading {bkg} background sample")
        camps = ["mc20a", "mc20d","mc20e"]
        tmp_df_bkg = pd.DataFrame()
        for camp in camps:
            tmp_df_bkg_camp = pd.read_hdf(feature_h5_path + bkg + "_" + camp + ".h5")
            tmp_df_bkg_camp = misc.stau_selections(tmp_df_bkg_camp)
            tmp_df_bkg = pd.concat([tmp_df_bkg, tmp_df_bkg_camp], ignore_index=True, axis=0)
        df_bkg = pd.concat([df_bkg, tmp_df_bkg], ignore_index=True, axis=0)
else:
    df_sig = pd.read_hdf(feature_h5_path + signal + "_" + signal_mass + cutstring + ".h5",
                         key=signal)
    for bkg in bkg_types:
        tmp_df_bkg = pd.read_hdf(feature_h5_path + bkg + cutstring + ".h5", key=bkg)
        df_bkg = pd.concat([df_bkg, tmp_df_bkg], ignore_index=True, axis=0)

### get event weights
if signal == "stau":
    df_sig_wgts = df_sig["scale_factor"]
    df_bkg_wgts = df_bkg["scale_factor"]
else:
    df_sig_wgts = df_sig["eventWeight"]
    df_bkg_wgts = df_bkg["eventWeight"]

df_all = pd.concat([df_sig, df_bkg], axis=0)
df_sig = df_all.iloc[:len(df_sig)]
df_bkg = df_all.iloc[len(df_sig):]
for v, var in enumerate(kinematics):
    print(f"Plotting {var}")
    print("-----> Weighted:")
    plotting.plot_kinematics(df_sig, df_bkg, signal_label, background_label,
                                  var, plot_path, standardised=False, normalise=False,
                                  log_scale=True, sig_wgts=df_sig["eventWeight"],
                                  bkg_wgts=df_bkg["eventWeight"], ex=cutstring)
    print("-----> Normed:")
    plotting.plot_kinematics(df_sig, df_bkg, signal_label, background_label,
                                  var, plot_path, standardised=False, normalise=True,
                                  log_scale=True, ex=cutstring)
    # Standardising kinematics
    print("-----> Standardising + plotting")
    standardised_values = norm.standardise(df_all.loc[:, var])
    df_all.loc[:, var] = standardised_values.astype('float32')  # convert to float32
    df_sig = df_all.iloc[:len(df_sig)]
    df_bkg = df_all.iloc[len(df_sig):]
    plotting.plot_kinematics(df_sig, df_bkg, signal_label, background_label, var, plot_path,
                                  standardised=True, normalise=True, log_scale=True, ex=cutstring)
