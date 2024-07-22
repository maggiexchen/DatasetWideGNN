import numpy as np
import matplotlib.pyplot as plt
import math
import utils.misc as misc
import utils.plotting as plotting
import logging
import argparse



def GetParser():
    """Argument parser for reading Ntuples script."""
    parser = argparse.ArgumentParser(
        description="Reading Ntuples command line options."
    )
    parser.add_argument(
        "--MLconfig",
        "-c",
        type=str,
        required=True,
        help="Specify the config file for training",
    )

    parser.add_argument(
        "--userconfig",
        "-u",
        type=str,
        required=True,
        help="Specify the config for the user e.g. paths to store all the input/output data and results, signal model to look at",
    )

    return parser

parser = GetParser()
args = parser.parse_args()


### load user config
user_config_path = args.userconfig
user_config = misc.load_config(user_config_path)
print("Using user config ",user_config)
signal = user_config["signal"]
score_path = user_config["score_path"]

### load training config 
train_config_path = args.MLconfig
train_config = misc.load_config(train_config_path)
print("Using training config ",train_config)
hidden_sizes_gcn = train_config["hidden_sizes_gcn"]
hidden_sizes_mlp = train_config["hidden_sizes_mlp"]
dropout_rates = train_config["dropout_rates"]
batch_size = train_config["batch_size"]
epochs = train_config["epochs"]
gnn_type = train_config["gnn_type"]
num_nb_list = train_config["num_nb_list"]
linking_length = train_config["linking_length"]
eff = train_config["sigsig_eff"]
LR = train_config["LR"]


### check if linking length is given in the config, return ll or eff string
if linking_length is None:
    if eff is None:
        raise Exception("Need to specify a sig-sig efficiency for the adjacency matrix when training a gcn in the config")
    elif eff not in [0.6, 0.7, 0.8, 0.9]:
        raise Exception("not given a supported efficiency, (0.6, 0.7, 0.8, 0.9)")
    else:
        ll_str = "_LLEff" + str(eff).replace(".", "p")
else:
    eff = None
    print("linking length is given in config, IGNORING the sigsig_eff in the config!")
    ll_str = "_LL" + str(linking_length).replace(".", "p")

### create model label and result plot path
if len(hidden_sizes_gcn) == 0:
    model_label = signal\
          + "_MLP" + "-".join(map(str, hidden_sizes_mlp)).replace(".", "p")\
          + "_lr" + str(LR).replace(".", "p")\
          + "_dr" + "-".join(map(str, dropout_rates)).replace(".", "p")\
          + "_bs" + str(batch_size)\
          + "_e" + str(epochs)
else:
    model_label = signal\
            + f"_{gnn_type}" + "-".join(map(str, hidden_sizes_gcn)).replace(".", "p")\
            + "_MLP" + "-".join(map(str, hidden_sizes_mlp)).replace(".", "p")\
            + "_nb" + "-".join(map(str, num_nb_list))\
            + "_lr" + str(LR).replace(".", "p")\
            + ll_str\
            + "_dr" + "-".join(map(str, dropout_rates)).replace(".", "p")\
            + "_bs" + str(batch_size)\
            + "_e" + str(epochs)   
    

### load the model
score_path = score_path + model_label + "/"
val_sig_pred = np.load(score_path + "val_sig_pred.npy")
val_sig_wgts = np.load(score_path + "val_sig_wgts.npy")
val_bkg_pred = np.load(score_path + "val_bkg_pred.npy")
val_bkg_wgts = np.load(score_path + "val_bkg_wgts.npy")

val_to_tot_wgt = 5
val_sig_wgts = val_sig_wgts*val_to_tot_wgt
val_bkg_wgts = val_bkg_wgts*val_to_tot_wgt
val_sig_wgts = np.ones(len(val_sig_pred))
val_bkg_wgts = np.ones(len(val_bkg_pred))



logging.info("Plotting model outputs ...")
fig, axs = plt.subplots(2,1,#figsize=(8,10),
                        gridspec_kw={'height_ratios': [4, 1]},
                        sharex=True)
binning = np.linspace(0,1,51)
axs[0].hist(val_sig_pred, bins=binning, label="Signal (validation)", alpha=0.5, density=False, color="darkorange", weights=val_sig_wgts)
axs[0].hist(val_bkg_pred, bins=binning, label="Background (validation)", alpha=0.5, density=False, color="steelblue", weights=val_bkg_wgts)

# plotting.add_text(axs[0], text, doATLAS=False, startx=0.02, starty=0.95)
axs[0].legend(loc='upper right', fontsize=9)
# axs[0].set_xlabel("Output score", loc="right")
axs[0].set_ylabel("No. Events", loc="top")
ymin, ymax = axs[0].get_ylim()
axs[0].set_yscale('log')
axs[0].set_ylim((ymin, ymax*10))


def get_Z(n, b, sigma):
    return (n - b) / np.sqrt(b + (sigma**2))

Z_val = []
Z2_val = []
Z3_val = []
for x in binning:

    sig_tot = val_sig_wgts[np.where(val_sig_pred > x)].sum()
    bkg_tot = val_bkg_wgts[np.where(val_bkg_pred > x)].sum()
    n = sig_tot + bkg_tot
    sigma = .15*bkg_tot
    sigma2 = .3*bkg_tot
    sigma3 = .5*bkg_tot

    Z = get_Z(n, bkg_tot, sigma)
    Z2 = get_Z(n, bkg_tot, sigma2)
    Z3 = get_Z(n, bkg_tot, sigma3)

    Z_val.append(Z)
    Z2_val.append(Z2)
    Z3_val.append(Z3)

up_x = 3
axs[1].set_ylim(0, up_x)
for i in range(1, up_x):
    axs[1].axhline(y=i, linestyle='dotted', color='black')

# axs[1].set_xlabel('$D_{\mathrm{DNN}}$', fontsize = fs)
axs[1].set_xlabel("Output score", loc="right")
axs[1].set_ylabel('Z', loc = 'top')
axs[1].plot(binning, Z_val, label = r"$\sigma_b$ = 15%")
axs[1].plot(binning, Z2_val, label = r"$\sigma_b$ = 30%")
axs[1].plot(binning, Z3_val, label = r"$\sigma_b$ = 50%")
axs[1].legend()
fig.subplots_adjust(hspace=0.1)
print(np.nanmax(Z_val), np.nanmax(Z2_val))
plt.tight_layout()
plt.show()
