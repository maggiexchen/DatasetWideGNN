"""Module to calculate all the event-pair distances for a given metric, signal, variable set"""
import logging
import argparse
import math
import time

import utils.torch_distances as dis
import utils.misc as misc
import utils.plotting as plotting
import utils.adj_mat as adj
import utils.user_config as uconfig

import numpy as np
import torch

logging.getLogger().setLevel(logging.INFO)
torch.manual_seed(42)

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
    "--distance",
    "-d",
    type=str,
    required=True,
    help="Specify the type of distance to calculate",
)

parser.add_argument(
    "--userconfig",
    "-u",
    type=str,
    required=True,
    help="Specify the config for the user.",
)

parser.add_argument(
    "--batchsize",
    "-b",
    type=int,
    default=10000,
    required=False,
    help="",
)

start_time = time.time()

args = parser.parse_args()
user_config_path = str(args.userconfig)
user = uconfig.UserConfig.from_yaml(user_config_path)

variable = str(args.variable)
distance = str(args.distance)

signal_label, background_label = plotting.get_plot_labels(user.signal)
#cutstring = misc.get_cutstring(user.cuts)

logging.info("distance metric: %s", distance)
logging.info("variable set: %s", variable)
kinematics = misc.get_kinematics(variable, user.feature_dim)

# load in input files
logging.info('Importing signal and background files...')

# Standardise variables for distance calc by default.
standardise = True
if distance == "emd":
    # but don't standardise the kinematic variables for emd!
    if "LowLevel" not in variable:
        raise Exception("need to use low level variables for the EMD")
    standardise = False

full_sig, full_bkg, full_x, sig_wgt, bkg_wgt, sig_labels, bkg_labels, _, _ = \
    adj.data_loader(user.feature_h5_path, kinematics, ex=user.cutstring, signal=user.signal,
                    signal_mass=user.signal_mass, standardisation=standardise)


# option to weight events, atm only relevant for HHH.
global_bkg_wgt = 1.0
if user.signal == "hhh":
    SF_4b5b = 0.07 # placeholder value for HHH data-driven background
    global_bkg_wgt = SF_4b5b
bkg_wgt = bkg_wgt*global_bkg_wgt

batch_size = args.batchsize

save_path = user.dist_path + "/batched_" + str(batch_size) + "_" +\
    variable + "_" + distance + user.cutstring + "_distances/"
misc.create_dirs(save_path)

def calc_a_b_batched_distances(species_a, species_b, full_a, full_b, kinematics_list,
                               a_wgt, b_wgt, batchsize, subset_frac=0.01):
    """
    Function to calculate all the pair-wise distances between events of type a and b, in batches.
    pair-wise weights are also calculated as the product of the 2 event weights.
    A subset of the distances are saved/returned to be plotted.

    Args:
        species_a (str): "sig" or "bkg"
        species_b (str): "sig" or "bkg"
        full_a (torch.tensor): full set of species a events and the standardised kinematics needed.
        full_b (torch.tensor): full set of species b events and the standardised kinematics needed.
        a_wgt (torch.tensor): full set of species a event weights.
        b_wgt (torch.tensor): full set of species b event weights.
        batchsize: how many events per batch.
        subset_frac: what fraction of the distances to sample for the plotting. Default 10%
    Returns:
        (torch.tensor(float32)) flattened distance subset
        (torch.tensor(float32)) flattened weight subset
    """
    logging.info("Calculating %s %s distances, and subsample for plotting.", species_a, species_b)

    distance_subsample = torch.empty(0, dtype=torch.float32)
    wgt_subsample = torch.empty(0, dtype=torch.float32)

    # find the numbers of events and batches we have
    num_a_events = full_a.shape[0]
    num_a_batches = math.ceil(num_a_events/batchsize)
    num_b_events = full_b.shape[0]
    num_b_batches = math.ceil(num_b_events/batchsize)
    logging.info("%s %s events", str(num_a_events), species_a)
    logging.info("%s %s events", str(num_b_events), species_b)
    logging.info("%s %s batches", str(num_a_batches), species_a)
    logging.info("%s %s batches", str(num_b_batches), species_b)

    if distance=="emd":
        kinematics_indices = dis.get_emd_kinematics_key(kinematics_list, signal="LQ")

    for i in range(num_a_batches):
        # find the event index range for species a
        start_idx_a = i * batchsize
        end_idx_a = min((i + 1) * batchsize, num_a_events)
        batch_a = full_a[start_idx_a:end_idx_a]
        batch_a_wgt = a_wgt[start_idx_a:end_idx_a]

        if distance == "emd":
            batch_a = dis.get_event_vectors(batch_a, kinematics_indices)

        for j in range(num_b_batches):
            # ignore redundant elements from symmetry
            if ((i < j) and (species_a == species_b)):
                continue
            logging.info("%s %s files %s %s", species_a, species_b, str(i), str(j))

            # find the event index range for species b
            start_idx_b = j * batchsize
            end_idx_b = min((j + 1) * batchsize, num_b_events)
            batch_b = full_b[start_idx_b:end_idx_b]
            batch_b_wgt = b_wgt[start_idx_b:end_idx_b]

            # calculate the distances for this batch vectorially.

            if distance == "emd":
                batch_b = dis.get_event_vectors(batch_b, kinematics_indices)

            batch_ab = dis.distance_calc(batch_a, batch_b, distance)
            batch_ab_wgt = torch.ger(batch_a_wgt, batch_b_wgt)
            logging.debug("done distances")
            del batch_b, batch_b_wgt

            # save all the wgts and distances for this batch to a torch tensor .pt file.
            batch_dict = {'distance': batch_ab, 'weight': batch_ab_wgt}
            torch.save(batch_dict, save_path +
                       f'{species_a}{species_b}_distances_batch_{i:02d}_{j:02d}.pt')
            del batch_dict
            logging.debug("saved file")

            # flatten the distance/weight tensors;
            # take a uniform subset of them for later plotting.
            # Uniform is chosen to ensure we get a fair sample
            #   from all the MET slices and background types.
            flat_distances = torch.flatten(batch_ab).to(torch.float32)
            del batch_ab
            flat_wgts = torch.flatten(batch_ab_wgt).to(torch.float32)
            del batch_ab_wgt
            logging.debug("got flat things")
            subset_indices = np.linspace(0,
                                         flat_distances.shape[0]-1,
                                         int(subset_frac*flat_distances.shape[0]),
                                         dtype=int)
            distance_subsample = torch.cat((distance_subsample, flat_distances[subset_indices]))
            del flat_distances
            wgt_subsample = torch.cat((wgt_subsample, flat_wgts[subset_indices]))
            del flat_wgts
            logging.debug("made subsets")

        del batch_a, batch_a_wgt

    return distance_subsample, wgt_subsample

# Calculate the pair-wise distances for each category in batches;
#  save them to a torch tensor file for each batch.
#  Also take a subset of 10% of the values and return as a tensor to plot.
logging.info("Calculating batched distances ... ")
sigsig_distance_subsample, sigsig_wgt_subsample = \
    calc_a_b_batched_distances("sig", "sig", full_sig, full_sig, kinematics,
                               sig_wgt, sig_wgt, batch_size)
sigbkg_distance_subsample, sigbkg_wgt_subsample = \
    calc_a_b_batched_distances("sig", "bkg", full_sig, full_bkg, kinematics,
                               sig_wgt, bkg_wgt, batch_size)
del full_sig, sig_wgt
bkgbkg_distance_subsample, bkgbkg_wgt_subsample = \
    calc_a_b_batched_distances("bkg", "bkg", full_bkg, full_bkg, kinematics,
                               bkg_wgt, bkg_wgt, batch_size)
del full_bkg, bkg_wgt

# Plot the distance distributions for the subset.
logging.info("Plotting ... ")
plot_path = user.plot_path + "/" + variable + "/"
misc.create_dirs(plot_path)
plotting.plot_distances(sigsig_distance_subsample.numpy(),
                        sigbkg_distance_subsample.numpy(),
                        bkgbkg_distance_subsample.numpy(),
                        sigsig_wgt_subsample.numpy(),
                        sigbkg_wgt_subsample.numpy(),
                        bkgbkg_wgt_subsample.numpy(),
                        variable, distance,
                        signal_label, background_label,
                        plot_path)

logging.info("--- %s seconds ---", (time.time() - start_time))
