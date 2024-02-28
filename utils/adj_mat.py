import numpy
import pandas as pd
import math
import torch
import time
st = time.time()
import matplotlib.pyplot as plt
import mplhep as hep
import utils.normalisation as norm
import utils.torch_distances as dis
import utils.misc as misc
import utils.plotting as plot

def data_loader(path, f_type, kinematics, n_sig=1000, n_bkg=1000, norm_kin=True):
    """
    Function to load our sign and bkg data into pandas dataframes

    Args:
        path (str): base path for input file directory
        f_type (str): input file extension
        kinematics (list(str)): list of kinematic variables to load as dataframe columns
        n_sig (int): number of sig events to load
        n_bkg (int): number of bkg events to load

    Returns:
        (float) cityblock distance
    """

    df_sig =  pd.read_hdf(path+"/hhh_split_files/sig_"+str(f_type)+".h5", key="sig_"+str(f_type))
    df_bkg =  pd.read_hdf(path+"/hhh_split_files/bkg_"+str(f_type)+".h5", key="bkg_"+str(f_type))
    df_sig = df_sig.sample(n=n_sig, random_state=42)
    df_bkg = df_bkg.sample(n=n_bkg, random_state=42)
    df_sig_wgts = df_sig["eventWeight"]
    df_bkg_wgts = df_bkg["eventWeight"]
    df_sig = df_sig[kinematics]
    df_bkg = df_bkg[kinematics]
    df_all = pd.concat([df_sig, df_bkg], axis=0)
    # set truth labels for is signal
    sig_label = [1]*len(df_sig)
    bkg_label = [0]*len(df_bkg)
    # Standardising kinematics
    for var in kinematics:
        if norm_kin:
            df_all.loc[:, var] = norm.standardise(df_all.loc[:, var])
        df_sig = df_all.iloc[:len(df_sig)]
        df_bkg = df_all.iloc[len(df_sig):]
        plot.plot_kinematic_hists(df_sig, df_bkg, var, path, standardise=norm_kin)
    # convert pd dataframes to torch tensors
    torch_sig = torch.tensor(df_sig.values, dtype=torch.float32)
    torch_bkg = torch.tensor(df_bkg.values, dtype=torch.float32)
    torch_sig_wgts = torch.tensor(df_sig_wgts.values, dtype=torch.float32)
    torch_bkg_wgts = torch.tensor(df_bkg_wgts.values, dtype=torch.float32)
    # concatenating signal and background events / weights
    torch_all = torch.concat((torch_sig, torch_bkg), dim=0)
    truth_labels = torch.tensor(numpy.concatenate((sig_label, bkg_label)), dtype=torch.float32)

    return torch_sig, torch_bkg, torch_all, torch_sig_wgts, torch_bkg_wgts, truth_labels

def create_adj_mat(a, length):
    """
    Function to filter a matrix of distances into a binary adjacency matrix

    Args:
        a (pytorch.tensor): matrix to filter
        length (float): linking length

    Returns:
        (float) cityblock distance
    """
    return (a < length).float()

def create_node_wgts(a, b):
    a_col = a.view(-1,1)
    b_col = b.view(1,-1)
    outer = torch.matmul(a_col, b_col)
    return torch.transpose(outer, 0, 1)

def generate_adj_mat(x, x_wgts, distance, linking_length):
    """
    Function create a binary adjacency matrix

    Args:
        x (pytorch.tensor): matrix of events and kinematics
        x_wgts (pytorch.tensor): matrix of event weights
        distance (str): distance metric to use
        linking_length (float): linking length

    Returns:
        (float) cityblock distance
    """
    # initialise adjacency matrix
    adj_mat = torch.empty((0, len(x)))
    node_wgts = torch.empty((0, len(x_wgts)))

    # calculate distances
    if distance == "euclidean":
        distance_matrix = dis.euclidean(x, x)
    elif distance == "cityblock":
        distance_matrix = dis.cityblock(x, x)
    elif distance == "cosine":
        distance_matrix = dis.cosine(x, x)
    else:
        raise Exception("not given a supported distance metric")

    adj_mat = create_adj_mat(distance_matrix, linking_length)
    print(f"Time taken for adjacency matrix generation: {time.time() - st}")
    return adj_mat
