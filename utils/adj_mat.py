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

def data_loader(file_path, f_type, kinematics):
    df_sig =  pd.read_hdf(file_path+"/split_files/sig_"+str(f_type)+".h5", key="sig_"+str(f_type))
    df_bkg =  pd.read_hdf(file_path+"/split_files/bkg_"+str(f_type)+".h5", key="bkg_"+str(f_type))
    df_sig = df_sig.sample(n=1000)
    df_bkg = df_bkg.sample(n=1000)
    df_sig_wgts = df_sig["eventWeight"]
    df_bkg_wgts = df_bkg["eventWeight"]
    df_sig = df_sig[kinematics]
    df_bkg = df_bkg[kinematics]
    sig_label = [1]*len(df_sig)
    bkg_label = [0]*len(df_bkg)
    # MAD scaling
    for var in kinematics:
        df_sig.loc[:, var], df_bkg.loc[:, var] = norm.MAD_norm(df_sig.loc[:, var], df_bkg.loc[:, var])
        plot.plot_kinematic_hists(df_sig, df_bkg, var, file_path)
    # convert pd dataframes to torch tensors
    torch_sig = torch.tensor(df_sig.values, dtype=torch.float32)
    torch_bkg = torch.tensor(df_bkg.values, dtype=torch.float32)
    torch_sig_wgts = torch.tensor(df_sig_wgts.values, dtype=torch.float32)
    torch_bkg_wgts = torch.tensor(df_bkg_wgts.values, dtype=torch.float32)
    # concatenating signal and background events / weights
    torch_all = torch.concat((torch_sig, torch_bkg), dim=0)
    torch_wgts = torch.concat((torch_sig_wgts, torch_bkg_wgts), dim=0)
    truth_labels = torch.tensor(numpy.concatenate((sig_label, bkg_label)), dtype=torch.float32)

    return torch_sig, torch_bkg, torch_all, torch_wgts, truth_labels

def create_adj_mat(a, length):
    return (a < length).float()

def create_node_wgts(a, b):
    a_col = a.view(-1,1)
    b_col = b.view(1,-1)
    outer = torch.matmul(a_col, b_col)
    return torch.transpose(outer, 0, 1)

def generate_adj_mat(x, x_wgts, dis_type, linking_length):
    # initialise adjacency matrix
    adj_mat = torch.empty((0, len(x)))
    node_wgts = torch.empty((0, len(x_wgts)))

    # calculate distances
    if dis_type == "euclidean":
        distance_subset = dis.euclidean(x, x)
    elif dis_type == "cityblock":
        distance_subset = dis.cityblock(x, x)
    elif dis_type == "cosine":
        distance_subset = dis.cosine(x, x)
    adj_mat_subset = create_adj_mat(distance_subset, linking_length)
    adj_mat = torch.concat((adj_mat_subset, adj_mat), dim=0)
        
    print(f"Time taken for adjacency matrix generation: {time.time() - st}")
    return adj_mat
