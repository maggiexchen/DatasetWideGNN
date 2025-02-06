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
import utils.plotting as plotting
import re
import pdb
import glob
import logging
logging.getLogger().setLevel(logging.INFO)

import psutil
process = psutil.Process()

cpu = torch.device('cpu')
device = torch.device('cuda:0')


def data_loader(h5_path, plot_path, kinematics, ex="", plot=False, signal="LQ", standardisation=False):
    """
    Function to load our sign and bkg data into pandas dataframes

    Args:
        h5_path (str): path for input file directory
        plot_path (str): path for output plot directory
        kinematics (list(str)): list of kinematic variables to load as dataframe columns
        ex (str): input file extension, default is empty
        plot (bool): flag to plot raw and standardised kinematics, default is False
        signal (str): type of signal, to determine the bkgs present, default is LQ
        standardisation (bool): flag to do standardisation
    Returns:
        (torch.tensor(float32)): signal events/kinematics tensor
        (torch.tensor(float32)): background events/kinematics tensor
        (torch.tensor(float32)): all events/kinematics tensor
        (torch.tensor(float32)): signal event weight tensor
        (torch.tensor(float32)): background event weight tensor
        (torce.tensor(float32)): all event truth labels ie. 1 for sig, 0 for bkg
    """
    signal_label, background_label = plotting.get_plot_labels(signal)
    bkg_types = misc.get_background_types(signal)
    print("Loading for kinematics: ", kinematics)
    
    if len(ex) > 0:
        ex = "_"+ex
    if signal == "stau":
        logging.info("Loading stau signal sample(s) ...")
        camps = ["mc20a", "mc20d","mc20e"]
        df_sig = pd.DataFrame()
        for camp in camps:
            df_sig_camp = pd.read_hdf(h5_path+"/StauStau_"+camp+str(ex)+".h5")
            df_sig_camp = misc.sig_mass_point(df_sig_camp, mass_points = ['100_50'])
            df_sig_camp = misc.stau_selections(df_sig_camp)
            df_sig = pd.concat([df_sig, df_sig_camp], ignore_index=True, axis=0)
    else:
        df_sig =  pd.read_hdf(h5_path+str(signal)+str(ex)+".h5", key=str(signal)+str(ex))
        # df_sig =  pd.read_hdf(h5_path+str(signal)+str(ex)+".h5", key="sig"+str(ex))
    
    df_bkg = pd.DataFrame()
    if signal == "stau":
        for bkg in bkg_types:
            print(f"loading {bkg} background sample for {ex}")
            camps = ["mc20a", "mc20d","mc20e"]
            tmp_df_bkg = pd.DataFrame()
            for camp in camps:
                tmp_df_bkg_camp = pd.read_hdf(h5_path+bkg+"_"+camp+str(ex)+".h5")
                tmp_df_bkg_camp = misc.stau_selections(tmp_df_bkg_camp)
                tmp_df_bkg = pd.concat([tmp_df_bkg, tmp_df_bkg_camp], ignore_index=True, axis=0)
            df_bkg = pd.concat([df_bkg, tmp_df_bkg], ignore_index=True, axis=0)
            
    else:
        for bkg in bkg_types:
            tmp_df_bkg = pd.read_hdf(h5_path+bkg+str(ex)+".h5", key=bkg+str(ex))
            df_bkg = pd.concat([df_bkg, tmp_df_bkg], ignore_index=True, axis=0)

    ### get event weights
    if signal == "stau":
        df_sig_wgts = df_sig["scale_factor"]
        df_bkg_wgts = df_bkg["scale_factor"]
    else:
        df_sig_wgts = df_sig["eventWeight"]
        df_bkg_wgts = df_bkg["eventWeight"]

    df_sig = df_sig[kinematics]
    df_bkg = df_bkg[kinematics]
    # set truth labels for is signal
    sig_label = [1]*len(df_sig)
    bkg_label = [0]*len(df_bkg)
    if standardisation:
        # Standardising kinematics
        df_all = pd.concat([df_sig, df_bkg], axis=0)
        for var in kinematics:
            if plot:
                df_sig = df_all.iloc[:len(df_sig)]
                df_bkg = df_all.iloc[len(df_sig):]
                plotting.plot_kinematic_hists(df_sig, df_bkg, signal_label, background_label, var, plot_path, standardise=False)
            print(f"-----> Standardising {var}:")
            standardised_values = norm.standardise(df_all.loc[:, var])
            df_all.loc[:, var] = standardised_values.astype('float32')  # convert to float32
            df_sig = df_all.iloc[:len(df_sig)]
            df_bkg = df_all.iloc[len(df_sig):]
            if plot:
                plotting.plot_kinematic_hists(df_sig, df_bkg, signal_label, background_label, var, plot_path, standardise=True)
            
    if plot:
        for var in kinematics:
            plotting.plot_kinematic_hists(df_sig, df_bkg, signal_label, background_label, var, plot_path, standardise=False)\

    # convert pd dataframes to torch tensors
    torch_sig = torch.tensor(df_sig.values, dtype=torch.float32).to(cpu)
    torch_bkg = torch.tensor(df_bkg.values, dtype=torch.float32).to(cpu)
    torch_sig_wgts = torch.tensor(df_sig_wgts.values, dtype=torch.float32).to(cpu)
    torch_bkg_wgts = torch.tensor(df_bkg_wgts.values, dtype=torch.float32).to(cpu)
    # concatenating signal and background events
    torch_all = torch.concat((torch_sig, torch_bkg), dim=0)

    return torch_sig, torch_bkg, torch_all, torch_sig_wgts, torch_bkg_wgts, torch.tensor(sig_label).to(cpu), torch.tensor(bkg_label).to(cpu)


def create_adj_mat(a, length):
    """
    Function to filter a matrix of distances into a binary adjacency matrix

    Args:
        a (pytorch.tensor(float32)): matrix to filter
        length (float): linking length

    Returns:
        (float): cityblock distance
    """
    return (a <= length).int()


def create_node_wgts(a, b):
    """
    Function to create ....

    Args:
        a (torch.tensor(*)): first tensor
        b (torch.tensor(*)): second tensor
    Returns:
        (torch.tensor(*)): weight tensor
    """
    a_col = a.view(-1,1)
    b_col = b.view(1,-1)
    outer = torch.matmul(a_col, b_col)
    return torch.transpose(outer, 0, 1)


def generate_adj_mat(x, x_wgts, distance, linking_length):
    """
    Function create a binary adjacency matrix

    Args:
        x (pytorch.tensor(float32)): matrix of events and kinematics
        x_wgts (pytorch.tensor(float32)): matrix of event weights
        distance (str): distance metric to use
        linking_length (float): linking length

    Returns:
        (pytorch.tensor) adjacency matrix
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


def generate_adj_mat_from_batch(distance, linking_length):
    """
    Function create a binary adjacency matrix from pre-calculated distances

    Args:
        distance (torch.tensor(float32)): pair-wise distances between events
        linking_length (float): linking length

    Returns:
        (pytorch.tensor) adjacency matrix
    """
    adj_mat = create_adj_mat(distance, linking_length)
    return adj_mat


def generate_batched_nonzero_ind(dist_path, variable, distance, t, linking_length, flip=False, edge_wgt=False):
    """
    Function that loads in the distances in batches, within each batch, apply the linking length,
    and returns non-zero indices within that batch

    Args:
        dist_path (str): path to the saved batched distance files
        variable (str): kinematic variable type in the file names
        distance (str): distance metric type in the file names
        t (str): type of distance (sigsig, sigbkg or bkgbkg)
        linking length (float): chosen linking length to apply
        flip (bool): True if edges are made for distance < linking length
    
    Returns:
        (torch.tensor())indices of non-empty elements in the adj matrix
    """
    # Load in files in batches (sigsig, sigbkg, or bkgbkg) by the i and j indices
    dist_dir = dist_path+"/batched_"+variable +"_"+distance+"_distances/"
    files = sorted(glob.glob(dist_dir + t + '*.pt'))
    print(len(files), " files found for "+t+" distances")
    # apply linking length within each batch, and pull out non-zero indices
    indices = torch.empty(0, dtype=torch.int32)#.to("cuda")
    if edge_wgt:
        edge_wgts = torch.empty(0, dtype=torch.float32)
        inf_mask = torch.empty(0, dtype=torch.bool)
    batch_size = 30000
    print("Loading batch size ", batch_size)
    for f in files:
        # get the i and j batch numbers
        # use -1 and -2 here to count from back - to account for possible numbers earlier in the file path (e.g. atlas3)
        i_ind = int(re.findall(r'\d+', f)[-2])
        j_ind = int(re.findall(r'\d+', f)[-1])
        if t=="sigsig" or t=="bkgbkg":
            if i_ind >= j_ind:
                print("File ", i_ind, j_ind)
                distance = torch.load(f)["distance"]# .to("cuda")

                # apply the linking length to the distances in that batch
                if flip:
                    mask = distance <= linking_length
                else:
                    mask = distance >= linking_length
                ind = mask.nonzero().to(torch.int32)
                if edge_wgt:
                    edge_wgts_tmp = (1 / distance[mask])
                    inf_mask_tmp = torch.isinf(edge_wgts_tmp)
                    edge_wgts_tmp[inf_mask_tmp] = 0.
                print(f"CPU allocated before {process.memory_info().rss/(1024 ** 3):.2f}, GB")
                del distance, mask
                
                # add to the row and column indices according to the i and j indices of that file (this hurts my brain)
                ind[:,0] += i_ind*batch_size
                ind[:,1] += j_ind*batch_size

                indices = torch.cat((indices, ind))
                if edge_wgt:
                    edge_wgts = torch.cat((edge_wgts, edge_wgts_tmp))
                    inf_mask = torch.cat((inf_mask, inf_mask_tmp))
                if i_ind != j_ind:
                    # swapping the columns for the opposite corner of a symmetric adjacency matrix 
                    ind_lowerleft = ind[:, torch.tensor([0, 1])][:, torch.tensor([1, 0])]
                    indices = torch.cat((indices, ind_lowerleft))
                    del ind_lowerleft
                    if edge_wgt:
                        edge_wgts = torch.cat((edge_wgts, edge_wgts_tmp))
                        inf_mask = torch.cat((inf_mask, inf_mask_tmp))
                del ind
                if edge_wgt:
                    del edge_wgts_tmp
                print(f"CPU allocated after {process.memory_info().rss/(1024 ** 3):.2f}, GB")
        
        else:
            print("File ", i_ind, j_ind)
            distance = torch.load(f)["distance"]# .to("cuda")

            # apply the linking length to the distances in that batch
            if flip:
                mask = distance <= linking_length
            else:
                mask = distance >= linking_length
            ind = mask.nonzero().to(torch.int32)
            if edge_wgt:
                edge_wgts_tmp = 1 / distance[mask]
                inf_mask_tmp = torch.isinf(edge_wgts_tmp)
                edge_wgts_tmp[inf_mask_tmp] = 0.
                inf_mask = torch.cat((inf_mask, inf_mask_tmp))

            print(f"CPU allocated before {process.memory_info().rss/(1024 ** 3):.2f}, GB")
            del distance, mask

            # add to the row and column indices according to the i and j indices of that file (this hurts my brain)
            ind[:,0] += i_ind*batch_size
            ind[:,1] += j_ind*batch_size
            indices = torch.cat((indices, ind))
            if edge_wgt:
                edge_wgts = torch.cat((edge_wgts, edge_wgts_tmp))
                del edge_wgts_tmp
            del ind
            print(f"CPU allocated after {process.memory_info().rss/(1024 ** 3):.2f}, GB")
    if edge_wgt:
        edge_wgts = norm.minmax(edge_wgts, torch.min(edge_wgts), torch.max(edge_wgts))
        edge_wgts[inf_mask] = 1.
        return indices, edge_wgts
    else:
        return indices


def generate_sparse_adj_mat(sigsig, sigbkg, bkgsig, bkgbkg, N):
    """
    Function to generator the adjacency matrix (and the correspondingly formatted indices) as a torch.sparse_csr_tensor, from the sets of non-zero rows/columns.

    Args:
        sigsig (torch.tensor()): indices of sigsig distances that have passed the linking length requirement
        sigbkg (torch.tensor()): indices of sigbkg distances that have passed the linking length requirement
        bkgsig (torch.tensor()): indices of bkgsig distances that have passed the linking length requirement
        bkgbkg (torch.tensor()): indices of bkgbkg distances that have passed the linking length requirement
        N (int): the length of signal+background in the final full adjacency matrix (N x N)
    Returns:
        (torch.sparse_csr_tensor(float32)): adjacency matrix
        (torch.tensor(int32)): ordered list of non-empty row indices
        (torch.tensor(int32)): compressed row format for non-empty row indices
        (torch.tensor(int32)): ordered list of non-empty column indices        
        (torch.tensor(float32)): ordered list of values for non-empty cells in the adj 
    """

    torch.set_printoptions(threshold = 10000)
    full_ind_unsorted = torch.cat((sigsig, sigbkg, bkgsig, bkgbkg)).round().to(torch.int32)

    # order the rows/cols to be in ascending row order.
    logging.info("Sorting indices ...")
    tmp = full_ind_unsorted[full_ind_unsorted[:,1].sort()[1]]
    full_ind = tmp[tmp[:,0].sort()[1]]
    full_ind_shape = full_ind.shape[0]
    del tmp
    
    logging.info("Getting rows and columns ...")
    row_ind = full_ind[:,0]
    col_ind = full_ind[:,1].contiguous()
    del full_ind
    
    logging.info("writing sparse adj mat, row and column indices")
    csr_count = row_ind.bincount(minlength=N)
    csr_row = torch.cat([torch.tensor([0], dtype=torch.int32), csr_count.cumsum(dim=0, dtype=torch.int32)])
    sparse_adj_mat = torch.sparse_csr_tensor(csr_row, col_ind, torch.ones(full_ind_shape), (N, N), dtype=torch.float16)
    crow_ind = sparse_adj_mat.crow_indices()

    for i,x in enumerate(crow_ind):
      if x > x+1: print("ARGH: ",x," is larger than",x+1)
    cols_ind = sparse_adj_mat.col_indices()
    values = sparse_adj_mat.values()

    return sparse_adj_mat, row_ind, crow_ind, cols_ind, values

