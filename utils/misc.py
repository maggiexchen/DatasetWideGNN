"""Miscellaneous utils functions"""
import os
import glob
import hashlib

import yaml
import torch
#import math
import pandas as pd
#import pdb
import uproot
import numpy as np

torch.manual_seed(42)
cpu = torch.device('cpu')
device = torch.device('cuda:0')

def get_cutstring(cuts):
    """
    Function to get a str summarising the cuts to use in filenames/book-keeping.

    Args:
        cuts (dict): form {'variable': {'threshold': number, 'operation': '>'}}

    Returns:
        (str) cutstring
    """
    cutstring = ""
    for var,cut in cuts.items():
        if not "threshold" in cut.keys():
            raise KeyError("no threshold for {var}, check your cut dictionary...")
        cutstring = cutstring + "_" + var + str(cut["threshold"]).replace(".","p")
    return cutstring

def print_mem_info():
    """
    Function to calculate and print out the total memory available and used on CUDA.
    """
    unit_gb = 1024**3
    t = torch.cuda.get_device_properties(0).total_memory/unit_gb
    r = torch.cuda.memory_reserved(0)/unit_gb
    a = torch.cuda.memory_allocated(0)/unit_gb
    f = r-a  # free inside reserved
    print("total [Gb]: ", t, "reserved [Gb]: ", r, "allocated [Gb]:", a, "free [Gb]: ", f)

def create_dirs(path):
    """
    Function to create any directories needed to store the output of a function.

    Args:
        path (str): the path of directories you want to exist

    Returns:
        void
    """
    base = path.split("/")[0]
    dirs = path.split("/")[1:]
    tmp_dir = ""
    for folder in dirs:
        tmp_dir = tmp_dir + "/" + folder
        if not os.path.isdir(base + "/" + tmp_dir):
            try:
                os.mkdir(base + "/" + tmp_dir)
                print("creating: ", base + "/" + tmp_dir)
            except OSError as error:
                print(error)

    return 0

def get_background_types(signal_type):
    """
    Function to obtain the list of background types to use for a given signal type.
    Throws an exception for an unknown category

    Args:
        signal_type (str): the category of signal you want fetch the corresponding backgrounds for

    Returns:
        list(str): the list of background types
    """
    if signal_type == "hhh":
        background_type = ["bkg"]
    elif signal_type == "LQ":
        background_type = ["singletop", "ttbar"]
    elif signal_type == "stau":
        background_type = ['Wjets',
                           'Zlljets',
                           'Zttjets2214',
                           'diboson0L','diboson1L','diboson2L',
                           'diboson3L','diboson4L','triboson',
                           'higgs',
                           'singletop','topOther','ttV','ttbar_incl']
    else:
        raise ValueError("Signal type is either hhh, LQ or stau")
    return background_type


def get_kinematics(variable, dim=None):
    """
    Function to obtain the list of names of kinematic variables to use for a given category.
    Throws an exception for an unknown category

    Args:
        variable (str): the category of variables you want the list for
        dim (int): dimension for embedding features - not needed otherwise.

    Returns:
        lists(str): a list of kinematic variables
    """
    if variable == "mass":
        kinematics = ["mH1","mH2","mH3","mHHH"]
    elif variable == "angular":
        # angular kinematics
        kinematics = ["dRH1","dRH2","dRH3","meandRBB"]
    elif variable == "shape":
        # event shape kinematics
        kinematics = ["sphere3dv2b","sphere3dv2btrans",
                      "aplan3dv2b","theta3dv2b"]
    elif variable == "combined":
        kinematics = ["mH1","mH2","mH3","mHHH",
                      "dRH1","dRH2","dRH3","meandRBB",
                      "sphere3dv2b","sphere3dv2btrans",
                      "aplan3dv2b","theta3dv2b"]
    elif variable == "mass_and_angular":
        kinematics = ["mH1","mH2","mH3","mHHH",
                      "dRH1","dRH2","dRH3","meandRBB"]
    elif variable == "mass_and_shape":
        kinematics = ["mH1","mH2","mH3","mHHH",
                      "sphere3dv2b","sphere3dv2btrans",
                      "aplan3dv2b","theta3dv2b"]
    elif variable == "LQ_HighLevel":
        kinematics = ['met', 'sumptllbb', 'mindPhiMETl',  'mtl1', 'mtl2']
    elif variable == "LQ_LowLevel":
        kinematics = ['bjet1pt', 'bjet2pt', 'lep1pt', 'lep2pt',
                      'bjet1eta', 'bjet2eta', 'lep1eta', 'lep2eta',
                      'bjet1phi', 'bjet2phi', 'lep1phi', 'lep2phi',
                      'met', 'metphi']
    elif variable == "LQ-All":
        kinematics = ['xsec', 'genWeight', 'njets', 'nbjets',
                      'bjet1eta', 'bjet1phi', 'bjet1pt', 'bjet2eta', 'bjet2phi', 'bjet2pt',
                      'lep1eta', 'lep1phi', 'lep1pt', 'lep2eta', 'lep2phi', 'lep2pt',
                      'met', 'metphi', 'metsigHt', 'sumptllbb', 'sumptllbbMET',
                      'mt2', 'mtl1', 'mtl2',  'mtlb1', 'mtlb2',
                      'mtlmin', 'mtlbmin', 'summtlb', 'summtl',
                      'mindPhiMETl', 'maxdPhiMETl', 'mindPhiMETb', 'maxdPhiMETb',
                      'avedPhiMETl', 'avedPhiMETb',
                      'dPhil1MET', 'dPhil2MET', 'dPhib1MET', 'dPhib2MET',
                      'dRl1b1', 'dRl1b2', 'dRl2b1', 'dRl2b2',
                      'sumdRlb', 'mindRlb', 'invsumdRlb', 'invmindRlb']
        #labels = []
    elif variable == "embedding":
        if dim is None:
            raise ValueError("Please specify the number of emdedded features used")
        else:
            embedding_dim = dim
            kinematics = [f'feat_{i + 1:02d}' for i in range(embedding_dim)]
    else:
        raise ValueError("bruh, pick a supported variable set or define a new one")

    return kinematics

def get_kinematics_labels(variable):
    """
    Function to get plot text label for kinematics set

    Args:
        variable (str): name of variable set
    
    Returns:
        (str): label
    """
    var_label_dict = {
        "mass": "HHH mass variables",
        "angular": "HHH angular variables",
        "shape": "Kinematic shape variables",
        "combined": "HHH mass, angular and kinematic shape variables",
        "mass and angular": "HHH mass and angular variables",
        "mass and shape": "HHH mass and kinematic shape variables",
        "LQ_HighLevel": "LQ High-level kinematic variables",
        "LQ_LowLevel": "LQ Low-level kinematic variables",
        "embedding": "Latent space variables",
        "stau": "stau variables"
    }

    if variable in var_label_dict:
        return var_label_dict[variable]
    else:
        raise KeyError("bruh, pick a better variable set (mass, angular, shape,\
                       combined, mass_and_angular, mass_and_shape, LQ_LowLevel,\
                       LQ_HighLevel, stau, embedding)")

def get_kinematics_staus(variable):
    """
    Function to obtain the list of names of kinematic variables to use for a given category.
    Throws an exception for an unknown category

    Args:
        variable (str): the category of variables you want the list for

    Returns:
        list(str): a list of kinematic variables
    """

    kin_var_0j = [
        ### met
        'met_Et', 'met_Signif', 'TST_Et',
        ### tau
        'tauPt', 'tauEta', 'tauNTracks',
        # 'tauM', #check this
        ### lep
        'lepPt', 'lepEta', 'lepD0', 'lepD0Sig', 'lepZ0',
        'lepZ0SinTheta','lepFlavor', #'lepCharge',
        ### dphi (losing parity info bc of abs? check this!)
        'dPhi_met_tst', 'dPhi_met_lep', 'dPhi_met_tau',
        'dPhi_tst_lep', 'dPhi_tst_tau', 'dPhi_lep_tau',
        ### dEta and dR
        'dEta_lep_tau', 'dR_lep_tau',
        ### angular
        'sum_cos',
        'met_cen',
        'cPhi1', 'cPhi2',
        'cos_star',
        ### balance
        'tau_lep_bal', 'met_bal',
        ### mass
        'mT_tau_met', 'mT_lep_met',
        'mT_sum', 'mCT_tau_lep','m_inv_tau_lep',
        ### mt2
        'mT2_0', 'mT2_10', 'mT2_20', 'mT2_30',
        'mT2_40', 'mT2_50', 'mT2_60',
        ### meff
        # 'myMeffInc20', 'myMeffInc20_tau',
        # 'myMeffInc30', 'myMeffInc30_tau',
        'myMeffInc40', 'myMeffInc40_tau',
        # 'myMeffInc50', 'myMeffInc50_tau',
        # 'nbjets_85WP'
        ### also try absEta?
    ]

    kin_var_j =  [
        'jetPt', 'jetEta', 'jetM',
        'dPhi_met_jet', 'dPhi_tst_jet',
        'dPhi_lep_jet', 'dPhi_tau_jet',
        'dEta_lep_jet', 'dEta_tau_jet',
        'dR_lep_jet', 'dR_tau_jet',
        'myHt40',
        'njets20', 'njets30', 'njets40', 'njets50',
        ]

    rtaus_var_0j = [#'rtau1_x', 'rtau1_y', 'rtau2_x', 'rtau2_y',
            'rtau1Pt', #'rtau1Phi',
            'rtau2Pt', #'rtau2Phi',
            'dPhi_met_rtau1', 'dPhi_met_rtau2',
            'dPhi_tst_rtau1', 'dPhi_tst_rtau2',
            'dPhi_rtau1_rtau2',
            'dR_rtau1_rtau2',
            'm_inv_rtaus',
            'rtaus_bal',
            'mCT_rtaus',
            'cos_star_rtaus', #'scale_factor'
            ]

    rtaus_var_j = [
            'dPhi_rtau1_jet', 'dPhi_rtau2_jet',
            'dR_rtau1_jet', 'dR_rtau2_jet',
            ]

    kin_var_0j = kin_var_0j + rtaus_var_0j
    kin_var_j = kin_var_j + rtaus_var_j

    if variable == "all":
        kinematics = kin_var_0j + kin_var_j
    elif variable == "no_jets":
        kinematics = kin_var_0j
    elif variable == "jets":
        kinematics = kin_var_j
    elif variable == "distance":
        # distance_var = kin_var
        kinematics = [
            'lep1D0Sig',
            'lep1Pt',
            'met_Et',
            'met_Signif',
            'm_inv_tau_lep',
            'mT_lep_met',
            'mT_sum',
            'mT_tau_met',
            'tauPt',
        ]
    else:
        raise ValueError("bruh, pick a better variable set for staus\
                         (all, no_jets, jets, distance)")

    return kinematics

def sig_mass_point(df_sig, mass_points):
    """
    Function to obtain signal for desired mass point(s)
    Args:
        df_sig (pd.DataFrame): the signal dataframe
        mass_points (list(str)): the mass point(s) you want to extract
    Returns:
        (pd.DataFrame): the signal dataframe for the desired mass point(s)
    """
    mass_point_dsid = { "100_1": 537028,
                "100_25": 537029,
                "100_50": 537030,
                "100_75": 537031,
                "100_90": 537032,
                "150_1": 537033,
                "150_50": 537034,
                "150_75": 537035,
                "150_100": 537036,
                "150_125": 537037,
                "150_140": 537038,
                "200_1": 537039,
                "200_50": 537040,
                "200_100": 537041,
                "200_125": 537042,
                "200_150": 537043,
                "200_175": 537044,
                "200_190": 537045,
                "250_1": 537046,
                "250_50": 537047,
                "250_100": 537048,
                "250_150": 537049,
                "250_175": 537050,
                "250_200": 537051,
                "250_225": 537052,
                "250_240": 537053,
                }

    df_sig_new = pd.DataFrame()
    for mp in mass_points:
        df_sig_mp = df_sig[df_sig.DatasetNumber == mass_point_dsid[mp]]
        df_sig_new = pd.concat([df_sig_new, df_sig_mp])

    return df_sig_new

def stau_selections(df):
    """
    Function to apply event selection for stau samples

    Args:
        df (pandas.df): initial stau event frame

    Returns
        (pandas.df): selected stau event frame
    """
    ### zero out the kinematic variables for events with no jets
    kin_var_j = get_kinematics_staus("jets")
    df.loc[df.njets40 == 0, kin_var_j] = 0

    df = df[df.nbjets_85WP == 0]
    df = df[df.met_Et > 15]
    df = df[df.met_Signif > 1.5]
    df = df[df.mT2_0 < 100]
    df = df[abs(df.dR_lep_tau) < 3.6]
    df = df[((df.mT_sum) > 70) &
            ((df.mT_lep_met > 20) | (df.mT_tau_met > 90)) &
            ((df.mT_lep_met > 90) | (df.mT_tau_met > 20))]
    df = df[((df.cPhi2 + df.cPhi1)> -1.25) &
            ((df.cPhi1 > 0.5) | (df.cPhi2 > -0.75)) &
            ((df.cPhi1 > -0.75) | (df.cPhi2 > 0.5))]

    return df

def cut_operation(df, cuts):
    """
    Function to apply a set of cuts to events,

    Args:
        df (pandas.df): input event dataframe before cuts

    Returns:
        (pandas.df): output event dataframe after cuts
    """
    conditions = []
    for variable, cut in cuts.items():
        threshold = cut.get("threshold")
        operation = cut.get("operation")
        if operation == ">":
            conditions.append(df[variable] > threshold)
        elif operation == "<":
            conditions.append(df[variable] < threshold)
        elif operation == ">=":
            conditions.append(df[variable] >= threshold)
        elif operation == "<=":
            conditions.append(df[variable] <= threshold)
        elif operation == "==":
            conditions.append(df[variable] == threshold)
        else:
            raise ValueError(f"Unsupported operation: {operation}")
    combined_condition = conditions[0]
    for condition in conditions[1:]:
        combined_condition &= condition
    return df[combined_condition]

def get_h5_paths(path, variable, distance, label="sampled_train"):
    """
    Function to obtain a list of h5 path + filenames for
      sig-sig, sig-bkg and bkg-bkg distance storage.

    Args:
        path (str): the path of directories you want to write the h5 files to.
        variable (str): the variable category under consideration
        distance (str): the distance metric under consideration
        label (str): the extra info needed for the filenames

    Returns:
        (str): sig-sig h5 file
        (str): sig-bkg h5 file
        (str): bkg-bkg h5 file
    """
    prefix = path + variable + "_" + distance
    types = ["sigsig", "sigbkg", "bkgbkg"]
    files = [prefix + "_" + t + "_" + label + ".h5" for t in types]

    return files[0], files[1], files[2]

def get_ntuples(path, signal):
    """
    Function to open the ntuples for our signal with uproot.

    Args:
        path (str): directory containing the ntuples
        signals (str): type of ntuples we expect to find/open

    Return:
        (list(str)): list of signal then background ntuples.
    """
    if signal == "hhh":
        signal_file = uproot.open(path + "6b_resonant_TRSM/out3_reco_6j_521176.root:tree")
        background_file = uproot.open(path + "data/out3_data_reco_5j.root:tree")
        return signal_file, background_file

    if signal == "LQ":
        signal_file = uproot.open(path + "GNNTree_LQ.root:tree")
        singletop_file = uproot.open(path + "GNNTree_singletop.root:tree")
        ttbar_file = uproot.open(path + "GNNTree_ttbar.root:tree")
        return signal_file, singletop_file, ttbar_file

def get_batched_distances(dist_path, variable, distance, batch_size,
                          species, sample=True, sample_frac=0.01, cutstring=""):
    """
    Function to obtain a list of .pt path + filenames for 
      sig-sig, sig-bkg and bkg-bkg distance storage.

    Args:
        dist_path (str): the path of directories you want to write the h5 files to.
        variable (str): the variable category under consideration
        distance (str): the distance metric under consideration
        batch_size (int): batch size to find the distance path
        species (str): the type of distance to load (sigsig, sigbkg, or bkgbkg)
        sample (bool): whether to sample from the distance distribution (default is True)
        sample_frac (float): what fraction of the distances per batch file
          to sample (default 1%)
        cutstring (str): cut string to find the distance path (default is "" for no cuts)

    Returns:
        (torch tensor): distance tensor
        (torch tensor): event weight tensor
    """
    dist_dir = dist_path + "/batched_" + str(batch_size) + "_"\
               + variable + "_" + distance + cutstring + "_distances/"
    print("looking for distances in :", dist_dir)
    files = glob.glob(dist_dir + species + '*0_0.pt')
    distance = torch.empty(0, dtype=torch.float32)
    wgt = torch.empty(0, dtype=torch.float32)
    max_dist = 0
    if sample:
        for f in files:
            print("loaded file: ", f)
            distance_tmp = torch.flatten(torch.load(f)["distance"]).to(torch.float32)
            max_dist = max(torch.max(distance_tmp), max_dist)
            batch_ind = np.linspace(0, len(distance_tmp)-1,
                                    int(sample_frac*len(distance_tmp)), dtype=int)
            print(species + " distance", torch.load(f)["distance"].shape)
            distance = torch.cat((distance, distance_tmp[batch_ind]))
            del distance_tmp
            wgt_tmp = torch.flatten(torch.load(f)["weight"]).to(torch.float32)
            wgt = torch.cat((wgt, wgt_tmp[batch_ind]))
            del wgt_tmp
    else:
        for f in files:
            distance_tmp = torch.flatten(torch.load(f)["distance"])
            max_dist = max(torch.max(distance_tmp), max_dist)
            print(species + " distance", torch.load(f)["distance"].shape)
            distance = torch.cat((distance, distance_tmp))
            del distance_tmp
            wgt_tmp = torch.flatten(torch.load(f)["weight"])
            wgt = torch.cat((wgt, wgt_tmp))
            del wgt_tmp

    return distance, wgt, max_dist

def load_config(file_path):
    """
    Function that loads in the training config file for a specific model
    Args:
        file_path (str): The path to find the config file
    Returns:
        (dict): the configuration dictionary
    """
    with open(file_path, "r", encoding="utf-8") as yaml_file:
        config = yaml.safe_load(yaml_file)
    return config

def get_train_mean_std(train_data):
    """
    Function to calculate the mean and standard deviation of the training data for all variables
    Args:
        train_data (pd.DataFrame): the training data
    Returns:
        tensor of len(number of variables): the mean of the training data
        tensor of len(number of variables): the standard deviation of the training data
    Additional Note:
        After testing, the std returned is different to the one used by the tensorflow or numpy.
        This is because the std is calculated using the unbiased estimator
          (N-1) in torch (Bessel's correction) by default.
    """
    means = train_data.mean(0)
    stds = train_data.std(0) #, correction = 0) ### correction = 0 turns off Bessel's correction
    return means, stds


def torch_standardise(in_tensor, train_mean, train_std):
    """
    Function to standardise a tensor using the mean and standard deviation of the training set
    Args:
        in_tensor (torch tensor): the tensor to standardise
        train_mean (torch tensor): the mean of the training set
        train_std (torch tensor): the standard deviation of the training set
    Returns:
        (torch tensor): the standardised tensor
    """
    return (in_tensor - train_mean) / train_std


def get_hist_initial_weights(file_path):
    """
    Function to open appropriate TTrees in our Ntuple Root files,
      for a given input path and signal choice
    Args:
        file_path (str): The path to the directory containing the ntuples
    Returns:
        a dictionary: containing N arrays of IntialWeights
          where N is the nubmer of signal or background
    """
    hist_initial_weights = uproot.open(file_path)["InitialWeights"].to_numpy()
    return hist_initial_weights


def calc_event_weight(df, initial_weights_arr, lumi):
    """
    Function to calculate the event weight to a given luminosity for an MC sample,
      and create a new column to store this for each event.
    Args:
        df (pandas dataframe): dataframe of events/variables for your sample.
        initial_weights_arr (df(float)): The sum of initial generator weights for sample.
        lumi (float): The luminosity you want to scale your samples to fb^-1.
    """
    sum_initial_weight = initial_weights_arr[0][2]
    xsec = df["xsec"]
    gen_weight = df["genWeight"]

    event_weight = xsec * gen_weight * lumi / sum_initial_weight
    return event_weight

def stable_int_from_string(s):
    """
    Function to turn a hash string into an integer

    Args:
        s (str): input string of int

    Returns:
        (int) stable int
    """
    h = hashlib.md5(s.encode('utf-8')).hexdigest()
    return int(h, 16)  # Big integer, deterministic everywhere

def assign_fold_deterministically(event_id, n_folds):
    """
    Function to assign k-fold fold for given event.

    Args:
        event_id (float): id key for event
        n_folds (int): number of k-folds to split between

    Returns:
        (int): fold number
    """
    seed = stable_int_from_string(str(event_id)) % (2**32)
    rng = np.random.RandomState(seed)
    return rng.randint(n_folds)
