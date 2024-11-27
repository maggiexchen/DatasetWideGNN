import os
import yaml
import glob
import torch
import math
import pandas as pd
import pdb
import uproot
torch.manual_seed(42)

cpu = torch.device('cpu')
device = torch.device('cuda:0')

def print_mem_info():
    """
    Function to calculate and print out the total memory available and used on CUDA.
    """
    GB = 1024**3
    t = torch.cuda.get_device_properties(0).total_memory/GB
    r = torch.cuda.memory_reserved(0)/GB
    a = torch.cuda.memory_allocated(0)/GB
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
        if not os.path.isdir(base+"/"+tmp_dir):
            try:
                os.mkdir(base+"/"+tmp_dir)
                print("creating: ", base+"/"+tmp_dir)
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
        background_type = [
                            'Wjets',
                            'Zlljets',
                            'Zttjets2214',
                            'diboson0L','diboson1L','diboson2L',
                            'diboson3L','diboson4L','triboson',
                            'higgs',
                            'singletop','topOther','ttV','ttbar_incl'
                            ]
    else:
        raise Exception("Signal type is either hhh, LQ or stau")
    return background_type


def get_kinematics(variable):
    """
    Function to obtain the list of names of kinematic variables to use for a given category.
    Throws an exception for an unknown category

    Args:
        variable (str): the category of variables you want the list for

    Returns:
        list(str): the list of names of kinematic variables
    """
    if variable == "mass":
        # mass-based kinematics
        #kinematics = ["mH1","mH2","mH3","mHHH","mHcosTheta","meanmH","rmsmH","meanmBB","rmsmBB","meanPt","rmsPt","ht","massfraceta","massfracphi","massfracraw"]
        kinematics = ["mH1","mH2","mH3","mHHH"]
    elif variable == "angular":
        # angular kinematics
        kinematics = ["dRH1","dRH2","dRH3","meandRBB"]
    elif variable == "shape":
        # event shape kinematics
        kinematics = ["sphere3dv2b","sphere3dv2btrans","aplan3dv2b","theta3dv2b"]
    elif variable == "combined":
        kinematics = ["mH1","mH2","mH3","mHHH","dRH1","dRH2","dRH3","meandRBB","sphere3dv2b","sphere3dv2btrans","aplan3dv2b","theta3dv2b"]
    elif variable == "mass_and_angular":
        kinematics = ["mH1","mH2","mH3","mHHH","dRH1","dRH2","dRH3","meandRBB"]
    elif variable == "mass_and_shape":
        kinematics = ["mH1","mH2","mH3","mHHH","sphere3dv2b","sphere3dv2btrans","aplan3dv2b","theta3dv2b"]
    elif variable == "LQ":
        kinematics = ['met', 'sumptllbb', 'mindPhiMETl',  'mtl1', 'mtl2']
    else:
        raise Exception("bruh, pick a better variable set (mass, angular, shape, combined, mass_and_angular, mass_and_shape)")

    return kinematics

def get_kinematics_staus(variable):
    """
    Function to obtain the list of names of kinematic variables to use for a given category.
    Throws an exception for an unknown category

    Args:
        variable (str): the category of variables you want the list for

    Returns:
        list(str): the list of names of kinematic variables
    """
       
    kin_var_0J = [
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

    kin_var_J =  [
        'jetPt', 'jetEta', 'jetM',
        'dPhi_met_jet', 'dPhi_tst_jet', 
        'dPhi_lep_jet', 'dPhi_tau_jet',
        'dEta_lep_jet', 'dEta_tau_jet',
        'dR_lep_jet', 'dR_tau_jet',
        'myHt40', 
        'njets20', 'njets30', 'njets40', 'njets50',
        ]

    rtaus_var_0J = [#'rtau1_x', 'rtau1_y', 'rtau2_x', 'rtau2_y', 
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

    rtaus_var_J = [
            'dPhi_rtau1_jet',  
            'dPhi_rtau2_jet',
            'dR_rtau1_jet',
            'dR_rtau2_jet', 
            ]

    kin_var_0J = kin_var_0J + rtaus_var_0J
    kin_var_J = kin_var_J + rtaus_var_J

    if variable == "all": 
        kinematics = kin_var_0J + kin_var_J
    elif variable == "no_jets":
        kinematics = kin_var_0J
    elif variable == "jets":
        kinematics = kin_var_J
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
        raise Exception("bruh, pick a better variable set for staus (all, no_jets, jets, distance)")
    
    return kinematics

def sig_mass_point(df_sig, mass_points = ['100_50']):
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

    ### zero out the kinematic variables for events with no jets
    kin_var_J = get_kinematics_staus("jets")
    df.loc[df.njets40 == 0, kin_var_J] = 0

    df = df[df.nbjets_85WP == 0]
    df = df[df.met_Et > 15]
    df = df[df.met_Signif > 1.5]
    df = df[df.mT2_0 < 100]
    df = df[abs(df.dR_lep_tau) < 3.6]
    df = df[((df.mT_sum) > 70) & ((df.mT_lep_met > 20) | (df.mT_tau_met > 90)) & ((df.mT_lep_met > 90) | (df.mT_tau_met > 20))]
    df = df[((df.cPhi2+df.cPhi1)> -1.25) & ((df.cPhi1 > 0.5) | (df.cPhi2 > -0.75)) & ((df.cPhi1 > -0.75) | (df.cPhi2 > 0.5))]

    return df

def calc_eventWeight(df, initialWeights_arr, lumi):
    sumInitialWeight = initialWeights_arr[0][2]
    xsec = df["xsec"]
    genWeight = df["genWeight"]
    
    # event weights = xsec * genWeight * lumi / sumInitialWeight
    eventWeight = xsec * genWeight * lumi / sumInitialWeight
    return eventWeight


def get_h5_paths(path, variable, distance, label="sampled_train"):
    """
    Function to obtain a list of h5 path+filenames for sig-sig, sig-bkg and bkg-bkg distance storage. 

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
    files = [prefix+"_"+t+"_"+label+".h5" for t in types]

    return files[0], files[1], files[2]

def get_batched_distances(dist_path, variable, distance, t, sample=True):
    """
    Function to obtain a list of .pt path+filenames for sig-sig, sig-bkg and bkg-bkg distance storage. 

    Args:
        dist_path (str): the path of directories you want to write the h5 files to.
        variable (str): the variable category under consideration
        distance (str): the distance metric under consideration
        t (str): the type of distance to load (sigsig, sigbkg, or bkgbkg)
        sample (bool): whether to sample from the distance distribution (default is True)

    Returns:
        (torch tensor): distance tensor
        (torch tensor): event weight tensor
    """
    dist_dir = dist_path+"/batched_"+variable +"_"+distance+"_distances/"
    files = glob.glob(dist_dir + t + '*.pt')
    distance = torch.empty(0, dtype=torch.float16)
    wgt = torch.empty(0, dtype=torch.float16)
    if sample:
        num_sample = 20000
        batch_sample = math.ceil(num_sample / len(files))
        sample_count = 0
        while sample_count < num_sample:
            for f in files:
                distance_tmp = torch.flatten(torch.load(f)["distance"])
                wgt_tmp = torch.flatten(torch.load(f)["weight"])
                batch_ind = torch.randperm(len(distance_tmp))[:batch_sample]
                print(t+" distance", torch.load(f)["distance"].shape)
                distance = torch.cat((distance, distance_tmp[batch_ind]))
                wgt = torch.cat((wgt, wgt_tmp[batch_ind]))
                del distance_tmp
                del wgt_tmp
                sample_count += batch_sample
    else:
        for f in files:
            distance_tmp = torch.flatten(torch.load(f)["distance"])
            wgt_tmp = torch.flatten(torch.load(f)["weight"])
            print(t+" distance", torch.load(f)["distance"].shape)
            distance = torch.cat((distance, distance_tmp))
            wgt = torch.cat((wgt, wgt_tmp))
            del distance_tmp
            del wgt_tmp
            sample_count += batch_sample

    return distance, wgt

def load_config(file_path):
    """
    Function that loads in the training config file for a specific model
    Args:
        file_path (str): The path to find the config file
    Returns:
        (dict): the configuration dictionary
    """
    with open(file_path, "r") as yaml_file:
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
        This is because the std is calculated using the unbiased estimator (N-1) in torch (Bessel's correction) by default.
    """
    means = train_data.mean(0)
    stds = train_data.std(0) #, correction = 0) ### correction = 0 turns off Bessel's correction
    return means, stds


def torch_standardise(input, train_mean, train_std):
    """
    Function to standardise a tensor using the mean and standard deviation of the training set
    Args:
        input (torch tensor): the tensor to standardise
        train_mean (torch tensor): the mean of the training set
        train_std (torch tensor): the standard deviation of the training set
    Returns:
        (torch tensor): the standardised tensor
    """
    return (input - train_mean) / train_std


def get_histInitialWeights(file_path):
    """
    Function to open appropriate TTrees in our Ntuple Root files, for a given input path and signal choice
    Args:
        file_path (str): The path to the directory containing the ntuples
    Returns:
        a dictionary: containing N arrays of IntialWeights where N is the nubmer of signal or background
    """
    histInitialWeights = uproot.open(file_path)["InitialWeights"].to_numpy()
    return histInitialWeights


def GetEventWeight(df, lumi, sumInitWeights):
    """
    Function to calculate the event weight to a given luminosity for an MC sample, and create a new column to store this for each event.
    Args:
        df (pandas dataframe): dataframe for the input events/variables for your MC sample
        lumi (float): The luminoisty you want to scale your samples to.
        sumInitWeights (float): The sum of the initial generator weights of all of the generated samples (before some cuts are placed when producing ntuples) = the effective number of events generated initially.
    """
    df["eventWeight"] = df["xsec"] * df["genWeight"] * lumi / sumInitWeights
