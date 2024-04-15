import os
import yaml
import glob
import torch

def create_dirs(path):
    """
    Function to create any directories needed to store the output of a function.

    Args:
        path (str): the path of directories you want to exist

    Returns:
        void
    """
    base = path.split("/")[0]
    dirs = path.split("/")[1:-1]
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
    else:
        raise Exception("bruh, pick a better variable set (mass, angular, shape, combined, mass_and_angular, mass_and_shape)")

    return kinematics

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

def get_batched_files(path, variable, distance, t):
    """
    Function to obtain a list of .pt path+filenames for sig-sig, sig-bkg and bkg-bkg distance storage. 

    Args:
        path (str): the path of directories you want to write the h5 files to.
        variable (str): the variable category under consideration
        distance (str): the distance metric under consideration
        t (str): the typeof distance to load (sigsig, sigbkg, or bkgbkg)

    Returns:
        (str): sig-sig pt file
        (str): sig-bkg pt file
        (str): bkg-bkg pt file
    """
    prefix = path+"/batched_"+variable +"_"+distance+"_distances/"
    files = glob.glob(prefix+t+'*0.pt')
    distance = []
    wgt = []
    for f in files:
        distance.append(torch.flatten(torch.load(f)["distance"]))
        wgt.append(torch.flatten(torch.load(f)["weight"]))
    distance = torch.cat(distance, dim=0)
    wgt = torch.cat(wgt, dim=0)
    return distance, wgt

def load_config(file_path):
    """
    Function that loads in the training config file for a specific model
    """
    with open(file_path, "r") as yaml_file:
        config = yaml.safe_load(yaml_file)
    return config