import os

def create_dirs(path):
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

def get_h5_paths(file_path, variable, distance, label="sampled_train"):
  prefix = file_path + variable + "_" + distance
  types = ["sigsig", "sigbkg", "bkgbkg"]
  files = [prefix+"_"+t+"_"+label+".h5" for t in types]
  return files[0], files[1], files[2]
