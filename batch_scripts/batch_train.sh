#! /bin/bash

# change to your own area to run from:
cd /home/pacey/GNN/TestMphys/hhhgraph/

source setup/setup_conda_env_Holly.sh

# check chmod +x war run on your torch_train.py, and this script.
python torch_train.py --MLconfig config/ml_default.yaml --user config/user_Holly.yaml
