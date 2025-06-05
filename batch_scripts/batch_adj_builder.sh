#! /bin/bash


# change to your own area to run from:
cd /home/pacey/GNN/new_merge/hhhgraph/

source setup/setup_conda_env_Holly.sh

# check chmod +x war run on your torch_train.py, and this script.
python torch_adj_builder.py --MLconfig config/ml_LQ.yaml --user config/user_Holly.yaml -b 10000

