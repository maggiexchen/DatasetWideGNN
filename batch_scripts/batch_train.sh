#! /bin/bash
echo "ml config " $1
echo "userconfig " $2

# change to your own area to run from:
cd /data/atlas/users/honscheid/HHH_GNN/hhhgraph

source /data/atlas/users/honscheid/HHH_GNN/hhhgraph/setup/setup_conda_env_Ruben.sh

# check chmod +x war run on your torch_train.py, and this script.
python torch_train.py --MLconfig $1 --user $2
