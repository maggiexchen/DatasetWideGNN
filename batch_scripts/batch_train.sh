#! /bin/bash
echo "ml config " $1
echo "userconfig " $2

# change to your own area to run from:
cd /home/pacey/GNN/master/hhhgraph/

source setup/setup_conda_env_Holly.sh

# check chmod +x war run on your torch_train.py, and this script.
python torch_train.py --MLconfig $1 --user $2
