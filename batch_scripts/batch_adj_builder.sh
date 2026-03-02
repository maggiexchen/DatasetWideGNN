#! /bin/bash
echo "batch size" $1
echo "ml config" $2
echo "user config" $3

# change to your own area to run from:
cd /home/pacey/GNN/master/hhhgraph/

source setup/setup_conda_env_Holly.sh

# check chmod +x war run on your torch_train.py, and this script.
python torch_adj_builder.py --MLconfig $2 --user $3 -b $1

