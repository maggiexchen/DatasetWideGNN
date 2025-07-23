#! /bin/bash
echo "batch size:" $1
echo "distance:" $2
echo "variable:" $3

# change to your own area to run from:
cd /home/pacey/GNN/new_merge/hhhgraph/

source setup/setup_conda_env_Holly.sh

# check chmod +x was run on your script and this script.

echo "running: python linking_length.py --variable $3 --user config/user_Holly.yaml --MLconfig config/ml_LQ.yaml --distance $2 --batchsize $1"
python linking_length.py --variable $3 --user config/user_Holly.yaml --MLconfig config/ml_LQ.yaml --distance $2 --batchsize $1

