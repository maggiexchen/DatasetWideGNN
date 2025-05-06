#! /bin/bash

# change to your own area to run from:
cd /home/pacey/GNN/new_merge/hhhgraph/

source setup/setup_conda_env_Holly.sh

# check chmod +x was run on your script and this script.

echo "running: python linking_length.py --variable LQ_HighLevel --user config/user_Holly.yaml --MLconfig config/ml_LQ.yaml --distance euclidean"
python linking_length.py --variable LQ_HighLevel --user config/user_Holly.yaml --MLconfig config/ml_LQ.yaml --distance euclidean

