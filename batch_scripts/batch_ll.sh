#! /bin/bash

# change to your own area to run from:
cd /home/pacey/GNN/TestMphys/hhhgraph/

source setup/setup_conda_env_Holly.sh

# check chmod +x was run on your script and this script.
python linking_length.py --user config/user_Holly.yaml -v LQ -d euclidean

