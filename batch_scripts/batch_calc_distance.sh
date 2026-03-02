#! /bin/bash
echo "batch size:" $1
echo "distance:" $2
echo "variable:" $3

# change to your own area to run from:
cd /home/pacey/GNN/master/hhhgraph/

source setup/setup_conda_env_Holly.sh

# check chmod +x was run on your script and this script.

#echo "Running: python calc_distance.py --user config/user_Holly.yaml -v LQ -d euclidean -b $1"
##python calc_distance.py --user config/user_Holly.yaml -v LQ_HighLevel -d $2 -b $1
python calc_distance.py --user config/user_Holly.yaml -v $3 -d $2 -b $1
#python calc_distance_cuda_parallel.py --user config/user_Holly.yaml -v LQ_HighLevel -d euclidean -b $1
