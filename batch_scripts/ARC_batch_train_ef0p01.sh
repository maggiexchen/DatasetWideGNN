#! /bin/bash
#SBATCH --nodes=1
#SBATCH --time=47:59:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --partition=medium
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=holly.pacey@physics.ox.ac.uk
#SBATCH --mem=1000G
#SBATCH --clusters=htc

module load Anaconda3
source activate $DATA/torch-cpu

# change to your own area to run from:
cd $HOME/GNN/hhhgraph/

# check chmod +x war run on your torch_train.py, and this script.
echo "python torch_train.py --MLconfig config/ml_LQ_ef0p01.yaml --user config/user_Holly_ARC.yaml"
python torch_train.py --MLconfig config/ml_LQ_ef0p01.yaml --user config/user_Holly_ARC.yaml 

