#! /bin/bash
#SBATCH --nodes=1
#SBATCH --time=11:59:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --partition=short
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=holly.pacey@physics.ox.ac.uk
#SBATCH --mem=1000G
#SBATCH --clusters=htc

module load Anaconda3
source activate $DATA/torch-gpu

# change to your own area to run from:
cd $HOME/GNN/hhhgraph/

# check chmod +x war run on your torch_train.py, and this script.
echo "python torch_adj_builder.py --MLconfig config/ml_LQ_ef0p05.yaml --user config/user_Holly_ARC.yaml -b 10000"
python torch_adj_builder.py --MLconfig config/ml_LQ_ef0p05.yaml --user config/user_Holly_ARC.yaml -b 10000

