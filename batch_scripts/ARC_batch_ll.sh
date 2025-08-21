#! /bin/bash
#SBATCH --nodes=1
#SBATCH --time=02:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --partition=short
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=holly.pacey@physics.ox.ac.uk
#SBATCH --mem=40G

module load Anaconda3
source activate $DATA/torch-gpu

# change to your own area to run from:
cd $HOME/GNN/hhhgraph/

echo "running: python linking_length.py --variable LQ_HighLevel --user config/user_Holly.yaml --MLconfig config/ml_LQ.yaml --distance cosine --batchsize 10000"
python linking_length.py --variable LQ_HighLevel --user config/user_Holly_ARC.yaml --MLconfig config/ml_LQ.yaml --distance cosine --batchsize 10000

