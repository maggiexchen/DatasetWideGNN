#! /bin/bash
#SBATCH --nodes=1
#SBATCH --time=02:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --partition=short
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=holly.pacey@physics.ox.ac.uk

module load Anaconda3
source activate $DATA/torch-gpu

# change to your own area to run from:
cd $HOME/GNN/hhhgraph/

#echo "Running: python calc_distance.py --user config/user_Holly.yaml -v $3 -d $2 -b $1"
echo "python calc_distance.py --user config/user_Holly_ARC.yaml -v LQ_HighLevel -d cosine -b 10000"
python calc_distance.py --user config/user_Holly_ARC.yaml -v LQ_HighLevel -d cosine -b 10000
