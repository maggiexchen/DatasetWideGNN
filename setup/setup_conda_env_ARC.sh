srun -p interactive --pty /bin/bash

!# /bin/bash

# Load the version of Anaconda you need
module load Anaconda3

# Create an environment in $DATA and give it an appropriate name
export CONPREFIX=$DATA/torch-gpu
conda create --prefix $CONPREFIX

# Activate your environment
source activate $CONPREFIX

# Install packages...
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
conda install conda-forge::pytorch_geometric
conda install -c conda-forge shap
conda install anaconda::pandas
conda install matplotlib
conda install pytables
conda install pandas

pip install uproot
pip install mplhep
pip install fsspec
pip install torchinfo
pip install torcheval
pip install energyflow
pip install pydantic
pip install pyg-lib -f https://data.pyg.org/whl/torch-2.3.0+cu121.html
