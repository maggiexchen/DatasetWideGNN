# If you need to setup Holly's conda : eval "$(/data/atlas/users/pacey/miniconda/bin/conda shell.bash hook)"

conda create --name torch-gpu

conda activate torch-gpu
conda update -n base -c defaults conda

conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
conda install conda-forge::pytorch_geometric
conda install -c conda-forge shap
conda install anaconda::pandas
conda install conda-forge::matplotlib
conda install conda-forge::pytables
conda install conda-forge::pandas
conda install -c conda-forge uproot

pip install mplhep
pip install fsspec
pip install torchinfo
pip install torcheval
pip install energyflow
pip install pydantic
