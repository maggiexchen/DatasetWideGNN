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

pip install mplhep
pip install fsspec
pip install torchinfo
