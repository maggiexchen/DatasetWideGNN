srun -p interactive --pty /bin/bash

!# /bin/bash

# Load the version of Anaconda you need
module load Anaconda3

# Create an environment in $DATA and give it an appropriate name
export CONPREFIX=$DATA/torch-cpu
conda create --prefix $CONPREFIX python=3.10

# Activate your environment
source activate $CONPREFIX

pip install pyyaml pydantic pandas shap tables uproot mplhep fsspec energyflow

pip install torch==2.1.0+cpu torchvision==0.16.0+cpu torchaudio==2.1.0+cpu -f https://download.pytorch.org/whl/torch_stable.html

pip install pyg_lib==0.3.1+pt21cpu torch_cluster==1.6.3+pt21cpu torch_scatter==2.1.2+pt21cpu torch_sparse==0.6.18+pt21cpu torch_spline_conv==1.2.2+pt21cpu -f https://data.pyg.org/whl/torch-2.1.0+cpu.html

pip install torch-geometric
