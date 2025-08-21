!# /bin/bash

# Load the version of Anaconda you need
module load Anaconda3

# Create an environment in $DATA and give it an appropriate name
export CONPREFIX=$DATA/torch-cpu

# Activate your environment
source activate $CONPREFIX


