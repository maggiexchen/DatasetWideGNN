### Hyperparameter scan for DNN & GNN models

## DNN model optimisation
First make sure you run 
`chmod +x train_DNN.sh`

Modify the config files in `config/`, they should follow the same format as the usual user and ml configs

Modify the parameters in `train_DNN.sh` such as the `BASE_DIR` and `SAVE_DIR` to match your own directories.

For DNN scan, run
`./ train.DNN.sh <variables> DNN`

It will create a directory within `hyperparameter_optimisation` and save all the models, jsons and plots there.

## GNN model optimisation
Not quite ready yet.