### Hyperparameter scan for DNN & GNN models

## DNN model optimisation
For the very first time, run
`chmod +x train_DNN.sh` and `chmod +x train_GNN.sh`

Modify the config files in `config/`, they should follow the same format as the usual user and ml configs

Modify the parameters in `train_DNN.sh` and `train_GNN.sh` such as the `BASE_DIR` and `SAVE_DIR` to match your own directories.

For DNN scan, run
`./train_DNN.sh <variables> DNN`

For GNN scan, run
`./train_GNN.sh <variables> <model (GCN/Graph)> <distance>`

It will create a directory within `hyperparameter_optimisation` and save all the models, jsons and plots there.
