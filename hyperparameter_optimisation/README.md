### Hyperparameter scan for DNN & GNN models

## DNN model optimisation
For the very first time, run
`chmod +x train_DNN.sh` and `chmod +x train_GNN.sh`

Modify the config files in `config/`, they should follow the same format as the usual user and ml configs
Note that for the GNN opitmisation, the distances, linking lengths, row/column indices of the adjacency matrix need to have already been generated with `../calc_distanc.py`, `../linking_length.py` and `../torch_adj_builder.py`. The file paths to them should be specified in the config file staring with `config/user_`.

Modify the parameters in `train_DNN.sh` and `train_GNN.sh` such as the `BASE_DIR` and `SAVE_DIR` to match your own directories.

For DNN scan, run
`./train_DNN.sh <model input variables (LQ_LowLevel / LQ_HighLevel)>`

For GNN scan, run
`./train_GNN.sh <model input variables (LQ_LowLevel / LQ_HighLevel)> <model (GCN/Graph)> <distance (euclidean, cosine, emd)>`

It will create a directory within `hyperparameter_optimisation/` and save all the models, metadata and performance json files and plots there.

To get the set of parameters for the best model (highest validation auc) and plot the correlation between all the parameters and performance metrics, run
`python plot_scans.py -m DNN -i <model input variables (LQ_LowLevel / LQ_HighLevel)>` for DNN, or 
`python plot_scans.py -m DNN -i <model input variables (LQ_LowLevel / LQ_HighLevel)> -d <distance (euclidean, cosine, emd)> -dv <distance variables (LQ_LowLevel / LQ_HighLevel)> -e <edge fraction>` for GNN
The corrlation plot will be saved in the `WhereYouSavedTheScans/plots/`
