#!/bin/bash

BASE_DIR="/data/atlas/atlasdata3/maggiechen/gnn_project"

# Define the path to the user config & ml training file

USER_CONFIG_FILE=$BASE_DIR"/hyperparameter_optimisation/config/user_Maggie_GNN_scan.yaml"
ML_CONFIG_FILE=$BASE_DIR"/hyperparameter_optimisation/config/ml_LQ_GNN_scan.yaml"

metadata_save_path=$SAVE_DIR"metadata/"
mkdir -p $metadata_save_path
cp "$0" "$metadata_save_path/train_GNN.sh"
cp "$USER_CONFIG_FILE" "$metadata_save_path/user_Maggie_GNN_scan.yaml"
cp "$ML_CONFIG_FILE" "$metadata_save_path/ml_LQ_GNN_scan.yaml"

# GNN parameters to fix
ML_VAR_LEVEL=$1 # LQ_LowLevel or LQ_HighLevel
DIST_VAR_LEVEL=$2 # variables used to calculate the graph
MODEL=$3 # GCN or Graph
DISTANCE=$4 # euclidean, cosine, or emd
EDGE_FRAC=0.1
EPOCH=20
SINGLEFOLD=1
VALFRAC=4

SAVE_DIR=$BASE_DIR"/hyperparameter_optimisation/"$MODEL"_"$DISTANCE"_"$DIST_VAR_LEVEL"_Inputs_"$ML_VAR_LEVEL"_EdgeFrac_"$EDGE_FRAC"/"

sed -i "s|^ml_variable: .*|ml_variable: $ML_VAR_LEVEL|" "$ML_CONFIG_FILE"
sed -i "s|^distance_variable: .*|distance_variable: $DIST_VAR_LEVEL|" "$ML_CONFIG_FILE"
sed -i "s|^gnn_type: .*|gnn_type: $MODEL|" "$ML_CONFIG_FILE"
sed -i "s|^distance: .*|distance: $DISTANCE|" "$ML_CONFIG_FILE"
sed -i "s|^edge_frac: .*|edge_frac: $EDGE_FRAC|" "$ML_CONFIG_FILE"
sed -i "s|^epochs: .*|epochs: $EPOCH|" "$ML_CONFIG_FILE"
sed -i "s|^single_fold: .*|single_fold: $SINGLEFOLD|" "$ML_CONFIG_FILE"
sed -i "s|^num_folds: .*|num_folds: $VALFRAC|" "$ML_CONFIG_FILE"

BATCHSIZE=(1024 2048)
DROPOUT=(0)
LR=(0.0005 0.001 0.005 0.1)
LR_PATIENCE=(3)

# MLP_HIDDEN_NODES=(5 10)
# MLP_LAYERS=(3)
# GNN_HIDDEN_NODES=(5 10 15 20)
# GNN_LAYERS=(2 3 4)
# NEIGHBOURS=(4 8 12 24)
# NEIGHBOURS_LAYERS=(2 4 6)

MLP_HIDDEN_NODES=(5)
MLP_LAYERS=(2)
GNN_HIDDEN_NODES=(10)
GNN_LAYERS=(2)
NEIGHBOURS=(10)
NEIGHBOURS_LAYERS=(2)

# Define model dir and file name
model_save_path=$SAVE_DIR"models/"
plot_save_path=$SAVE_DIR"plots/"
sed -i "s|^model_path: .*|model_path: $model_save_path|" "$USER_CONFIG_FILE"
sed -i "s|^plot_path: .*|plot_path: $plot_save_path|" "$USER_CONFIG_FILE"

# Loop through the parameter values
# First loop through the LR, LR patience, dropout, batchsize
for lr_pat in "${LR_PATIENCE[@]}"; do
    for dropout in "${DROPOUT[@]}"; do
        for bs in "${BATCHSIZE[@]}"; do
            for lr in "${LR[@]}"; do
                sed -i "s|^patience_LR: .*|patience_LR: $lr_pat|" "$ML_CONFIG_FILE"
                echo "LR patience: " $lr_pat
                sed -i "s|^LR: .*|LR: $lr|" "$ML_CONFIG_FILE"
                echo "LR: " $lr
                sed -i "s|^batch_size: .*|batch_size: $bs|" "$ML_CONFIG_FILE"
                echo "Batchsize: " $bs
                
                for nbs in "${NEIGHBOURS[@]}"; do
                    for nb_layers in "${NEIGHBOURS_LAYERS[@]}"; do
                        nb_sampling="["
                        for ((n=1; n<=nb_layers; n++)); do
                            nb_sampling+="$nbs"
                            if [ $n -lt $nb_layers ]; then
                                nb_sampling+=", "
                            fi
                        done
                        nb_sampling+="]"
                        sed -i "s|^num_nb_list: .*|num_nb_list: $nb_sampling|" "$ML_CONFIG_FILE"
                        echo "Neighbour sampling: " $nb_sampling

                        for dnn_nodes in "${MLP_HIDDEN_NODES[@]}"; do
                            for dnn_layers in "${MLP_LAYERS[@]}"; do
                                mlp_hidden_layers="["
                                dropout_rates="["
                                for ((i=1; i<=dnn_layers; i++)); do
                                    mlp_hidden_layers+="$dnn_nodes"
                                    if [ $i -lt $dnn_layers ]; then
                                        mlp_hidden_layers+=", "
                                    fi
                                done
                                for ((j=1; j<=dnn_layers; j++)); do
                                    dropout_rates+="$dropout"
                                    if [ $j -le $dnn_layers ]; then
                                    dropout_rates+=", "
                                    fi
                                done
                                mlp_hidden_layers+="]"
                                sed -i "s|^hidden_sizes_mlp: .*|hidden_sizes_mlp: $mlp_hidden_layers|" "$ML_CONFIG_FILE"
                                echo "MLP layers: " $mlp_hidden_layers 

                                for gnn_nodes in "${GNN_HIDDEN_NODES[@]}"; do
                                    for gnn_layers in "${GNN_LAYERS[@]}"; do
                                    gnn_hidden_layers="["
                                        for ((k=1; k<=gnn_layers; k++)); do
                                            gnn_hidden_layers+="$gnn_nodes"
                                            if [ $k -lt $gnn_layers ]; then
                                            gnn_hidden_layers+=", "
                                            fi
                                        done
                                        for ((l=$dnn_layers; l<$((dnn_layers + gnn_layers)); l++)); do
                                            dropout_rates+="$dropout"
                                            if [ $l -lt $((gnn_layers + dnn_layers-1)) ]; then
                                            dropout_rates+=", "
                                            fi
                                        done
                                        dropout_rates+="]"
                                        gnn_hidden_layers+="]"
                                        sed -i "s|^hidden_sizes_gcn: .*|hidden_sizes_gcn: $gnn_hidden_layers|" "$ML_CONFIG_FILE"
                                        echo "GNN layers: " $gnn_hidden_layers          
                                        sed -i "s|^dropout_rates: .*|dropout_rates: $dropout_rates|" "$ML_CONFIG_FILE"
                                        echo "Dropout rates: " $dropout_rates
                                        
                                        python $BASE_DIR"/torch_train.py" -u "$USER_CONFIG_FILE" -c "$ML_CONFIG_FILE"
                                    done
                                done
                            done
                        done
                    done
                done
            done
        done
    done
done