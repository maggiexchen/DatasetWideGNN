#!/bin/bash

BASE_DIR="/data/atlas/atlasdata3/maggiechen/gnn_project"
SAVE_DIR="/data/atlas/atlasdata3/maggiechen/gnn_project/hyperparameter_optimisation"

# Define the path to the user config & ml training file

USER_CONFIG_FILE=$BASE_DIR"/hyperparameter_optimisation/config/user_Maggie_DNN_scan.yaml"
ML_CONFIG_FILE=$BASE_DIR"/hyperparameter_optimisation/config/ml_LQ_DNN_scan.yaml"

# GNN parameters to fix
VAR_LEVEL=$1 # LQ_LowLevel or LQ_HighLevel
MODEL="DNN" # DNN, GCN or Graph
EPOCH=5
SINGLEFOLD=1
VALFRAC=4

sed -i "s|^kinematic_variable: .*|kinematic_variable: $VAR_LEVEL|" "$ML_CONFIG_FILE"
sed -i "s|^gnn_type: .*|gnn_type: $MODEL|" "$ML_CONFIG_FILE"
sed -i "s|^epochs: .*|epochs: $EPOCH|" "$ML_CONFIG_FILE"
sed -i "s|^single_fold: .*|single_fold: $SINGLEFOLD|" "$ML_CONFIG_FILE"
sed -i "s|^n_fold: .*|n_fold: $VALFRAC|" "$USER_CONFIG_FILE"

# Parameters to optimise
BATCHSIZE=(1024 2048)
DROPOUT=(0)
LR=(0.0005 0.001 0.005 0.1)
LR_PATIENCE=(3)

MLP_HIDDEN_NODES=(5 10 15 20)
MLP_LAYERS=(2 3 4)

# Define model dir and file name
model_save_path=$SAVE_DIR"/"$MODEL"_"$VAR_LEVEL"_EdgeFrac"$EDGE_FRAC"/models/"
plot_save_path=$SAVE_DIR"/"$MODEL"_"$VAR_LEVEL"_EdgeFrac"$EDGE_FRAC"/plots/"
sed -i "s|^model_path: .*|model_path: $model_save_path|" "$USER_CONFIG_FILE"
sed -i "s|^plot_path: .*|plot_path: $plot_save_path|" "$USER_CONFIG_FILE"

# Loop through the parameter values
# First loop through the LR, LR patience, dropout, batchsize
for lr_pat in "${LR_PATIENCE[@]}"; do
    for dropout in "${DROPOUT[@]}"; do
        for bs in "${BATCHSIZE[@]}"; do
            for lr in "${LR[@]}"; do
                sed -i "s|^patience_LR: .*|patience_LR: $lr_pat|" "$ML_CONFIG_FILE"
                sed -i "s|^LR: .*|LR: $lr|" "$ML_CONFIG_FILE"
                sed -i "s|^batch_size: .*|batch_size: $bs|" "$ML_CONFIG_FILE"
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
                        for ((i=1; i<=dnn_layers; i++)); do
                            dropout_rates+="$dropout"
                            if [ $i -lt $dnn_layers ]; then
                            dropout_rates+=", "
                            fi
                        done
                        mlp_hidden_layers+="]"
                        dropout_rates+="]"
                        sed -i "s|^hidden_sizes_mlp: .*|hidden_sizes_mlp: $mlp_hidden_layers|" "$ML_CONFIG_FILE"
                        echo "Training with MLP layers: " $mlp_hidden_layers 
                        sed -i "s|^dropout_rates: .*|dropout_rates: $dropout_rates|" "$ML_CONFIG_FILE"

                        python $BASE_DIR"/torch_train.py" -u"$USER_CONFIG_FILE" -c "$ML_CONFIG_FILE"
                    done
                done
            done
        done
    done
done