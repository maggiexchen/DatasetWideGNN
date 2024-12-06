#!/bin/bash

# Define the path to the YAML file
CONFIG_FILE="config/user_Maggie.yaml"

# Define the ranges or lists of values you want to scan through
MARGINS=(0.5 1.0 2.0)
EMBEDDING_DIMS=(18)
PENALTY=(1 5 10 20)
# Loop through the parameter values
for margin in "${MARGINS[@]}"; do
    for embedding_dim in "${EMBEDDING_DIMS[@]}"; do
        for penalty in "${PENALTY[@]}"; do
            echo "Running with margin=$margin, embedding_dim=$embedding_dim, penalty=$penalty"

            # Update the YAML file directly with the new parameter values
            sed -i "s/^margin: .*/margin: $margin/" "$CONFIG_FILE"
            sed -i "s/^embedding_dim: .*/embedding_dim: $embedding_dim/" "$CONFIG_FILE"
            sed -i "s/^penalty: .*/penalty: $penalty/" "$CONFIG_FILE"
            python train.py -u "$CONFIG_FILE"
        done
    done
done