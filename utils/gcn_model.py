"""Class to define the GNN model"""
#import pdb
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GraphConv
from torch_geometric.nn.conv.gat_conv_weighted import GATConv_weighted
from torch_geometric.utils._softmax_weighted import softmax_weighted
from torch.utils.checkpoint import checkpoint

sys.path.insert(0, 'external/pyg_custom')

class GCNClassifier(nn.Module):
    """
    Class defining pytorch MLP Classifier Network
    """
    def __init__(self, input_size, hidden_sizes_gcn, hidden_sizes_mlp,
                 output_size, dropout_rates, gnn_type = "GCN"):
        """
        Function to initialise GCNClassifier Class instance

        Args:
            input_size (int): dimension of input node features
            hidden_sizes_gcn (list(int)): list of number of hidden nodes in each GCN layer.
            hidden_sizes_mlp (list(int)): list of number of hidden nodes in each MLP layer.
            output_size (int): dimension of output features (for binary classification this is 1).
            dropout_rates (list(float)): list of the dropout rate in each GCN, then MLP, layer.
            gnn_type (str): layer type to use, GCN, GraphConv or GAT
        """
        super(GCNClassifier, self).__init__()

        assert len(dropout_rates) == len(hidden_sizes_gcn) + len(hidden_sizes_mlp),\
            "len(dropout_rates) should equal the number of hidden layers"
        self.layers_gcn = nn.ModuleList()
        self.batch_norms_gcn = nn.ModuleList()
        self.dropout_gcn = nn.ModuleList()
        i = 0
        for i, hidden_size in enumerate(hidden_sizes_gcn):
            if gnn_type == "GCN":
                self.layers_gcn.append(GCNConv(input_size, hidden_size))
            elif gnn_type == "GAT":
                self.layers_gcn.append(GATConv_weighted(input_size, hidden_size))
            elif gnn_type == "Graph":
                self.layers_gcn.append(GraphConv(input_size, hidden_size))
            else:
                raise ValueError("Invalid GNN type, please choose from 'GCN', 'GAT', 'Graph'")
            self.batch_norms_gcn.append(nn.BatchNorm1d(hidden_size))
            self.dropout_gcn.append(nn.Dropout(p=dropout_rates[i]))
            input_size = hidden_size

        self.layers_mlp = nn.ModuleList()
        self.batch_norms_mlp = nn.ModuleList()
        self.dropout_mlp = nn.ModuleList()
        for j, hidden_size in enumerate(hidden_sizes_mlp):
            self.layers_mlp.append(nn.Linear(input_size, hidden_size))
            self.batch_norms_mlp.append(nn.BatchNorm1d(hidden_size))
            if len(hidden_sizes_gcn) > 0:
                self.dropout_mlp.append(nn.Dropout(p=dropout_rates[len(hidden_sizes_gcn)+j]))
            else:
                self.dropout_mlp.append(nn.Dropout(p=dropout_rates[j]))
            input_size = hidden_sizes_mlp[j]

        self.output_layer = nn.Linear(input_size, output_size)
        self.gnn_type = gnn_type

    def forward(self, x, edge_index, edge_weights=None, mc_weights=None):
        """
        Function for forward propogation of the network layer

        Args:
            x (torch.tensor(float32)): Matrix of input features for each event
            edge_index ():  ...
            edge_weights ():
            mc_weights (): 
        Returns:
            (torch.tensor) 
        """
        gnn_type = self.gnn_type
        def gcn_forward(x, edge_index, gnn_type, edge_weights=None, mc_weights=None):
            for layer, batch_norm, dropout in \
                zip(self.layers_gcn, self.batch_norms_gcn, self.dropout_gcn):
                # Weights are the edge values in the sparse tensor object
                # for GATConv, use edge_attr instead of edge_weight
                if edge_weights is None:
                    x = F.relu(dropout(batch_norm(layer(x, edge_index))))
                else:
                    if gnn_type == "GAT":
                        x = F.relu(dropout(batch_norm(\
                            layer(x, edge_index, edge_weight=mc_weights,edge_attr=edge_weights))))
                    else:
                        x = F.relu(dropout(batch_norm(\
                            layer(x, edge_index, edge_weight=edge_weights))))
            return x
        x = checkpoint(gcn_forward, x, edge_index, gnn_type,\
                       edge_weights, mc_weights, use_reentrant=False)

        def mlp_forward(x):
            for layer, batch_norm, dropout in zip(self.layers_mlp, self.batch_norms_mlp,
                                                  self.dropout_mlp):
                x = F.relu(dropout(batch_norm(layer(x))))
            return x
        x = checkpoint(mlp_forward, x, use_reentrant=False)

        output = torch.sigmoid(self.output_layer(x))
        return output
