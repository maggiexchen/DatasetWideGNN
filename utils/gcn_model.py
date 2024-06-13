import torch.nn as nn
import torch.nn.functional as F
import torch
# from utils.gcn_layer import GCNLayer
from torch_geometric.nn import GCNConv, GATConv, GraphConv
# from torch_geometric.nn import GATConv
import pdb

class GCNClassifier(nn.Module):
    def __init__(self, input_size, hidden_sizes_gcn, hidden_sizes_mlp, output_size, dropout_rates, gnn_type = "GCN"):
        """
        input_size (int): dimension of input node features
        hidden_sizes_gcn (list(int)): list of number of hidden nodes in each GCN layer.
        hidden_sizes_mlp (list(int)): list of number of hidden nodes in each MLP layer.
        output_size (int): dimension of output features (for binary classification this is 1).
        dropout_rates (list(float)): list of the dropout rate in each GCN, then MLP, layer.
        """
        super(GCNClassifier, self).__init__()

        assert len(dropout_rates) == len(hidden_sizes_gcn) + len(hidden_sizes_mlp), "The number of dropout rates should be equal to the number of hidden layers in the model"
        self.layers_gcn = nn.ModuleList()
        self.batch_norms_gcn = nn.ModuleList()
        self.dropout_gcn = nn.ModuleList()
        i = 0
        for i in range(len(hidden_sizes_gcn)):
            # self.layers.append(GCNLayer(input_size, hidden_sizes[i]))
            if gnn_type == "GCN":
                self.layers_gcn.append(GCNConv(input_size, hidden_sizes_gcn[i]))
            elif gnn_type == "GAT":
                self.layers_gcn.append(GATConv(input_size, hidden_sizes_gcn[i]))#, edge_dim = 1))
            elif gnn_type == "Graph":
                self.layers_gcn.append(GraphConv(input_size, hidden_sizes_gcn[i]))
            else:
                raise ValueError("Invalid GNN type, please choose from 'GCN', 'GAT', 'GraphConv'")
            self.batch_norms_gcn.append(nn.BatchNorm1d(hidden_sizes_gcn[i]))
            self.dropout_gcn.append(nn.Dropout(p=dropout_rates[i]))
            input_size = hidden_sizes_gcn[i]
        
        self.layers_mlp = nn.ModuleList()
        self.batch_norms_mlp = nn.ModuleList()
        self.dropout_mlp = nn.ModuleList()
        for j in range(len(hidden_sizes_mlp)):
            self.layers_mlp.append(nn.Linear(input_size, hidden_sizes_mlp[j]))
            self.batch_norms_mlp.append(nn.BatchNorm1d(hidden_sizes_mlp[j]))
            self.dropout_mlp.append(nn.Dropout(p=dropout_rates[i+j]))
            input_size = hidden_sizes_mlp[j]
        
        
        # self.output_layer = GCNLayer(input_size, output_size)
        # self.output_layer = GCNConv(input_size, output_size)
        self.output_layer = nn.Linear(input_size, output_size)
            
    def forward(self, x, edge_index):
        """
        Function for forward propogation of the network layer
        Args:
            x (torch.tensor(float32)): Matrix of input features for each event
            edge_index ():  ...
        Returns:
            (torch.tensor) 
        """
        for layer, batch_norm, dropout in zip(self.layers_gcn, self.batch_norms_gcn, self.dropout_gcn):
            # when using sparse tensor object, edge_weight is not used
            # Weights are the edge values in the sparse tensor object
            # for GATConv, use edge_attr instead of edge_weight
            # x = F.relu(dropout(batch_norm(layer(x, edge_index, edge_weight=edge_weights)))) 
            x = F.relu(dropout(batch_norm(layer(x, edge_index))))
        for layer, batch_norm, dropout in zip(self.layers_mlp, self.batch_norms_mlp, self.dropout_mlp):
            x = F.relu(dropout(batch_norm(layer(x))))
        
        output = torch.sigmoid(self.output_layer(x))
        return output
