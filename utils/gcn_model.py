import torch.nn as nn
import torch.nn.functional as F
import torch
# from utils.gcn_layer import GCNLayer
from torch_geometric.nn import GCNConv, GATConv, GraphConv
# from torch_geometric.nn import GATConv
import pdb

class GCNClassifier(nn.Module):
    def __init__(self, input_size, hidden_sizes_gcn, hidden_sizes_mlp, output_size, dropout_rates):
        """
        input_sizes: dimension of input node features
        hidden_sizes: number of nodes in hidden graph layers as a list
        output_sizes: dimension of output features (for binary classification is 1)
        """
        super(GCNClassifier, self).__init__()

        assert len(dropout_rates) == len(hidden_sizes_gcn) + len(hidden_sizes_mlp), "The number of dropout rates should be equal to the number of hidden layers in the model"
        self.layers_gcn = nn.ModuleList()
        self.batch_norms_gcn = nn.ModuleList()
        self.dropout_gcn = nn.ModuleList()
        for i in range(len(hidden_sizes_gcn)):
            # self.layers.append(GCNLayer(input_size, hidden_sizes[i]))
            # self.layers_gcn.append(GCNConv(input_size, hidden_sizes_gcn[i]))
            # self.layers_gcn.append(GATConv(input_size, hidden_sizes_gcn[i], edge_dim = 1))
            self.layers_gcn.append(GraphConv(input_size, hidden_sizes_gcn[i]))
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
            
    def forward(self, x, edge_index, edge_weights):
        for layer, batch_norm, dropout in zip(self.layers_gcn, self.batch_norms_gcn, self.dropout_gcn):
            x = F.relu(dropout(batch_norm(layer(x, edge_index, edge_weight=edge_weights)))) # for GATConv, use edge_attr instead of edge_weight
            # x = F.relu(dropout(batch_norm(layer(x, edge_index))))
        for layer, batch_norm, dropout in zip(self.layers_mlp, self.batch_norms_mlp, self.dropout_mlp):
            x = F.relu(dropout(batch_norm(layer(x))))
        
        output = torch.sigmoid(self.output_layer(x))
        return output
