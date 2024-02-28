import torch.nn as nn
import torch.nn.functional as F
import torch
# from utils.gcn_layer import GCNLayer
from torch_geometric.nn import GCNConv

class GCNClassifier(nn.Module):
    def __init__(self, input_size, hidden_sizes_gcn, hidden_sizes_mlp, output_size):
        """
        input_sizes: dimension of input node features
        hidden_sizes: number of nodes in hidden graph layers as a list
        output_sizes: dimension of output features (for binary classification is 1)
        """
        super(GCNClassifier, self).__init__()
        self.layers_gcn = nn.ModuleList()
        self.batch_norms_gcn = nn.ModuleList()
        for i in range(len(hidden_sizes_gcn)):
            # self.layers.append(GCNLayer(input_size, hidden_sizes[i]))
            self.layers_gcn.append(GCNConv(input_size, hidden_sizes_gcn[i]))
            self.batch_norms_gcn.append(nn.BatchNorm1d(hidden_sizes_gcn[i]))
            input_size = hidden_sizes_gcn[i]
        
        self.layers_mlp = nn.ModuleList()
        self.batch_norms_mlp = nn.ModuleList()
        for j in range(len(hidden_sizes_mlp)):
            self.layers_mlp.append(nn.Linear(input_size, hidden_sizes_mlp[j]))
            self.batch_norms_mlp.append(nn.BatchNorm1d(hidden_sizes_mlp[j]))
            input_size = hidden_sizes_mlp[j]
        
        
        # self.output_layer = GCNLayer(input_size, output_size)
        self.output_layer = GCNConv(input_size, output_size)
            
    def forward(self, x, edge_index):
        for layer, batch_norm in zip(self.layers_gcn, self.batch_norms_gcn):
            x = F.relu(batch_norm(layer(x, edge_index)))
        for layer, batch_norm in zip(self.layers_mlp, self.batch_norms_mlp):
            x = F.relu(batch_norm(layer(x)))
        
        output = torch.sigmoid(self.output_layer(x, edge_index))
        return output
