import torch.nn as nn
import torch.nn.functional as F
import torch
from utils.gcn_layer import GCNLayer
from torch_geometric.nn import GCNConv

class GCNClassifier(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        """
        input_sizes: dimension of input node features
        hidden_sizes: number of nodes in hidden graph layers as a list
        output_sizes: dimension of output features (for binary classification is 1)
        """
        super(GCNClassifier, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(len(hidden_sizes)):
            # self.layers.append(GCNLayer(input_size, hidden_sizes[i]))
            self.layers.append(GCNConv(input_size, hidden_sizes[i]))
            input_size = hidden_sizes[i]
        
        
        # self.output_layer = GCNLayer(input_size, output_size)
        self.output_layer = GCNConv(input_size, output_size)

    def forward(self, x, edge_index):
        for layer in self.layers:
            x = F.relu(layer(x, edge_index))
        output = torch.sigmoid(self.output_layer(x, edge_index))
        return output
