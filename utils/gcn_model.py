import torch.nn as nn
import torch.nn.functional as F
import torch
from utils.gcn_layer import GCNLayer

class GCNClassifier(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, dropout_rate):
        """
        input_sizes: dimension of input node features
        hidden_sizes: number of nodes in hidden graph layers as a list
        output_sizes: dimension of output features (for binary classification is 1)
        """
        super(GCNClassifier, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(len(hidden_sizes)):
            self.layers.append(GCNLayer(input_size, hidden_sizes[i], dropout_rate))
            input_size = hidden_sizes[i]
        
        self.output_layer = GCNLayer(input_size, output_size, dropout_rate)

    def forward(self, x, adjacency_matrix):
        for layer in self.layers:
            x = F.relu(layer(x, adjacency_matrix))
        output = torch.sigmoid(self.output_layer(x, adjacency_matrix))
        return output
