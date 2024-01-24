import torch.nn as nn
import torch.nn.functional as F
import torch
from utils.gcn_layer import GCNLayer

class DNNClassifier(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        """
        input_sizes: dimension of input node features
        hidden_sizes: number of nodes in hidden graph layers as a list
        output_sizes: dimension of output features (for binary classification is 1)
        """
        super(DNNClassifier, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(len(hidden_sizes)):
            self.layers.append(nn.Linear(input_size, hidden_sizes[i]))
            input_size = hidden_sizes[i]
        
        self.output_layer = nn.Linear(input_size, output_size)

    def forward(self,x):
        for layer in self.layers:
            x = F.relu(layer(x))
        output = torch.sigmoid(self.output_layer(x))
        return output
