"""Class to define the DNN model"""
import utils.dnn_layer as dnn

import torch.nn as nn
import torch.nn.functional as F
import torch

class DNNClassifier(nn.Module):
    """
    Class defining pytorch MLP Classifier Network
    """
    def __init__(self, input_size, hidden_sizes, output_size, dropout_rate):
        """
        Function to initialise DNNClassifier Class instance

        Args:
            input_size (int): dimension of input node features
            hidden_sizes (list(int): number of nodes in hidden graph layers as a list
            output_size (int): dimension of output features (for binary classification is 1)
            dropout rate (list(float)): list of dropout rates to apply per layer
        """
        super(DNNClassifier, self).__init__()
        self.layers = nn.ModuleList()
        for i in enumerate(hidden_sizes):
            self.layers.append(nn.Linear(input_size, hidden_sizes[i], dropout_rate))
            input_size = hidden_sizes[i]

        self.output_layer = nn.Linear(input_size, output_size, dropout_rate)

    def forward(self,x):
        """
        Function defining forward propagation of MLP

        Args:
            x (torch.tensor): input feature vector
        Returns:
            (torch.tensor): output of the final hidden layer
        """
        for layer in self.layers:
            x = F.relu(layer(x))
        output = torch.sigmoid(self.output_layer(x))

        return output
