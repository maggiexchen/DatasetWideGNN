"""Class to define the DNN layers for the network"""
#import math

import torch
import torch.nn as nn
#from torch.nn.parameter import Parameter
#from torch.nn.modules.module import Module

class DNNLayer(nn.Module):
    """
    Class defining pytorch MLP layer
    """
    # TODO add attention layer ...
    def __init__(self, dim_in, dim_out, dropout_rate, use_batch_norm=True):
        """
        Function to initialise DNNLayer Class instance

        Args:
            dim_in (int): dimension of input node features
            dim_out (int): dimension of output features (for binary classifier, final layer=1)
            dropout rate (list(float)): list of dropout rates to apply per layer
            use_batch_norm (bool): use 1D batch normalisation.
        """
        super(DNNLayer, self).__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.train_weight = nn.Parameter(torch.FloatTensor(dim_in, dim_out))
        self.train_bias = nn.Parameter(torch.FloatTensor(dim_out))

        if use_batch_norm:
            self.batch_norm = nn.BatchNorm1d(dim_out)

        self.dropout = nn.Dropout(p=dropout_rate)
        self.reset_parameters()

    def reset_parameters(self):
        """
        Function to reset MLP hidden layer weights/biases to zero.
        """
        nn.init.xavier_uniform_(self.train_weight.data)
        nn.init.zeros_(self.train_bias.data)

    def forward(self, x):
        """
        Function defining forward propagation of MLP hidden layer

        Args:
            x (torch.tensor): input feature vector
        Returns:
            (torch.tensor): output of the hidden layer
        """
        output =  torch.matmul(x, self.train_weight)+self.train_bias

        if self.use_batch_norm:
            output = self.batch_norm(output)

        output = self.dropout(output)

        return output
