import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import math

class GCNLayer(nn.Module):
    # TODO add attention layer ...
    def __init__(self, dim_in, dim_out, use_batch_norm=True):
        """
        dim_in: dimension of input node features
        dim_out: dimension of output features (for binary classification is 1 for final layer)
        node_weights: weights (event weights) of the nodes (events)
        """
        super(GCNLayer, self).__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.use_batch_norm = use_batch_norm

        self.train_weight = nn.Parameter(torch.FloatTensor(dim_in, dim_out))
        self.train_bias = nn.Parameter(torch.FloatTensor(dim_out))

        if self.use_batch_norm:
            self.batch_norm = nn.BatchNorm1d(dim_out)
        
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.train_weight.data)
        nn.init.zeros_(self.train_bias.data)

    def forward(self, x, adjacency_matrix):
    #def forward(self, x):
        output = torch.matmul(torch.matmul(adjacency_matrix,x), self.train_weight)+self.train_bias
        if self.use_batch_norm:
            output = self.batch_norm(output)

        return output
