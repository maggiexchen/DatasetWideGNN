import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import math

class GCNLayer(nn.Module):
    # TODO add attention layer ...
    def __init__(self, dim_in, dim_out):
        """
        dim_in: dimension of input node features
        dim_out: dimension of output features (for binary classification is 1)
        node_weights: weights (event weights) of the nodes (events)
        """
        super(GCNLayer, self).__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.train_weight = nn.Parameter(torch.FloatTensor(dim_in, dim_out))
        self.train_bias = nn.Parameter(torch.FloatTensor(dim_out))
        #self.node_weights = nn.Parameter(node_weights)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.train_weight.data)
        nn.init.zeros_(self.train_bias.data)

    def forward(self, x, adjacency_matrix):
        #weighted_node_features = torch.mul(x, self.node_weights[:, None])
        # [N x # kinematics] x [# kinematics x # hidden nodes] = [N x # hidden nodes]
        # [N x N] x [N x # hidden nodes] = [N x # hidden nodes]
        # output = torch.matmul(adjacency_matrix, torch.matmul(weighted_node_features, self.train_weight)) + self.train_bias
        # weighted_adj_mat = torch.mul(adjacency_matrix, self.node_weights[:, None])
        output = torch.matmul(adjacency_matrix, torch.matmul(x, self.train_weight))+self.train_bias
        return output