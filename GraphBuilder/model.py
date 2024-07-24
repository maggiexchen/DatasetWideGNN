from embedding import EmbeddingNet
import torch
import torch.nn as nn

class MetricLearningModel(nn.Module):
    def __init__(self, input_dim, embedding_dim):
        super(MetricLearningModel, self).__init__()
        self.embedding_net = EmbeddingNet(input_dim, embedding_dim)

    def forward(self, x1, x2):
        embed1 = self.embedding_net(x1)
        embed2 = self.embedding_net(x2)
        return embed1, embed2

def contrastive_hinge_loss(output1, output2, label, margin=1.0):
    distances = torch.norm(output1 - output2, p=2, dim=1)
    losses = 0.5 * (label * distances ** 2 + (1 - label) * torch.relu(margin - distances) ** 2)
    return losses.mean()