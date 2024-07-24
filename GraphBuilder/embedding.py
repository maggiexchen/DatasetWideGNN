import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

class EventDataset(Dataset):
    def __init__(self, pairs, labels):
        self.pairs = pairs
        self.labels = labels

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx][0], self.pairs[idx][1], self.labels[idx]

class EmbeddingNet(nn.Module):
    def __init__(self, input_dim, embedding_dim):
        super(EmbeddingNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, embedding_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class ContrastiveHingeLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveHingeLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        distances = torch.norm(output1 - output2, p=2, dim=1)
        losses = 0.5 * (label * distances ** 2 + (1 - label) * torch.relu(self.margin - distances) ** 2)
        return losses.mean()
