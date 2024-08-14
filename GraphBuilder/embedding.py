import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.checkpoint import checkpoint

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
        self.embedding_dim = embedding_dim
        self.fc1 = nn.Linear(input_dim, 64)
        self.bn1 = nn.BatchNorm1d(64)

        self.fc2 = nn.Linear(64, 64)
        self.bn2 = nn.BatchNorm1d(64)

        self.fc3 = nn.Linear(64, embedding_dim)
        self.bn3 = nn.BatchNorm1d(embedding_dim)

    def forward(self, x):
        x = checkpoint(self.fc1, x)
        x = self.bn1(x)
        x = torch.relu(x)
        
        x = checkpoint(self.fc2, x)
        x = self.bn2(x)
        x = torch.relu(x)
        
        x = checkpoint(self.fc3, x)
        x = self.bn3(x)
        
        return x


class ContrastiveHingeLoss(nn.Module):
    def __init__(self, embedding_dim, margin=1.0, pen=1.0):
        super(ContrastiveHingeLoss, self).__init__()
        self.margin = margin
        self.embedding_dim = embedding_dim
        self.pen = pen
    
    def forward(self, output1, output2, label):
        distance = F.pairwise_distance(output1, output2, p=2)
        loss = label * torch.pow(distance, 2) + (1-label) * torch.pow(F.relu(self.margin-distance),2) * self.pen
        return loss.mean()