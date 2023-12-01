import numpy as np
import pandas as pd
import torch

def cosine(a, b):
    numerator = torch.matmul(a, torch.transpose(c, 0, 1))
    denominator = torch.sqrt(torch.tensordot(torch.sum(torch.square(a),dim=1), torch.sum(torch.square(b),dim=1)))
    return 1 - numerator / denominator

def euclidean(a, b):
    a_expanded = torch.unsqueeze(a, dim=1)
    b_expanded = torch.unsqueeze(b, dim=0)
    return torch.transpose(torch.sqrt(torch.sum(torch.square(a_expanded-b_expanded),dim=-1)),0,1)

def cityblock(a, b):
    a_expanded = torch.unsqueeze(a, dim=1)
    b_expanded = torch.unsqueeze(b, dim=0)
    return torch.transpose(torch.sum(torch.abs(a_expanded-b_expanded),dim=-1),0,1)
