import numpy as np
import pandas as pd
import torch

def cosine(a, b):
    numerator = torch.matmul(a, b, transpose_b=True)
    denominator = torch.sqrt(torch.tensordot(torch.sum(torch.square(a),dim=1), torch.sum(torch.square(b),dim=1)))
    print(1 - numerator / denominator)
    return 1 - numerator / denominator

def euclidean(a, b):
    a_expanded = torch.unsqueeze(a, dim=1)
    b_expanded = torch.unsqueeze(b, dim=0)
    return torch.sqrt(torch.sum(torch.square(a_expanded-b_expanded),dim=-1))

def cityblock(a, b):
    a_expanded = torch.unsqueeze(a, dim=1)
    b_expanded = torch.unsqueeze(b, dim=0)
    return torch.sum(torch.abs(a_expanded-b_expanded),dim=-1)

