import numpy as np
import pandas as pd
import utils.misc as misc
import energyflow as ef
import torch

def cosine(a, b):
    """
    Function to obtain the cosine distance between two vectors

    Args:
        a (pytorch.tensor): first vector
        b (pytorch.tensor): second vector

    Returns:
        (float) cosine distance
    """
    numerator = torch.matmul(a, torch.transpose(b, 0, 1)).to(torch.float32)
    denominator = torch.matmul(torch.unsqueeze(torch.norm(a, dim=1), 1), torch.unsqueeze(torch.norm(b, dim=1), 0)).to(torch.float32)

    return (1 - numerator/denominator).to(torch.float16)


def euclidean(a, b):
    """
    Function to obtain the euclidean distance between two vectors

    Args:
        a (pytorch.tensor): first vector
        b (pytorch.tensor): second vector

    Returns:
        (float) euclidean distance
    """
    a_expanded = torch.unsqueeze(a, dim=1).to(torch.float32)
    b_expanded = torch.unsqueeze(b, dim=0).to(torch.float32)

    return torch.sqrt(torch.sum(torch.square(a_expanded-b_expanded),dim=-1)).to(torch.float32)

def cityblock(a, b):
    """
    Function to obtain the cityblock distance between two vectors

    Args:
        a (pytorch.tensor): first vector
        b (pytorch.tensor): second vector

    Returns:
        (float) cityblock distance
    """
    a_expanded = torch.unsqueeze(a, dim=1).to(torch.float16)
    b_expanded = torch.unsqueeze(b, dim=0).to(torch.float16)

    return torch.sum(torch.abs(a_expanded-b_expanded),dim=-1).to(torch.float16)


def torch_emd(a, b, objects, kinematics):
    a = misc.get_event_vectors_torch(a, objects, kinematics)
    b = misc.get_event_vectors_torch(b, objects, kinematics)
    d = ef.emd.emds(a,b, R=1.0, gdim=2, n_jobs=-1)
    return  torch.from_numpy(d)

