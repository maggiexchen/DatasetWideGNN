"""Module of functions for distance metric calculations on pytorch tensors"""
import utils.misc as misc

import torch
import energyflow as ef
#import numpy as np

def get_emd_kinematics_key(keys, signal="LQ"):
    """
    Function to get indices for EMD kinematics in event tensor, so we can match them
      into the order we need for the EMD calculation

    Args:
        keys (list(str)): ordered list of keys - how kinematics are stored in tensor
        signal (str): type of signal to calcualted predefined EMD format for (default LQ)

    Returns:
        (list(int)): list of indices for the kinematics in the order we need for EMD
    """

    if signal == "LQ":

        key_indices = [keys.index('bjet1pt'), keys.index('bjet1eta'), keys.index('bjet1phi'),
                       keys.index('bjet2pt'), keys.index('bjet2eta'), keys.index('bjet2phi'),
                       keys.index('lep1pt'), keys.index('lep1eta'), keys.index('lep1phi'),
                       keys.index('lep2pt'), keys.index('lep2eta'), keys.index('lep2phi'),
                       keys.index('met'), keys.index('metphi')]

        return key_indices
    else:
        raise ValueError("only LQ EMD supported at the moment...")


def get_event_vectors(tensor, key_indices, signal="LQ"):
    """
    Function to get EMD-compatible event representations for batch of events.

    Args:
        tensor (torch.tensor): input event set
        keys (list(str)): ordered list of keys - how kinematics are stored in tensor
        signal (str): type of signal to calcualted predefined EMD format for (default LQ)

    Returns:
        (torch.tensor): EMD formatted event tensor
    """

    if signal == "LQ":

        # append new column to be the 'met Z' that we want to see as 0.
        n_events = tensor.size(dim=0)
        #print("start", tensor[0], tensor.shape)
        if len(key_indices) == 14:
            key_indices.insert(-1, 14)

        metz = torch.zeros([n_events, 1], dtype=torch.float32)
        tensor = torch.cat([tensor, metz], dim=1)
        del metz
        #print("added metz",tensor[0], tensor.shape)

        ordered = [x for x in range(tensor.size(dim=1))]
        #print("ordering setup: ", ordered, key_indices)
        tensor = tensor[:,torch.tensor(ordered)][:, torch.tensor(key_indices)]
        #print("ordered",tensor[0], tensor.shape)

        new_tensor = tensor.reshape(n_events, 5, 3)
        del tensor, key_indices
        #print("reshaped",new_tensor[0], new_tensor.shape)

        return new_tensor

    else:
        raise ValueError("only LQ EMD supported at the moment...")


def distance_calc(a, b, metric):
    """
    Function to pick a distance metric function and calculate it for a,b arrays

    Args:
        a (torch.tensor): 1st set of events for distance calculation
        b (torch.tensor): 2nd set of events for distance calculation
        metric (str): distance metric to use

    Returns:
        (torch.tensor): matrix of axb distances
    """
    if metric == "euclidean":
        d = euclidean(a,b)
    elif metric == "cityblock":
        d = cityblock(a,b)
    elif metric == "braycurtis":
        d = braycurtis(a,b)
    elif metric == "cosine":
        d = cosine(a,b)
    elif metric == "emd":
        d = emd(a,b)
    else:
        d = None
        print("Please specify a valid distance metric, from euclidean, cityblock, braycurtis or cosine")

    if torch.sum(torch.isnan(d)).item() != 0:
        raise ArithmeticError("NaN present in distances")
    else:
        return d

def emd(a, b, r=1.0, gdim=2, n_jobs=-1):
    """
    Function to obtain the Earth Mover's distance between two vectors

    Args:
        a (torch.tensor): first set of events
        b (torch.tensor): second set of events
        r (float): "The R parameter in the EMD definition that controls the relative
          importance of the two terms. Must be greater than or equal to half of the
          maximum ground distance in the space in order for the EMD to be a valid
          metric satisfying the triangle inequality."
        gdim (int): "The dimension of the ground metric space. Useful for restricting
          which dimensions are considered part of the ground space when using the internal
          euclidean distances between particles. Has no effect if dists are provided."
        n_jobs (int): "The number of cpu cores to use. A value of None or -1 will use
          as many threads as there are CPUs on the machine."

    Returns:
        (torch.tensor): matrix of axb distances
    """
    emd_np = ef.emd.emds(a, b, R=r, gdim=gdim, n_jobs=n_jobs)
    return torch.tensor(emd_np)


def cosine(a, b):
    """
    Function to obtain the cosine distance between two vectors

    Args:
        a (torch.tensor): first set of events
        b (torch.tensor): second set of events

    Returns:
        (torch.tensor): matrix of axb distances
    """
    numerator = torch.matmul(a, torch.transpose(b, 0, 1)).to(torch.float32)
    denominator = torch.matmul(torch.unsqueeze(torch.norm(a, dim=1), 1),
                               torch.unsqueeze(torch.norm(b, dim=1), 0)).to(torch.float32)

    return (1 - numerator/denominator).to(torch.float32)


def euclidean(a, b):
    """
    Function to obtain the euclidean distance between two vectors

    Args:
        a (torch.tensor): first set of events
        b (torch.tensor): second set of events

    Returns:
        (torch.tensor): matrix of axb distances
    """
    a_expanded = torch.unsqueeze(a, dim=1).to(torch.float32)
    b_expanded = torch.unsqueeze(b, dim=0).to(torch.float32)

    return torch.sqrt(torch.sum(torch.square(a_expanded-b_expanded),dim=-1)).to(torch.float32)

def cityblock(a, b):
    """
    Function to obtain the cityblock distance between two sets of events

    Args:
        a (torch.tensor): first set of events
        b (torch.tensor): second set of events

    Returns:
        (torch.tensor): matrix of axb distances
    """
    a_expanded = torch.unsqueeze(a, dim=1).to(torch.float32)
    b_expanded = torch.unsqueeze(b, dim=0).to(torch.float32)

    return torch.sum(torch.abs(a_expanded-b_expanded),dim=-1).to(torch.float32)


def braycurtis(a, b):
    """
    Function to obtain the braycurtis distance between two sets of events

    Args:
        a (torch.tensor): first set of events
        b (torch.tensor): second set of events

    Returns:
        (torch.tensor): matrix of axb distances
    """
    a_expanded = torch.unsqueeze(a, dim=1).to(torch.float32)
    b_expanded = torch.unsqueeze(b, dim=0).to(torch.float32)

    num = torch.sum(torch.abs(a_expanded-b_expanded), dim=-1)
    den = torch.sum(torch.abs(a_expanded), dim=-1) + torch.sum(torch.abs(b_expanded), dim=-1)
    return (num/den).to(torch.float32)


def torch_emd(a, b, objects, kinematics):
    """
    Function to obtain the Earth Mover's Distance between two sets of events

    Args:
        a (torch.tensor): first set of events
        b (torch.tensor): second set of events
        objects (list(str)): particles to use for EMD calculation
        kinematics (list(str)): list of kinematics needed for EMD calculation

    Returns:
        (torch.tensor): matrix of axb distances
    """
    a = misc.get_event_vectors_torch(a, objects, kinematics)
    b = misc.get_event_vectors_torch(b, objects, kinematics)
    d = ef.emd.emds(a,b, R=1.0, gdim=2, n_jobs=-1)
    return  torch.from_numpy(d)
