"""Functions to find LL thresholds for a particular sigsig_eff/edge_frac"""
import numpy as np
import torch

def find_threshold(tpr, fpr, eff, cut, flip=False):
    """
    Function to find the threshold on the distance metric that provides a given sig-sig efficiency.
    Args:
        tpr (numpy.array): of true-positive rates, sig-sig passing cut
        fpr (numpy.array): of false-positive rates, bkg-bkg(sig-bkg) passing cut
        eff (float): sig-sig efficiency desired
        cut (numpy.array): of thresholds considered
        flip (bool): True if sig-sig have smaller distances, False if sig-sig have larger distances
    Returns:
        (list(float)): coordinates of cut point to draw on on ROC curve
        (float): cut to apply on distance
    """
    # the tpr here is in reverse order with the discriminant cut, so it's <=
    if flip:
        tpr_index = np.argmax(tpr <= eff)
    else:
        tpr_index = np.argmax(tpr >= eff)
    return [tpr[tpr_index], fpr[tpr_index]], cut[tpr_index]


def find_threshold_edge_frac(sigsig, sigbkg, bkgbkg, edge_frac, flip):
    """
    Function that finds the threshold on the distance that provides a given edge fraction
    Args:
        sigsig (torch.Tensor): all sigsig connections
        sigbkg (torch.Tensor): all bkgbkg connections
        bkgbkg (torch.Tensor): all sigbkg connections
        edge_frac (float): target edge fraction
        flip (bool): friend or enemy graph?
    Returns:
        (list(float)): coordinates of cut point to draw on on ROC curve
        (float): cut to apply to distances
    """
    total_dist = torch.cat([sigsig, sigbkg, bkgbkg])
    total_len = total_dist.numel()
    if flip:
        sorted_dist, _ = torch.sort(total_dist)
    else:
        sorted_dist, _ = torch.sort(total_dist, descending=True)
    linking_length = []
    for frac in edge_frac:
        k = int(torch.ceil(torch.tensor(frac * total_len)))
        linking_length.append(sorted_dist[k - 1].item())
    return linking_length
