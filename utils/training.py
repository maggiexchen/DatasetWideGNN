import torch
import numpy as np

def weighted_bce_loss(output, target, class_weights, event_weights):
    """
    """
    sig_w_sum = class_weights[1] * target * event_weights
    bkg_w_sum = class_weights[0] * (1-target) * event_weights
    sig_loss =  sig_w_sum * torch.log(output+1e-10)
    bkg_loss =  bkg_w_sum * torch.log(1-output+1e-10)
    loss = sig_loss+bkg_loss
    sum_w = (sig_w_sum + bkg_w_sum).sum()
    loss = loss.sum() / sum_w
    return -loss

def binary_class_weights(labels, event_weights):
    """
    """
    num_sig = np.sum(event_weights[labels == 1])
    num_bkg = np.sum(event_weights[labels == 0])
    bkg_weight = 1
    sig_weight = num_bkg/num_sig
    return torch.tensor([bkg_weight, sig_weight], dtype=torch.float)