"""functions to apply various scalings to elements in an array"""
import numpy
import torch

def mad_norm(dist1, dist2):
    """
    Function to perform the median absolute deviation
    scaling of 2 dists, using the dist2 median and MAD.

    Args:
        dist1 (numpy.array): sig-like array to normalise
        dist2 (numpy.array): bkg-like array to normalise and derive the median and MAD

    Returns:
        (numpy.array): MAD normalised sig-like array
        (numpy.array): MAD normalised bkg-like array
    """
    dist2_median = numpy.median(dist2)
    dist2_mad = numpy.median(abs(dist2-dist2_median))
    norm_dist1 = (dist1 - dist2_median)/dist2_mad
    norm_dist2 = (dist2 - dist2_median)/dist2_mad

    return norm_dist1, norm_dist2


def minmax(dist, d_min, d_max):
    """
    Function to perform the minmax normalisation that scales a distribution to be between 0 and 1

    Args:
        dist (numpy.array): array to normalise
        d_min (float): original min value to use
        d_max (float): original max value to use

    Returns:
        (numpy.array): minmax normalised array
    """
    norm_dist = (dist - d_min)/(d_max-d_min)

    return norm_dist


# The reverse of minmax that returns the original values
def reverse_minmax(norm_value, d_min, d_max):
    """
    Function to reverse the minmax normalisation on a given value
    (that scaled a distribution to be between 0 and 1), to return to the original values.

    Args:
        norm_value (float): value to un-normalise
        d_min (float): original min value to restore
        d_max (float): original max value to restore

    Returns:
        (float): original value
    """
    value = norm_value * (d_max-d_min) + d_min

    return value

def standardise(dist):
    """
    Function to normalise a distribution to have zero mean and unit standard deviation.

    Args:
        dist (numpy.array): array to normalise

    Returns:
        (numpy.array): standardised array
    """
    mean = numpy.mean(dist)
    std = numpy.std(dist)
    return (dist-mean)/std

def standardise_tensor(dist):
    """
    Function to normalise a distribution to have zero mean and unit standard deviation.

    Args:
        dist (torch.tensor): tensor to normalise

    Returns:
        (torch.tensor): standardised tensor
    """
    mean = torch.mean(dist)
    std = torch.std(dist)
    return (dist-mean)/std
