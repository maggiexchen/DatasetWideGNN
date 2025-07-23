"""Functions to find LL thresholds for a particular sigsig_eff/edge_frac"""
import numpy as np
import torch

def find_threshold(tpr, fpr, eff, cut, is_target_closest=False):
    """
    Function to find the threshold on the distance metric that provides a given sig-sig efficiency.
    Args:
        tpr (numpy.array): of true-positive rates, target-target distance type passing cut
        fpr (numpy.array): of false-positive rates, reject distance type passing cut 
        eff (float): target-target efficiency desired
        cut (numpy.array): of thresholds considered
        is_target_closest (bool): True if target-target distances are smaller then the reject type
    Returns:
        (list(float)): coordinates of cut point to draw on on ROC curve
        (float): cut to apply on distance
    """
    # the tpr here is in reverse order with the discriminant cut, so it's <=
    if is_target_closest:
        tpr_index = np.argmax(tpr <= eff)
    else:
        tpr_index = np.argmax(tpr >= eff)
    return [tpr[tpr_index], fpr[tpr_index]], cut[tpr_index]


def find_threshold_edge_frac(sigsig, sigbkg, bkgbkg, edge_frac, x_max, do_friend_graph=True):
    """
    Function that finds the threshold on the distance that provides a given edge fraction
      uses a fine unweighted histogram cumulative sum approach
    Args:
        sigsig (torch.Tensor): all sigsig connections
        sigbkg (torch.Tensor): all bkgbkg connections
        bkgbkg (torch.Tensor): all sigbkg connections
        edge_frac (float): target edge fraction
        x_max (float): max distance value to consider across species
        do_friend_graph (bool): friend or enemy graph (default True)
    Returns:
        (list(float)): coordinates of cut point to draw on on ROC curve
        (float): cut to apply to distances
    """

    n_bins = 10001
    binning = np.linspace(0., x_max, n_bins)
    print("making sigsig hist")
    sigsig_hist = np.histogram(sigsig, bins=binning)
    del sigsig
    print("making sigbkg hist")
    sigbkg_hist = np.histogram(sigbkg, bins=binning)
    del sigbkg
    print("making bkgbkg hist")
    bkgbkg_hist = np.histogram(bkgbkg, bins=binning)
    del bkgbkg

    bins = sigsig_hist[1]
    #binwidth = bins[1] - bins[0]
    bin_centres = [ bins[b] + 0.5*(bins[b+1]-bins[b]) for b in range(0,n_bins-1) ]
    distance_hist_total = np.zeros(n_bins-1)
    distance_hist_total = np.add(sigsig_hist[0], distance_hist_total)
    distance_hist_total = np.add(sigbkg_hist[0], distance_hist_total)
    distance_hist_total = np.add(bkgbkg_hist[0], distance_hist_total)
    distance_hist_total = distance_hist_total / np.sum(distance_hist_total)

    print("made total hist")
    if not do_friend_graph:
        np.flip(distance_hist_total)

    cumulative_distance_hist = np.cumsum(distance_hist_total)
    print("made cumulative hist: ", cumulative_distance_hist)

    linking_length = []
    for frac in edge_frac:
        diff_array = np.absolute(cumulative_distance_hist - frac)
        index = diff_array.argmin()
        nearest_frac = cumulative_distance_hist[index]
        print(index, nearest_frac, frac)
        if not do_friend_graph:
            index = len(distance_hist_total) - index
        nearest_distance = bin_centres[index]
        print(index, nearest_distance, nearest_frac, frac)
        linking_length.append(nearest_distance)

    return linking_length


def find_threshold_edge_frac_continuous(sigsig, sigbkg, bkgbkg, edge_frac, do_friend_graph=True):
    """
    # This method is very memory intensive, problem for large samples -> try hist-based function instead.
    Function that finds the threshold on the distance that provides a given edge fraction
    Args:
        sigsig (torch.Tensor): all sigsig connections
        sigbkg (torch.Tensor): all bkgbkg connections
        bkgbkg (torch.Tensor): all sigbkg connections
        edge_frac (float): target edge fraction
        do_friend_graph (bool): friend or enemy graph (default True)
    Returns:
        (list(float)): coordinates of cut point to draw on on ROC curve
        (float): cut to apply to distances
    """
    total_dist = torch.cat([sigsig, sigbkg, bkgbkg])
    total_len = total_dist.numel()

    sorted_dist, _ = torch.sort(total_dist, descending=(not do_friend_graph))

    linking_length = []
    for frac in edge_frac:
        k = int(torch.ceil(torch.tensor(frac * total_len)))
        linking_length.append(sorted_dist[k - 1].item())
    return linking_length
