import numpy as np

def find_threshold(tpr, fpr, eff, cut):
    """
    Function to find the threshold on the distance metric that provides a given sig-sig efficiency.
    Args:
        tpr (numpy.array): of true-positive rates, sig-sig passing cut
        fpr (numpy.array): of false-positive rates, bkg-bkg(sig-bkg) passing cut
        eff (float): sig-sig efficiency desired
        cut (numpy.array): of thresholds considered
    Returns:
        (list(float)): coordinates of cut point to draw on on ROC curve
        (float): cut to apply on distance
    """
    # the tpr here is in reverse order with the discriminant cut, so it's <=
    tpr_index = np.argmax(tpr <= eff)
    return [tpr[tpr_index], fpr[tpr_index]], cut[tpr_index]

