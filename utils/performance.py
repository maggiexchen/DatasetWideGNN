import shap
import matplotlib.pyplot as plt
import utils.misc as misc
import numpy as np
import json
from sklearn.metrics import roc_curve, roc_auc_score

def calc_ROC(sig, bkg, sig_wgt, bkg_wgt, flip=False):
    """
    Function to calculate roc curve

    Args:
        sig (numpy.array): array of signal efficiencies/probabilities for each sig event
        bkg (numpy.array): array of signal efficiencies/probabilities for each bkg event
        sig_wgt (numpy.array): array of sig event weights
        bkg_wgt (numpy.array): array of blg event weights
        flip (bool): flipping the fpr and tpr definitions

    Returns:
        (numpy.array): true positive rate: fraction of sig passing a given threshold
        (numpy.array): false positive rate: fraction of bkg passing a given threshold
        (numpy.array): thresholds
        (float): overall AUC

    """
    if flip:
        y_sig = [0]*len(sig)
        y_bkg = [1]*len(bkg)
    else:
        y_sig = [1]*len(sig)
        y_bkg = [0]*len(bkg)
    x_combined = np.concatenate((sig, bkg))
    y_combined = np.concatenate((y_sig, y_bkg))
    wgt_combined = np.concatenate((sig_wgt, bkg_wgt))

    sklearn_fpr, sklearn_tpr, cut = roc_curve(y_combined, x_combined, sample_weight=wgt_combined)
    auc = roc_auc_score(y_combined, x_combined, sample_weight=wgt_combined)

    # IN THE CASE WHERE MOST SIGSIG DISTANCES ARE SMALLER THAN BKGBKG DISTANCES (E.G. TRSM HHH SIGNALS)
    # fpr here is the fraction of sigsig above a certain cut
    # tpr here is the fraction of sig(bkg)bkg above a certain cut
    # the actual tpr we want is the fraction of sigsig below a certain cut: (1-fnr)
    # and the actual fpr is the fraction of sig(bkg)bkg below a certain cut: (1-tnr)

    if flip:
        true_tpr = 1-sklearn_fpr
        true_fpr = 1-sklearn_tpr
    else:
        true_tpr = sklearn_tpr
        true_fpr = sklearn_fpr

    return true_tpr, true_fpr, cut, auc

def doShap(gcn_model, train_x, kinematics, path):
    """
    Function to calculate SHAP values and create figures plotting them

    Args:
        gcn_model (pytorch.model): the ML model to calculate values for
        train_x (pytorch.tensor): total signal and background input data for network
        kinematics (list(str)): list of kinematic variables to calculate shap values for
        path (str): base path to store output

    Returns:
        void

    """

    explainer = shap.DeepExplainer(gcn_model, train_x)
    shap_values = explainer.shap_values(train_x[:200, :])
    explanation = shap.Explanation(explainer.expected_value, shap_values, feature_names=kinematics)
  
    save_path = path+"/performance/"
    misc.create_dirs(save_path)
  
    plt.figure()
    shap.plots.beeswarm(shap_values)
    shap.summary_plot(shap_values, train_x[:200, :], feature_names=kinematics)
    plt.tight_layout()
    plt.savefig(save_path+"shapley_beeswarm.pdf")

def save_performance(train_loss, train_fpr, train_tpr, train_threshold, train_auc, val_loss, val_fpr, val_tpr, val_threshold, val_auc, path):
    perf_dict = {
        'train_loss': train_loss,
        'train_fpr': train_fpr.tolist(),
        'train_tpr': train_tpr.tolist(),
        'train_threshold': train_threshold.tolist(),
        'train_auc': train_auc.tolist(),
        'val_loss': train_loss,
        'val_fpr': train_fpr.tolist(),
        'val_tpr': train_tpr.tolist(),
        'val_threshold': train_threshold.tolist(),
        'val_auc': val_auc.tolist(),
    }
    save_path = path+"performance.json"
    with open(save_path, "w") as outfile:
        json.dump(perf_dict, outfile)
    
def save_metadata(train_sig_size, train_bkg_size, val_sig_size, val_bkg_size, hidden_sizes_gcn, hidden_sizes_mlp, LR, dropout_rates, epochs, path):
    meta_dict = {
        'train_sig_size': train_sig_size,
        'train_bkg_size': train_bkg_size,
        'val_sig_size': val_sig_size,
        'val_bkg_size': val_bkg_size,
        'hidden_sizes_gcn': hidden_sizes_gcn,
        'hidden_sizes_mlp': hidden_sizes_mlp,
        'LR': LR,
        'dropout_rates': dropout_rates,
        'epochs': epochs
    }
    save_path = path+"metadata.json"
    with open(save_path, "w") as outfile:
        json.dump(meta_dict, outfile)
    
