import shap
import matplotlib.pyplot as plt
import utils.misc as misc
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score

# fpr here is the fraction of sigsig above a certain cut
# tpr here is the fraction of sig(bkg)bkg above a certain cut
# the actual tpr we want is the fraction of sigsig below a certain cut: (1-fpr)
# and the actual fpr is the fraction of sig(bkg)bkg below a certain cut: (1-tpr)

def calc_ROC(sig, bkg, sig_wgt, bkg_wgt):
    """
    Function to calculate roc curve

    Args:
        sig (numpy.array): array of signal efficiencies/probabilities for each sig event
        bkg (numpy.array): array of signal efficiencies/probabilities for each bkg event
        sig_wgt (numpy.array): array of sig event weights
        bkg_wgt (numpy.array): array of blg event weights

    Returns:
        (numpy.array): true positive rate: fraction of sig passing a given threshold
        (numpy.array): false positive rate: fraction of bkg passing a given threshold
        (numpy.array): thresholds
        (float): overall AUC

    """

    y_sig = [0]*len(sig)
    y_bkg = [1]*len(bkg)
    x_combined = np.concatenate((sig, bkg))
    y_combined = np.concatenate((y_sig, y_bkg))
    wgt_combined = np.concatenate((sig_wgt, bkg_wgt))

    fpr, tpr, cut = roc_curve(y_combined, x_combined, sample_weight=wgt_combined)
    auc = roc_auc_score(y_combined, x_combined, sample_weight=wgt_combined)

    # fpr here is the fraction of sigsig above a certain cut
    # tpr here is the fraction of sig(bkg)bkg above a certain cut
    # the actual tpr we want is the fraction of sigsig below a certain cut: (1-fpr)
    # and the actual fpr is the fraction of sig(bkg)bkg below a certain cut: (1-tpr)

    true_tpr = 1-fpr
    true_fpr = 1-tpr

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
    plt.savefig(save_path+"mass_euclidean_x+conv_x_shapley_beeswarm.pdf")
  
    ax.legend(loc='upper right')
    ax.set_xlabel("GNN Score", loc="right")
    ax.set_ylabel("Normalised # events / bin", loc="top")
    fig.savefig(save_path+"mass_test_pred_x_standardised.pdf", transparent=True)

