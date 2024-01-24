import shap
import matplotlib.pyplot as plt
import utils.misc as misc

def doShap(gcn_model, train_x, kinematics, file_path):

  explainer = shap.DeepExplainer(gcn_model, train_x)
  shap_values = explainer.shap_values(train_x[:200, :])
  explanation = shap.Explanation(explainer.expected_value, shap_values, feature_names=kinematics)

  save_path = file_path+"/performance/"
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
  fig.savefig(save_path+"mass_test_pred_x_standardised_dummy.pdf", transparent=True)

