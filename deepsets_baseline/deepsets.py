import tensorflow as tf
import os
import numpy
import pandas as pd
import h5py
import matplotlib.pyplot as plt
import matplotlib.figure as figure
import mplhep as hep
import tensorflow.keras as keras
from tensorflow.keras import layers, Input, Model
import tensorflow.keras.losses as losses
import tensorflow.keras.layers as layers
from numpy.lib.recfunctions import structured_to_unstructured
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, auc
from Sum import Sum
#import dgl
#from dgl.nn import EGATConv

# define the input features
event_features = ["NPV", "sphere3dv2b", "aplan3dv2b", "mHHH",
                  "meandRBB", "rmsdRBB", "skewnessdRBB", "kurtosisdRBB"]
jet_features = ["jets_pt", "jets_eta", "jets_phi", "jets_E"]
misc_features = ["eventWeight", "eventNumber", "mX"]
MAXJETS = 10
LR = 0.01
MASKVAL = -999
BATCHSIZE=128
nEpochs=30

# load in the training dataset
fsignal = h5py.File("/data/atlas/atlasdata3/maggiechen/gnn_project/train_files/resonant_TRSM_6b_signal_new.h5", "r")
fbackground = h5py.File("/data/atlas/atlasdata3/maggiechen/gnn_project/train_files/5b_data_background_new.h5", "r")

# define jet-level features (nodes) and event-level features
sig_event_feats = fsignal["events"]
sig_jet_feats = fsignal["jets"]
bkg_event_feats = fbackground["events"]
bkg_jet_feats = fbackground["jets"]

# h5py to python arrays
sig_jet_feats = structured_to_unstructured(sig_jet_feats.fields(jet_features)[:])
sig_event_feats = structured_to_unstructured(sig_event_feats.fields(event_features)[:])
bkg_jet_feats = structured_to_unstructured(bkg_jet_feats.fields(jet_features)[:])
bkg_event_feats = structured_to_unstructured(bkg_event_feats.fields(event_features)[:])

print(sig_event_feats)
# labels for supervised learning
sig_labels = numpy.array([1]*len(sig_event_feats))
bkg_labels = numpy.array([0]*len(bkg_event_feats))

jet_sig_train, jet_sig_valid, y_sig_train, y_sig_valid = train_test_split(sig_jet_feats, sig_labels, test_size=0.3, shuffle= True)
jet_bkg_train, jet_bkg_valid, y_bkg_train, y_bkg_valid = train_test_split(bkg_jet_feats, bkg_labels, test_size=0.3, shuffle= True)
jet_sig_train[numpy.isnan(jet_sig_train)] = MASKVAL
jet_sig_valid[numpy.isnan(jet_sig_valid)] = MASKVAL
jet_bkg_train[numpy.isnan(jet_bkg_train)] = MASKVAL
jet_bkg_valid[numpy.isnan(jet_bkg_valid)] = MASKVAL
jet_train = numpy.concatenate((jet_sig_train, jet_bkg_train), axis=0)
jet_valid = numpy.concatenate((jet_sig_valid, jet_bkg_valid), axis=0)
y_train = numpy.concatenate((y_sig_train, y_bkg_train))
y_valid = numpy.concatenate((y_sig_valid, y_bkg_valid))


jetlayers = [ 32 , 32 , 32 , 32 ]
evtlayers = [ 64 , 64 , 64 , 64 ]

def buildModel(jlayers, elayers):
  inputs = layers.Input(shape=(None, len(jet_features)))

  outputs = inputs
  outputs = layers.Masking(mask_value=MASKVAL)(outputs)

  for nodes in jlayers[:-1]:
    outputs = layers.TimeDistributed(layers.Dense(nodes, activation='relu'))(outputs)
    outputs = layers.BatchNormalization()(outputs)

  outputs = layers.TimeDistributed(layers.Dense(jlayers[-1], activation='softmax'))(outputs)
  outputs = Sum()(outputs)

  for nodes in elayers:
    outputs = layers.Dense(nodes, activation='relu')(outputs)
    outputs = layers.BatchNormalization()(outputs)

  outclass = layers.Dense(1, activation='sigmoid')(outputs)

  return keras.Model(inputs=inputs, outputs=outclass)

model = buildModel([len(jet_features)] + jetlayers, evtlayers)
model.summary()

# compile model
model.compile(optimizer=keras.optimizers.Adam(learning_rate=LR), loss=losses.BinaryCrossentropy(), metrics=['accuracy','AUC',])

# train model

history = model.fit(jet_train, y_train, batch_size=BATCHSIZE, epochs=nEpochs, validation_data=(jet_valid, y_valid))

# convert the history.history dict to a pandas DataFrame:     
hist_df = pd.DataFrame(history.history) 

model_path = "deepsets_model/"
os.makedirs(model_path, exist_ok=True)

# save to json:
hist_json_file = model_path + 'history.json' 
with open(hist_json_file, mode='w') as f:
    hist_df.to_json(f)

# plotting train/val loss
fig = plt.figure(figsize=(15,12))
plt = fig.add_subplot(111)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.set_ylabel('Loss')
plt.set_xlabel('Epoch')
plt.legend(['train', 'validation'], loc='upper left')
name = model_path + "TrainValLossPlot.pdf"
fig.savefig(name)

y_sig_train_pred = model.predict(jet_sig_train)
y_bkg_train_pred = model.predict(jet_bkg_train)
y_sig_valid_pred = model.predict(jet_sig_valid)
y_bkg_valid_pred = model.predict(jet_bkg_valid)

import matplotlib.pyplot as plt
fig = plt.figure(figsize=(15,12))
plt.style.use(hep.style.ROOT)
hep.atlas.text(text='Internal', loc=1, fontsize=20)
hep.atlas.text(text=r'$\sqrt{s}$=13 TeV, 5b Data'+'\n'+'6b Resonant TRSM signal', loc=2, fontsize=20)
plt.hist(y_sig_train_pred, bins=50, histtype="step", density=True, linestyle='--', label="6b resonant TRSM (training)", color="steelblue")
plt.hist(y_sig_valid_pred, bins=50, histtype="step", density=True, label="6b resonant TRSM (validation)", color="steelblue")
plt.hist(y_bkg_train_pred, bins=50, histtype="step", density=True, linestyle='--', label="5b data (training)", color="darkorange")
plt.hist(y_bkg_valid_pred, bins=50, histtype="step", density=True, label="5b data (validation)", color="darkorange")
plt.legend()
ymin, ymax = plt.ylim()
plt.ylim(ymin, ymax*1.2)
plt.xlabel("NN Prediction")
plt.ylabel("No. events")
plot_name = model_path + "deepsets_pred.pdf"
fig.savefig(plot_name, transparent=True)

y_pred_train = model.predict(jet_train)
y_pred_valid = model.predict(jet_valid)

# signal eff, false postivie, and true positive rates for roc curves
fpr_train, tpr_train, sig_eff_train = roc_curve(y_train, y_pred_train)
fpr_val, tpr_val, sig_eff_val = roc_curve(y_valid, y_pred_valid)
roc_auc_train = roc_auc_score(y_train, y_pred_train)
roc_auc_val = roc_auc_score(y_valid, y_pred_valid)

# precision recall and auc
prec_train, rec_train, _ = precision_recall_curve(y_train, y_pred_train)
prec_val, rec_val, _ =precision_recall_curve(y_valid, y_pred_valid)
pr_auc_train = auc(y_train, y_pred_train)
pr_auc_val = auc(y_valid, y_pred_valid)

# plot roc curve
fig = plt.figure(figsize=(15,12))
plt.style.use(hep.style.ROOT)
hep.atlas.text(text='Internal', loc=1, fontsize=20)
hep.atlas.text(text='6b resonant TRSM signal MC, 5b Data', loc=2, fontsize=20)
plt.plot(fpr_train, tpr_train, label='Training ROC curve (AUC = {:.3f})'.format(roc_auc_train))
plt.plot(fpr_val, tpr_val, label='Validation ROC curve (AUC = {:.3f})'.format(roc_auc_val))
plt.legend()
ymin, ymax = plt.ylim()
plt.ylim(ymin, ymax*1.2)
plt.xlim(0,1)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plot_name = model_path+"ROC.pdf"
fig.savefig(plot_name, transparent=True)

# plot precision recall curve
fig = plt.figure(figsize=(15,12))
plt.style.use(hep.style.ROOT)
hep.atlas.text(text='Internal', loc=1, fontsize=20)
hep.atlas.text(text='6b resonant TRSM signal MC, 5b Data', loc=2, fontsize=20)
plt.plot(rec_train, prec_train, label='Training precision recall curve (AUC = {:.3f})'.format(pr_auc_train))
plt.plot(rec_val, prec_val, label='Validation precision recall curve (AUC = {:.3f})'.format(pr_auc_val))
plt.legend()
ymin, ymax = plt.ylim()
plt.ylim(ymin, ymax*1.2)
plt.xlim(0,1)
plt.xlabel("Recall")
plt.ylabel("Precision")
plot_name = model_path+"Precision_recall.pdf"
fig.savefig(plot_name, transparent=True)