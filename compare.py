import matplotlib.pyplot as plt
import mplhep as hep
import json
import numpy
import utils.misc as misc

path = "/data/atlas/atlasdata3/maggiechen/gnn_project/models/"
models = ["dnn", "gcn_D_inv", "gcn_D_inv_self", "gcn_D_half_inv", "gcn_D_half_inv_self", "gcn_D_frac", "gcn_D_frac_self"]

out_path = "/data/atlas/atlasdata3/maggiechen/gnn_project/plots/compare_models/"
misc.create_dirs(out_path)

# plot ROC curve comparison
fig, ax = plt.subplots()
for model in models:
    perf_path = path+model+"/performance.json"
    with open(perf_path, 'r') as perf_file:
        perf_dict = json.load(perf_file)
        val_fpr = perf_dict['val_fpr']
        val_tpr = perf_dict['val_tpr']
        val_auc = perf_dict['val_auc']
    plt.plot(val_fpr, val_tpr, label=model+' (AUC = {:.3f})'.format(val_auc))
plt.legend(loc="lower right", fontsize=9)
plt.xlim(0,1)
plt.xlabel("Background Efficiency", loc="right")
plt.ylabel("Signal Efficiency", loc="top")
fig.savefig(out_path+"validation_ROC_compare.pdf", transparent=True)

# plot training loss
fig, ax = plt.subplots()
for model in models:
    perf_path = path+model+"/performance.json"
    with open(perf_path, 'r') as perf_file:
        perf_dict=json.load(perf_file)
        train_loss = perf_dict["train_loss"]
    x_epoch = numpy.arange(1,len(train_loss)+1,1)
    plt.plot(x_epoch, train_loss, label=model)
plt.legend(loc="upper right", fontsize=9)
plt.xlabel("Epoch", loc="right")
plt.ylabel("Training loss", loc="top")
fig.savefig(out_path+"train_loss_compare.pdf", transparent=True)

# plot validation loss
fig, ax = plt.subplots()
for model in models:
    perf_path = path+model+"/performance.json"
    with open(perf_path, 'r') as perf_file:
        perf_dict=json.load(perf_file)
        val_loss = perf_dict["val_loss"]
    x_epoch = numpy.arange(1,len(val_loss)+1,1)
    plt.plot(x_epoch, val_loss, label=model)
plt.legend(loc="upper right", fontsize=9)
plt.xlabel("Epoch", loc="right")
plt.ylabel("Validation loss", loc="top")
fig.savefig(out_path+"val_loss_compare.pdf", transparent=True)