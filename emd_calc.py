import numpy as np
import pandas as pd
import energyflow as ef
# import pdb

df = pd.read_hdf('/home/srutherford/GNN_shared/hhhgraph/data/LQ_my_files/MetCut200/LQ_1.h5')
df_ttbar = pd.read_hdf('/home/srutherford/GNN_shared/hhhgraph/data/LQ_my_files/MetCut200/ttbar_1.h5')
df_singletop = pd.read_hdf('/home/srutherford/GNN_shared/hhhgraph/data/LQ_my_files/MetCut200/singletop_1.h5')

# Convert dataframe into EMD-compatible event representations
def get_event_vectors(df):
    events = []
    weights = []
    for _, row in df.iterrows():
        event = np.array([
            [row['bjet1pt'], row['bjet1eta'], row['bjet1phi']],
            [row['bjet2pt'], row['bjet2eta'], row['bjet2phi']],
            [row['lep1pt'], row['lep1eta'], row['lep1phi']],
            [row['lep2pt'], row['lep2eta'], row['lep2phi']],
            [row['met'], 0.0, row['metphi']] ### treat met as pseudo particle with eta=0
        ])
        events.append(event)
        weights.append(row['eventWeight'])
    return events, np.array(weights)

# Convert DataFrame to EMD-compatible format
events, weights_LQ = get_event_vectors(df)
events_ttbar, weights_ttbar = get_event_vectors(df_ttbar)
events_singletop, weights_singletop = get_event_vectors(df_singletop)
weights_bkg = np.concatenate([weights_ttbar, weights_singletop])

# Compute pairwise EMDs
emd_sigsig = ef.emd.emds(events, R=1.0, gdim=2, n_jobs=-1)
emd_sigbkg = ef.emd.emds(events, events_ttbar + events_singletop, R=1.0, gdim=2, n_jobs=-1)
emd_bkgbkg = ef.emd.emds(events_ttbar + events_singletop, R=1.0, gdim=2, n_jobs=-1)

### Flatten matrices to 1D arrays
emd_sigsig = emd_sigsig.flatten()
emd_sigbkg = emd_sigbkg.flatten()
emd_bkgbkg = emd_bkgbkg.flatten()

# Compute histogram weights as the product of event weights
weights_LQ_LQ_hist = np.outer(weights_LQ, weights_LQ).flatten()
weights_LQ_bkg_hist = np.outer(weights_LQ, weights_bkg).flatten()
weights_bkg_bkg_hist = np.outer(weights_bkg, weights_bkg).flatten()

### plot histograms (density normalised)
import matplotlib.pyplot as plt
plt.figure()
plt.hist(emd_sigsig, bins=100, alpha=0.5, label='Signal-Signal', density=True,  weights=weights_LQ_LQ_hist)
plt.hist(emd_sigbkg, bins=100, alpha=0.5, label='Signal-Background', density=True, weights=weights_LQ_bkg_hist)
plt.hist(emd_bkgbkg, bins=100, alpha=0.5, label='Background-Background', density=True, weights=weights_bkg_bkg_hist)
plt.legend()

plt.xlabel('EMD')
plt.ylabel('Density')
plt.show()

# Print or store the result
# print(emd_matrix)

