import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import os
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
import utils.misc as misc
import random

class PairDataset(Dataset):
    def __init__(self, signal_events, background_events, num_sig_samples, num_bkg_samples):
        self.signal_events = signal_events
        self.background_events = background_events
        self.num_sig_samples = num_sig_samples
        self.num_bkg_samples = num_bkg_samples

        # Assert that the number of features match
        assert signal_events.shape[1] == background_events.shape[1], "Signal and background events do not have the same number of features!"
        self.num_features = signal_events.shape[1]

        # Precompute pairs
        self.pairs, self.means, self.stds = self.generate_pairs()

    def generate_pairs(self):
        # Choose num_sig_samples and num_bkg_samples from the events
        sig_idx = torch.randperm(len(self.signal_events))[:self.num_sig_samples]
        bkg_idx = torch.randperm(len(self.background_events))[:self.num_bkg_samples]

        sampled_sig_ind = torch.arange(self.num_sig_samples)
        sampled_bkg_ind = torch.arange(self.num_bkg_samples)

        # standardising the sampled signal and background events
        sampled_full_x = pd.DataFrame(torch.cat((self.signal_events[sampled_sig_ind], self.background_events[sampled_bkg_ind])).numpy())
        means, stds = misc.get_train_mean_std(sampled_full_x)
        # then standardise the full signal and background samples
        self.signal_events = (self.signal_events - torch.Tensor(means))/torch.Tensor(stds)
        self.background_events = (self.background_events - torch.Tensor(means))/torch.Tensor(stds)

        cartesian_product_sigsig_ind = torch.cartesian_prod(sampled_sig_ind, sampled_sig_ind)
        cartesian_product_bkgbkg_ind = torch.cartesian_prod(sampled_bkg_ind, sampled_bkg_ind)
        cartesian_product_sigbkg_ind = torch.cartesian_prod(sampled_sig_ind, sampled_bkg_ind)
        pairs = []

        for idx1, idx2 in cartesian_product_sigsig_ind:
            pairs.append((self.signal_events[sig_idx[idx1]], self.signal_events[sig_idx[idx2]], 1))

        for idx1, idx2 in cartesian_product_bkgbkg_ind:
            pairs.append((self.background_events[bkg_idx[idx1]], self.background_events[bkg_idx[idx2]], 1))

        for idx1, idx2 in cartesian_product_sigbkg_ind:
            pairs.append((self.signal_events[sig_idx[idx1]], self.background_events[bkg_idx[idx2]], 0))

        return pairs, torch.Tensor(means), torch.Tensor(stds)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        signal_event, background_event, label = self.pairs[idx]
        means = self.means
        stds = self.stds
        return signal_event, background_event, label, means, stds