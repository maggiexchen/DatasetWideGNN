import numpy as np
import pandas as pd
import torch
from torch.utils.data import random_split, Dataset, DataLoader
# torch.manual_seed(42)

import os
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
import utils.misc as misc
import random
class PairDataset(Dataset):
    def __init__(self, signal_events, background_events, num_sig_samples, num_bkg_samples, standardise=True):
        self.signal_events = signal_events
        self.background_events = background_events
        self.num_sig_samples = num_sig_samples
        self.num_bkg_samples = num_bkg_samples
        self.standardise = standardise

        # Assert that the number of features match
        assert signal_events.shape[1] == background_events.shape[1], "Signal and background events do not have the same number of features!"
        self.num_features = signal_events.shape[1]

        # Precompute pairs
        if self.standardise:
            self.train_pairs, self.val_pairs, self.means, self.stds = self.generate_pairs()
        else:
            self.train_pairs, self, val_pairs = self.generate_pairs()
            self.means = None
            self.stds = None

    def generate_pairs(self):
        # Choose num_sig_samples and num_bkg_samples from the events
        sig_idx = torch.randperm(len(self.signal_events))
        bkg_idx = torch.randperm(len(self.background_events))

        sampled_train_sig_ind = sig_idx[:self.num_sig_samples]
        sampled_train_bkg_ind = bkg_idx[:self.num_bkg_samples]
        sampled_val_sig_ind = sig_idx[-int(self.num_sig_samples/2):]
        sampled_val_bkg_ind = bkg_idx[-int(self.num_bkg_samples/2):]

        # standardising the sampled signal and background events
        if self.standardise:
            print("Standardising ...")
            sampled_full_x = pd.DataFrame(torch.cat((self.signal_events[sampled_train_sig_ind], self.background_events[sampled_train_bkg_ind])).numpy())
            means, stds = misc.get_train_mean_std(sampled_full_x)
            # then standardise the full signal and background samples
            self.signal_events = (self.signal_events - torch.Tensor(means))/torch.Tensor(stds)
            self.background_events = (self.background_events - torch.Tensor(means))/torch.Tensor(stds)
        else:
            means, stds = None, None

        print("Generating pairs ...")
        train_cartesian_product_sigsig_ind = torch.cartesian_prod(sampled_train_sig_ind, sampled_train_sig_ind)
        train_cartesian_product_bkgbkg_ind = torch.cartesian_prod(sampled_train_bkg_ind, sampled_train_bkg_ind)
        train_cartesian_product_sigbkg_ind = torch.cartesian_prod(sampled_train_sig_ind, sampled_train_bkg_ind)
        print("Train pairs", len(train_cartesian_product_sigsig_ind), len(train_cartesian_product_bkgbkg_ind), len(train_cartesian_product_sigbkg_ind))

        val_cartesian_product_sigsig_ind = torch.cartesian_prod(sampled_val_sig_ind, sampled_val_sig_ind)
        val_cartesian_product_bkgbkg_ind = torch.cartesian_prod(sampled_val_bkg_ind, sampled_val_bkg_ind)
        val_cartesian_product_sigbkg_ind = torch.cartesian_prod(sampled_val_sig_ind, sampled_val_bkg_ind)
        print("Val pairs", len(val_cartesian_product_sigsig_ind), len(val_cartesian_product_bkgbkg_ind), len(val_cartesian_product_sigbkg_ind))

        train_pairs = []
        val_pairs = []

        for idx1, idx2 in train_cartesian_product_sigsig_ind:
            train_pairs.append((self.signal_events[sig_idx[idx1]], self.signal_events[sig_idx[idx2]], 1))

        for idx1, idx2 in train_cartesian_product_bkgbkg_ind:
            train_pairs.append((self.background_events[bkg_idx[idx1]], self.background_events[bkg_idx[idx2]], 1))

        for idx1, idx2 in train_cartesian_product_sigbkg_ind:
            train_pairs.append((self.signal_events[sig_idx[idx1]], self.background_events[bkg_idx[idx2]], 0))
        
        for idx1, idx2 in val_cartesian_product_sigsig_ind:
            val_pairs.append((self.signal_events[sig_idx[idx1]], self.signal_events[sig_idx[idx2]], 1))

        for idx1, idx2 in val_cartesian_product_bkgbkg_ind:
            val_pairs.append((self.background_events[bkg_idx[idx1]], self.background_events[bkg_idx[idx2]], 1))

        for idx1, idx2 in val_cartesian_product_sigbkg_ind:
            val_pairs.append((self.signal_events[sig_idx[idx1]], self.background_events[bkg_idx[idx2]], 0))

        print("Finished appending pairs ...")
        if self.standardise:
            return train_pairs, val_pairs, torch.Tensor(means), torch.Tensor(stds)
        else:
            return train_pairs, val_pairs

    def __len__(self):
        return len(self.train_pairs), len(self.val_pairs)

    def __getitem__(self, idx):
        train_signal_event, train_background_event, train_label = self.train_pairs[idx]
        val_signal_event, val_background_event, val_label = self.val_pairs[idx]
        means = self.means
        stds = self.stds
        return train_signal_event, train_background_event, val_signal_event, val_background_event, train_label, val_label, means, stds