import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
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
        self.pairs = self.generate_pairs()

    def generate_pairs(self):
        # Choose num_sig_samples and num_bkg_samples from the events
        sig_idx = torch.randperm(len(self.signal_events))[:self.num_sig_samples]
        bkg_idx = torch.randperm(len(self.background_events))[:self.num_bkg_samples]

        sampled_sig_ind = torch.arange(self.num_sig_samples)
        sampled_bkg_ind = torch.arange(self.num_bkg_samples)
        cartesian_product_sigbkg_ind = torch.cartesian_prod(sampled_sig_ind, sampled_bkg_ind)
        cartesian_product_sigsig_ind = torch.cartesian_prod(sampled_sig_ind, sampled_sig_ind)

        pairs = []

        for idx1, idx2 in cartesian_product_sigsig_ind:
            pairs.append((self.signal_events[sig_idx[idx1]], self.signal_events[sig_idx[idx2]], 1))
        
        for idx1, idx2 in cartesian_product_sigbkg_ind:
            pairs.append((self.signal_events[sig_idx[idx1]], self.background_events[bkg_idx[idx2]], 0))

        return pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        signal_event, background_event, label = self.pairs[idx]
        return signal_event, background_event, label

"""

class PairDataset(Dataset):
    def __init__(self, signal_events, background_events, num_samples, num_negatives_per_positive):
        self.signal_events = signal_events
        self.background_events = background_events
        self.num_samples = num_samples
        self.num_negatives_per_positive = num_negatives_per_positive

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Choose a random signal event
        signal_idx = random.randint(0, len(self.signal_events) - 1)
        signal_event = self.signal_events[signal_idx]

        # Positive pair: another signal event
        positive_idx = random.randint(0, len(self.signal_events) - 1)
        positive_event = self.signal_events[positive_idx]
        positive_label = 1

        # Negative pairs: background events
        negative_events = []
        for _ in range(self.num_negatives_per_positive):
            negative_idx = random.randint(0, len(self.background_events) - 1)
            negative_event = self.background_events[negative_idx]
            negative_events.append((signal_event, negative_event, 0))

        # Create the sample
        sample = [(signal_event, positive_event, positive_label)] + negative_events
        return sample

"""