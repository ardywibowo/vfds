import numpy as np
import argparse
import json
import os
import random
import torch
import torch.utils.data
import sys
from pandas import Series

class OpportunityDataset(torch.utils.data.Dataset):
    """
    This is the main class that calculates the spectrogram and returns the
    spectrogram, audio pair.
    """
    def __init__(self, data_folder, split="train", segment_length=20, normalize=True, 
                    interpolate=False, remove_features=False, label_type=6):
        
        self.segment_length = segment_length
        self.split = split

        data_files = ['S1-ADL1.dat', 'S1-ADL2.dat', 'S1-ADL3.dat', 'S1-ADL4.dat', 'S1-ADL5.dat', 
                        'S2-ADL1.dat', 'S2-ADL2.dat', 'S2-ADL3.dat', 'S2-ADL4.dat', 'S2-ADL5.dat', 
                        'S3-ADL1.dat', 'S3-ADL2.dat', 'S3-ADL3.dat', 'S3-ADL4.dat', 'S3-ADL5.dat', 
                        'S4-ADL1.dat', 'S4-ADL2.dat', 'S4-ADL3.dat', 'S4-ADL4.dat', 'S4-ADL5.dat',
                        'S1-Drill.dat', 'S2-Drill.dat', 'S3-Drill.dat', 'S4-Drill.dat']
        
        features, labels, instances = self.load_files(data_folder, data_files)

        if remove_features:
            features_delete = np.arange(45, 49)
            features_delete = np.concatenate([features_delete, np.arange(58, 62)])
            features_delete = np.concatenate([features_delete, np.arange(71, 75)])
            features_delete = np.concatenate([features_delete, np.arange(84, 88)])
            features_delete = np.concatenate([features_delete, np.arange(97, 101)])
            features_delete = np.concatenate([features_delete, np.arange(133, 242)])

            features = np.delete(features, features_delete, 1)

        if interpolate:
            features = np.array([Series(i).interpolate() for i in features.T]).T

        # Convert NaNs
        features[np.isnan(features)] = 0

        # Find unique labels
        unique_labels = []
        for i in range(labels.shape[1]):
            curr_unique = np.unique(labels[:, i])
            # curr_unique = np.delete(curr_unique, 0) # Delete NaN label
            unique_labels.append(curr_unique)

        # Convert labels to 0 -- num_classes
        self.num_classes = []
        for i, label in enumerate(unique_labels):
            # Count classes
            num_class = len(label) # num_classes
            self.num_classes.append(num_class)

            # Convert labels
            for j, l in enumerate(label):
                labels[labels[:, i] == l, i] = j
        self.num_classes = np.array(self.num_classes)

        # Divide based on subjects
        features_divided = []
        labels_divided = []
        for i in np.unique(instances):
            features_i = features[instances == i, :]
            labels_i = labels[instances == i]
            features_divided.append(features_i)
            labels_divided.append(labels_i)

        # Time is not the same length
        self.features = np.array(features_divided)
        self.labels = np.array(labels_divided)

        # Select label type
        features_divided = []
        labels_divided = []
        for features_segment, labels_segment in zip(self.features, self.labels):
            current_labels = labels_segment[:, label_type]
            labelled_times = (current_labels != -1)

            run_values, run_starts, run_lengths = find_runs(labelled_times)
            run_starts = run_starts[run_values == 1]
            run_lengths = run_lengths[run_values == 1]

            for run_start, run_length in zip(run_starts, run_lengths):
                # Add some nonlabels and adjust segment lengths
                run_start = max(0, run_start - 20)
                run_length = run_length + 20
                if run_length < self.segment_length:
                    run_length = self.segment_length
                    if run_start + run_length > features_segment.shape[0]:
                        run_start = features_segment.shape[0] - run_length
                
                current_features = features_segment[run_start : run_start+run_length, :]
                current_labels = labels_segment[run_start : run_start+run_length, label_type]

                features_divided.append(current_features)
                labels_divided.append(current_labels)
        
        features_divided = np.array(features_divided)
        labels_divided = np.array(labels_divided)

        features_all = np.vstack(features_divided)

        if normalize:
            min_train = np.nanmin(features_all, axis=0)
            max_train = np.nanmax(features_all, axis=0)

            useless = (max_train == min_train)
            for i in range(features_divided.shape[0]):
                features_divided[i] = features_divided[i][:, ~useless]
                features_divided[i] = 2 * (features_divided[i] - min_train) \
                    / (max_train - min_train) - 1

        np.random.seed(42)
        rng_state = np.random.get_state()
        np.random.shuffle(features_divided)
        np.random.set_state(rng_state)
        np.random.shuffle(labels_divided)

        total_samples = features_divided.shape[0]

        self.features_train = features_divided[:int(8./10. * total_samples)]
        self.labels_train = labels_divided[:int(8./10. * total_samples)]

        self.features_valid = features_divided[int(8./10. * total_samples) : \
            int(9./10. * total_samples)]
        self.labels_valid = labels_divided[int(8./10. * total_samples) : \
            int(9./10. * total_samples)]

        self.features_test = features_divided[int(9./10. * total_samples):]
        self.labels_test = labels_divided[int(9./10. * total_samples):]

        if split == 'train':
            self.selected_features = self.features_train
            self.selected_labels = self.labels_train
        elif split == 'valid':
            self.selected_features = self.features_valid
            self.selected_labels = self.labels_valid
        else:
            self.selected_features = self.features_test
            self.selected_labels = self.labels_test

    def select_set(self, split):
        self.split = split
        if split == 'train':
            self.selected_features = self.features_train
            self.selected_labels = self.labels_train
        elif split == 'valid':
            self.selected_features = self.features_valid
            self.selected_labels = self.labels_valid
        else:
            self.selected_features = self.features_test
            self.selected_labels = self.labels_test

    def __getitem__(self, index):
        feature = self.selected_features[index]
        label = self.selected_labels[index]
        
        # Take segment
        if len(feature) >= self.segment_length:
            max_start = len(feature) - self.segment_length
            start = random.randint(0, max_start)
            feature = feature[start : start + self.segment_length, :]
            label = label[start : start + self.segment_length]
        else:
            raise ValueError('Segment length too large')

        return (feature, label)

    def load_files(self, data_folder, data_files):
        all_data = []
        instances = []
        for i, data_file in enumerate(data_files):
            current_data = np.loadtxt(os.path.join(data_folder, 'dataset', data_file))
            current_instance = i * np.ones(current_data.shape[0])
            all_data.append(current_data)
            instances.append(current_instance)

        all_data = np.concatenate(all_data, 0)
        instances = np.concatenate(instances, 0)

        features = all_data[:, 1:243]
        labels = all_data[:, 243:]

        # Cut all zero ends
        zeros = np.zeros(labels.shape[1])
        labelled_times = np.where((labels != zeros).all(axis=1))[0]
        first_label = labelled_times[0]
        last_label = labelled_times[-1]

        features = features[first_label:last_label+1, :]
        labels = labels[first_label:last_label+1, :]
        instances = instances[first_label:last_label+1]

        return features, labels, instances

    def __len__(self):
        return self.selected_features.shape[0]

def find_runs(x):
    """Find runs of consecutive items in an array."""

    # ensure array
    x = np.asanyarray(x)
    if x.ndim != 1:
        raise ValueError('only 1D array supported')
    n = x.shape[0]

    # handle empty array
    if n == 0:
        return np.array([]), np.array([]), np.array([])

    else:
        # find run starts
        loc_run_start = np.empty(n, dtype=bool)
        loc_run_start[0] = True
        np.not_equal(x[:-1], x[1:], out=loc_run_start[1:])
        run_starts = np.nonzero(loc_run_start)[0]

        # find run values
        run_values = x[loc_run_start]

        # find run lengths
        run_lengths = np.diff(np.append(run_starts, n))

        return run_values, run_starts, run_lengths
