import argparse
import gzip
import json
import os
import random
import sys
from io import StringIO
from os import listdir
from os.path import isfile, join

import numpy as np
import torch
import torch.utils.data
from pandas import Series

class ExtrasensoryDataset(torch.utils.data.Dataset):
    """
    This is the main class that calculates the spectrogram and returns the
    spectrogram, audio pair.
    """
    def __init__(self, data_folder, split="train", segment_length=20, normalize=True, interpolate=False):
        
        self.compute_statistics(data_folder)
        data_files = [f for f in listdir(data_folder) if isfile(join(data_folder, f))]
        if split == 'train':
            data_files = data_files[:int(7.0/10.0 * len(data_files))]
        elif split == 'valid':
            data_files = data_files[int(7.0/10.0 * len(data_files)):int(8.0/10.0 * len(data_files))]
        else:
            data_files = data_files[int(8.0/10.0 * len(data_files)):]

        features, labels, instances = self.load_files(data_folder, data_files)

        if normalize:
            max_train = np.load(os.path.join(data_folder, 'stats', 'max_train.npy'))
            min_train = np.load(os.path.join(data_folder, 'stats', 'min_train.npy'))

            features = 2 * (features - min_train) / (max_train - min_train) - 1
        
        if interpolate:
            features = np.array([Series(i).interpolate() for i in features.T]).T

        # Convert NaNs
        features[np.isnan(features)] = 0
        
        # num_classes
        self.num_classes = labels.shape[1]

        # Divide based on subjects
        features_divided = []
        labels_divided = []
        for i in np.unique(instances):
            features_i = features[instances == i, :]
            labels_i = labels[instances == i]
            features_divided.append(features_i)
            labels_divided.append(labels_i)

        # Time is not the same length
        features_divided = np.array(features_divided)
        labels_divided = np.array(labels_divided)

        self.segment_length = segment_length

        self.features = features_divided
        self.labels = labels_divided

    def compute_statistics(self, data_folder):
        data_files = [f for f in listdir(data_folder) if isfile(join(data_folder, f))]
        features, labels, _ = self.load_files(data_folder, data_files)

        mean_train = np.nanmean(features, axis=0)
        std_train = np.nanstd(features, axis=0)
        max_train = np.max(features, axis=0)
        min_train = np.min(features, axis=0)

        np.save(os.path.join(data_folder, 'stats', 'mean_train'), mean_train)
        np.save(os.path.join(data_folder, 'stats', 'std_train'), std_train)
        np.save(os.path.join(data_folder, 'stats', 'max_train'), max_train)
        np.save(os.path.join(data_folder, 'stats', 'min_train'), min_train)

    def __getitem__(self, index):
        feature = self.features[index]
        label = self.labels[index]
        
        # Take segment
        if len(feature) >= self.segment_length:
            max_start = len(feature) - self.segment_length
            start = random.randint(0, max_start)
            feature = feature[start : start + self.segment_length, :]
            label = label[start : start + self.segment_length, :]
        else:
            raise ValueError('Segment length too large')

        return (feature, label)

    def load_files(self, data_folder, data_files):
        features = []
        labels = []
        missings = []
        timestamps = []
        instances = []
        for i, data_file in enumerate(data_files):
            X, Y, M, T, _, _ = read_user_data(os.path.join(data_folder, data_file))
            current_instance = i * np.ones(X.shape[0])

            features.append(X)
            labels.append(Y)
            missings.append(M)
            timestamps.append(T)
            instances.append(current_instance)

        features = np.concatenate(features, 0)
        labels = np.concatenate(labels, 0)
        missings = np.concatenate(missings, 0)
        timestamps = np.concatenate(timestamps, 0)
        instances = np.concatenate(instances, 0)

        # Handle zero labels
        labels = labels.astype(int)
        labels[missings == True] = -1

        return features, labels, instances

    def __len__(self):
        return self.features.shape[0]

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

def parse_header_of_csv(csv_str):
    # Isolate the headline columns:
    headline = csv_str[:csv_str.index('\n')]
    columns = headline.split(',')

    # The first column should be timestamp:
    assert columns[0] == 'timestamp'
    # The last column should be label_source:
    assert columns[-1] == 'label_source'
    
    # Search for the column of the first label:
    for (ci,col) in enumerate(columns):
        if col.startswith('label:'):
            first_label_ind = ci
            break 
        pass 

    # Feature columns come after timestamp and before the labels:
    feature_names = columns[1:first_label_ind] 
    # Then come the labels, till the one-before-last column:
    label_names = columns[first_label_ind:-1] 
    for (li,label) in enumerate(label_names):
        # In the CSV the label names appear with prefix 'label:', but we don't need it after reading the data:
        assert label.startswith('label:') 
        label_names[li] = label.replace('label:','') 
        pass 
    
    return (feature_names,label_names) 

def parse_body_of_csv(csv_str,n_features):
    # Read the entire CSV body into a single numeric matrix:
    full_table = np.loadtxt(StringIO(csv_str), delimiter=',', skiprows=1) 
    
    # Timestamp is the primary key for the records (examples):
    timestamps = full_table[:,0].astype(int) 
    
    # Read the sensor features:
    X = full_table[:,1:(n_features+1)] 
    
    # Read the binary label values, and the 'missing label' indicators:
    trinary_labels_mat = full_table[:,(n_features+1):-1]  # This should have values of either 0., 1. or NaN
    M = np.isnan(trinary_labels_mat)  # M is the missing label matrix
    Y = np.where(M,0,trinary_labels_mat) > 0.  # Y is the label matrix
    
    return (X,Y,M,timestamps) 

'''
Read the data (precomputed sensor-features and labels) for a user.
This function assumes the user's data file is present.
'''
def read_user_data(uuid):
    user_data_file = uuid 

    # Read the entire csv file of the user:
    with gzip.open(user_data_file,'rb') as fid:
        csv_str = fid.read() 
        pass 

    csv_str = csv_str.decode("utf-8") 

    (feature_names,label_names) = parse_header_of_csv(csv_str) 
    n_features = len(feature_names) 
    (X,Y,M,timestamps) = parse_body_of_csv(csv_str,n_features) 

    return (X,Y,M,timestamps,feature_names,label_names) 
