import numpy as np
import argparse
import json
import os
import random
import torch
import torch.utils.data
import sys

class UciHarDataset(torch.utils.data.Dataset):
    """
    This is the main class that calculates the spectrogram and returns the
    spectrogram, audio pair.
    """
    def __init__(self, data_folder, split='train', segment_length=20, normalize=True):
        
        if split == 'train':
            features = np.loadtxt(data_folder + "train/X_train.txt")
            labels = np.loadtxt(data_folder + "train/y_train.txt")
            subjects = np.loadtxt(data_folder + "train/subject_train.txt")

            features = features[subjects < max(subjects)-2, :]
            labels = labels[subjects < max(subjects)-2]
            subjects = subjects[subjects < max(subjects)-2]

            if normalize:
                mean_train = np.mean(features, axis=0)
                std_train = np.std(features, axis=0)

                features = (features - mean_train) / std_train
        elif split == 'valid':
            features = np.loadtxt(data_folder + "train/X_train.txt")
            labels = np.loadtxt(data_folder + "train/y_train.txt")
            subjects = np.loadtxt(data_folder + "train/subject_train.txt")

            features = features[subjects >= max(subjects)-2, :]
            labels = labels[subjects >= max(subjects)-2]
            subjects = subjects[subjects >= max(subjects)-2]

            if normalize:
                features_train = np.loadtxt(data_folder + "train/X_train.txt")
                mean_train = np.mean(features_train, axis=0)
                std_train = np.std(features_train, axis=0)

                features = (features - mean_train) / std_train
        elif split == 'test':
            features = np.loadtxt(data_folder + "test/X_test.txt")
            labels = np.loadtxt(data_folder + "test/y_test.txt")
            subjects = np.loadtxt(data_folder + "test/subject_test.txt")

            if normalize:
                features_train = np.loadtxt(data_folder + "train/X_train.txt")
                mean_train = np.mean(features_train, axis=0)
                std_train = np.std(features_train, axis=0)

                features = (features - mean_train) / std_train
        
        self.num_classes = int(np.max(labels))
        labels = labels - 1 # convert from 1 -- C to 0 -- C-1

        # Divide based on subjects
        features_divided = []
        labels_divided = []
        for i in np.unique(subjects):
            features_i = features[subjects == i, :]
            labels_i = labels[subjects == i]
            features_divided.append(features_i)
            labels_divided.append(labels_i)

        # Time is not the same length
        features_divided = np.array(features_divided)
        labels_divided = np.array(labels_divided)

        self.segment_length = segment_length

        self.features = features_divided
        self.labels = labels_divided

    def __getitem__(self, index):
        feature = self.features[index]
        label = self.labels[index]
        
        # Take segment
        if len(feature) >= self.segment_length:
            max_start = len(feature) - self.segment_length
            start = random.randint(0, max_start)
            feature = feature[start : start + self.segment_length, :]
            label = label[start : start + self.segment_length]
        else:
            raise ValueError('Segment length too large')
            # feature = torch.nn.functional.pad(feature, (0, self.segment_length - len(feature)), 'constant').data
            # label = torch.nn.functional.pad(label, (0, self.segment_length - len(feature)), 'constant').data

        return (feature, label)
    
    def __len__(self):
        return self.features.shape[0]