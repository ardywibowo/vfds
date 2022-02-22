import argparse
import json
import os
import time

import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

import seaborn as sns

from plotting import plot_single_trajectory
import multiprocessing
multiprocessing.set_start_method("spawn", True)

# Arguments
parser = argparse.ArgumentParser(description='Adaptive Concrete')

# Settings
parser.add_argument('--weights', type=str, default="ConcreteValid1-20200428-154850", help='weight name')
args = parser.parse_args()

if __name__ == "__main__":
    pred_dir = args.weights + '/pred'

    all_logits = np.load(pred_dir + '/logits.npy')
    all_labels = np.load(pred_dir + '/labels.npy')
    all_selections = np.load(pred_dir + '/selections.npy')

    unique_labels = np.unique(all_labels)

    feature_counts = np.zeros((all_selections.shape[-1], unique_labels.shape[0]))
    num_labels = np.zeros((unique_labels.shape[0]))
    for i in range(all_selections.shape[0]):
        current_selection = all_selections[i, :, :]
        current_truth = all_labels[i, :]
        for label in unique_labels:
            feature_counts[:, label] += np.sum(current_selection[current_truth == label, :], axis=0)
            num_labels[label] += np.sum(current_truth == label)

    sns.set(rc={'figure.figsize':(10,6)})
    sns.set(font_scale=2)
    # feature_counts = feature_counts / (all_selections.shape[0] * all_selections.shape[1])
    feature_indices = np.array(range(all_selections.shape[-1]))
    feature_counts = feature_counts / num_labels
    feature_totals = np.sum(feature_counts, axis=1) / unique_labels.shape[0]

    feature_counts = feature_counts[feature_totals > 0.01, :]
    feature_indices = feature_indices[feature_totals > 0.01]

    label_name = ['Walking', 'Walk Up', 'Walk Down', 'Sitting', 'Standing', 'Laying']
    
    sns_plot = sns.heatmap(np.transpose(feature_counts), xticklabels=feature_indices, yticklabels=label_name)
    plt.xlabel('Feature Number')
    plt.ylabel('Activity')
    sns_plot = sns_plot.get_figure()
    sns_plot.savefig("feature_heatmap2.pdf", bbox_inches='tight')
    sns_plot.savefig("feature_heatmap2.png", bbox_inches='tight')

    

