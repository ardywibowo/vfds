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
parser.add_argument('--weights', type=str, default="ConcretePlot1-20200526-015454", help='weight name')
args = parser.parse_args()

if __name__ == "__main__":
    pred_dir = args.weights + '/pred'

    all_logits = np.load(pred_dir + '/logits.npy')
    all_labels = np.load(pred_dir + '/labels.npy')
    all_selections = np.load(pred_dir + '/selections.npy')

    feature_counts = np.zeros((all_selections.shape[-1], all_labels.shape[-1]))
    num_labels = np.zeros((all_labels.shape[-1]))
    for i in range(all_selections.shape[0]):
        current_selection = all_selections[i, :, :]
        current_truth = all_logits[i, :]
        current_truth = current_truth > 0
        current_truth = current_truth.astype(int)
        # current_truth[current_truth == -1] = 0
        for j in range(all_labels.shape[-1]):
            feature_counts[:, j] += np.sum(current_selection[current_truth[:, j].astype(bool), :], axis=0)
            num_labels[j] += np.sum(current_truth[:, j])

    sns.set(rc={'figure.figsize':(16,8)})
    # feature_counts = feature_counts / (all_selections.shape[0] * all_selections.shape[1])
    feature_indices = np.array(range(all_selections.shape[-1]))

    feature_counts = feature_counts / num_labels
    feature_counts = feature_counts[:, num_labels != 0]
    feature_totals = np.sum(feature_counts, axis=1) / all_labels.shape[-1]

    feature_counts = feature_counts[(feature_totals > 0.05), :]
    feature_indices = feature_indices[(feature_totals > 0.05)]
    feature_totals = feature_totals[(feature_totals > 0.05)]

    # feature_counts = feature_counts / np.max(feature_totals)

    label_name = np.array(["Phone on table", "Sitting", "Indoors", "At home", "Lying down", "Talking", 
        "Sleeping", "At main workplace", "Phone in pocket", "Eating", "Watching TV", 
        "Surfing the internet", "Standing", "Walking", "Outside", "With friends", 
        "Phone in hand", "Computer work", "With co-workers", "Dressing", "Cooking", 
        "Washing dishes", "On a bus", "Grooming", "Drive - I'm the driver", "Toilet", 
        "At school", "In a car", "Drinking (alcohol)", "In a meeting", 
        "Drive - I'm a passenger", "Bathing - shower", "Strolling", "Singing", 
        "Shopping", "At a restaurant", "Doing laundry", "Running", "Exercise", 
        "Stairs - going up", "Stairs - going down", "Bicycling", "Lab work", 
        "In class", "Cleaning", "At a party", "At a bar", "At the beach", "At the gym", 
        "Elevator", "Phone in bag"])

    label_name = label_name[num_labels != 0]

    sns_plot = sns.heatmap(np.transpose(feature_counts), xticklabels=feature_indices, yticklabels=label_name)
    plt.xlabel('Feature Number')
    plt.ylabel('Activity')
    sns_plot = sns_plot.get_figure()
    # sns_plot.savefig("feature_heatmap3.pdf", bbox_inches='tight')
    # sns_plot.savefig("feature_heatmap3.png", bbox_inches='tight')

    

