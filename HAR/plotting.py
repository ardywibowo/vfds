import os

import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

def plot_single_trajectory(predictions, labels, sel_weights, save_dir):
    predictions = predictions.cpu().numpy()
    labels = labels.cpu().numpy()
    sel_weights = sel_weights.cpu().numpy()

    predictions = predictions[1:]
    labels = labels[1:]
    sel_weights = sel_weights[1:]
    time = range(len(sel_weights))

    nonzero_features = np.any(sel_weights, axis=0)
    nonzero_index = np.nonzero(nonzero_features)
    nonzero_index = nonzero_index[0]

    sel_weights = sel_weights[:, nonzero_features]

    fig = plt.figure()
    plt.subplot(211)
    plt.plot(time, labels, label="True")
    plt.plot(time, predictions, label="Prediction")
    plt.xlim([0, len(time)])
    plt.yticks([0, 1, 2, 3, 4, 5], 
        ['Walking', 'Walk Up', 'Walk Down', 'Sitting', 'Standing', 'Laying'], rotation=40)
    plt.legend(loc=4)
    plt.xlabel('Time (50 Hz)')

    plt.subplot(212)
    plt.imshow(sel_weights.transpose(), aspect='auto')
    plt.yticks(range(sel_weights.shape[1]), nonzero_index, rotation=20)
    plt.ylabel('Feature Number')
    plt.xlabel('Time (50 Hz)')

    save_dir = os.path.join(save_dir)
    plt.savefig(save_dir)
    plt.close()