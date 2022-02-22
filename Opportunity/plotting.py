import os

import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

def label_names(label_type):
    if label_type == 0:
        label_name = ['Stand', 'Walk', 'Sit', 'Lie']
    # elif label_type == 1:
    # elif label_type == 2:
    # elif label_type == 3:
    # elif label_type == 4:
    # elif label_type == 5:
    elif label_type == 6:
        label_name = ["No Label", "Open Door 1","Open Door 2","Close Door 1","Close Door 2","Open Fridge","Close Fridge",
            "Open Dishwasher","Close Dishwasher","Open Drawer 1","Close Drawer 1","Open Drawer 2",
            "Close Drawer 2","Open Drawer 3","Close Drawer 3","Clean Table","Drink from Cup","Toggle Switch",]

    return label_name

def plot_single_trajectory(predictions, labels, sel_weights, save_dir, label_type):
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

    names = label_names(label_type)

    fig = plt.figure()
    plt.subplot(211)
    plt.plot(time, labels, label="True")
    plt.plot(time, predictions, label="Prediction")
    plt.xlim([0, len(time)])
    plt.yticks(range(len(names)), 
        names, rotation=40)
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
