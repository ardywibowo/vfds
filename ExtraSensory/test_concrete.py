import argparse
import json
import os
import time

import numpy as np

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader

from sklearn.metrics import f1_score

from AdaptiveConcreteGRU import AdaptiveGRU
from ExtrasensoryDataset import ExtrasensoryDataset

from plotting import plot_single_trajectory
import multiprocessing
multiprocessing.set_start_method("spawn", True)

# Arguments
parser = argparse.ArgumentParser(description='Adaptive Concrete')

# Settings
parser.add_argument('--gpu', type=int, default=2, help='gpu device id')
parser.add_argument('--interpolate', type=bool, default=False, help='interpolate features')
parser.add_argument('--seg_len', type=int, default=100, help='segment length')
parser.add_argument('--n_mult', type=int, default=10, help='segment length')
parser.add_argument('--weights', type=str, default="Test", help='weight name')
parser.add_argument('--save_pred', type=bool, default=False, help='Save Predictions')
parser.add_argument('--plot', type=bool, default=False, help='Plot')
args = parser.parse_args()

def test(model, test_loader, epochs):
    
    inputs, labels = next(iter(test_loader))
    [batch_size, seq_length, num_features] = inputs.size()
    num_classes = model.output_size

    cur_time = time.time()
    pre_time = time.time()
    
    use_gpu = torch.cuda.is_available()

    all_logits = []
    all_labels = []
    all_selections = []
    
    union = 0
    intersection = 0

    tested_batch = 0
    errors = 0
    num_points = 0
    num_samples = 0
    features_selected = 0
    f1 = 0

    test_iter = iter(test_loader)
    index = 0
    for i in range(epochs):
        # Get data
        try:
            inputs, labels = next(test_iter)
        except StopIteration:
            test_iter = iter(test_loader)
            inputs, labels = next(test_iter)
        
        inputs = inputs.float()
        labels = labels.float()
        labels = labels.permute(1, 0, 2) # seq_length x batch x num_classes
        labels = labels.reshape(-1, labels.shape[2]) # seq_length * batch x num_classes
        
        if inputs.shape[0] != batch_size:
            continue
    
        if use_gpu:
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
        else: 
            inputs, labels = Variable(inputs), Variable(labels)
        
        # Make prediction
        outputs, num_selections, selection_weights = model(inputs)
        outputs = torch.cat(outputs) # seq_length * batch x num_classes
        selection_weights = torch.cat(selection_weights)

        # Hard treshold
        labelled_outputs = outputs[labels != -1]
        labelled_labels = labels[labels != -1]
        labelled_predictions = (labelled_outputs > 0)

        # Count errors
        labelled_labels = labelled_labels.cpu().detach().numpy()
        labelled_predictions = labelled_predictions.cpu().detach().numpy()
        num_selections = num_selections.cpu().detach().numpy()

        # F1 Score
        f1 += f1_score(labelled_labels, labelled_predictions)

        # IoU
        uni = (labelled_labels + labelled_predictions) > 0
        inter = labelled_labels * labelled_predictions

        union += np.sum(uni)
        intersection += np.sum(inter)
        iou = intersection / union

        # Accuracy
        errors += np.sum(labelled_predictions != labelled_labels)
        features_selected += num_selections
        num_points += batch_size * seq_length * num_features
        num_samples += labelled_labels.shape[0]
        
        # Print stats
        tested_batch += 1
        if tested_batch % 10 == 0:
            cur_time = time.time()
            print('Tested #: {}, errors: {}, IoU: {}, F1: {}, features selected: {}, time: {}'.format( \
                tested_batch * batch_size, \
                np.around([errors / num_samples], decimals=8), \
                np.around([iou], decimals=8), \
                np.around([f1 / tested_batch], decimals=8), \
                np.around([features_selected / num_points], decimals=8), \
                np.around([cur_time - pre_time], decimals=8) ) )
            pre_time = cur_time
        
        # Plot trajectory
        if args.plot:
            save_dir = os.path.join(args.weights, 'plots', str(index) + '.pdf')
            plot_single_trajectory(prediction, labels, selection_weights, save_dir=save_dir)
            index += 1

        # Save predictions
        if args.save_pred:
            pred_dir = os.path.join(args.weights, 'pred')

            all_logits.append(outputs.cpu().detach().numpy())
            all_labels.append(labels.cpu().detach().numpy())
            all_selections.append(selection_weights.cpu().detach().numpy())
            np.save(pred_dir + '/logits', np.array(all_logits))
            np.save(pred_dir + '/labels', np.array(all_labels))
            np.save(pred_dir + '/selections', np.array(all_selections))
    
    print('Tested: errors: {}, features selected: {}'.format(errors / num_samples, features_selected / num_points))

def create_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)

if __name__ == "__main__":
    print(args)
    torch.cuda.set_device(args.gpu)

    plot_dir = os.path.join(args.weights, 'plots')
    create_dir(plot_dir)

    pred_dir = os.path.join(args.weights, 'pred')
    create_dir(pred_dir)

    test_data = ExtrasensoryDataset('../ExtraSensory/', split='test', segment_length=args.seg_len)
    num_classes = test_data.num_classes

    test_loader = DataLoader(test_data, num_workers=1, shuffle=False, batch_size=1)

    inputs, labels = next(iter(test_loader))
    [batch_size, seq_length, input_size] = inputs.size()

    model = AdaptiveGRU(input_size, args.n_mult * input_size, num_classes, hard=True)
    model.load_state_dict(torch.load(args.weights + "/best_model.pt", map_location="cuda:{}".format(args.gpu)))
    model = model.to(args.gpu)

    test(model, test_loader, 6000)
