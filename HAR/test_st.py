import argparse
import json
import os
import time

import numpy as np

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader

from AdaptiveSTGRU import AdaptiveGRU
from uci_har import UciHarDataset

from plotting import plot_single_trajectory
import multiprocessing
multiprocessing.set_start_method("spawn", True)

# Arguments
parser = argparse.ArgumentParser(description='Adaptive Concrete')

# Settings
parser.add_argument('--gpu', type=int, default=1, help='init learning rate')
parser.add_argument('--weights', type=str, default="Test", help='weight name')
parser.add_argument('--plot', type=bool, default=False, help='Plot trajectories')
parser.add_argument('--save_pred', type=bool, default=False, help='Save predictions')
args = parser.parse_args()

def test(model, test_loader, epochs):
    
    inputs, labels = next(iter(test_loader))
    [batch_size, seq_length, num_features] = inputs.size()

    cur_time = time.time()
    pre_time = time.time()
    
    use_gpu = torch.cuda.is_available()
    
    all_logits = []
    all_labels = []
    all_selections = []

    tested_batch = 0
    errors = 0
    num_samples = 0
    features_selected = 0

    test_iter = iter(test_loader)
    index = 0
    for i in range(epochs):
        try:
            inputs, labels = next(test_iter)
        except StopIteration:
            test_iter = iter(test_loader)
            inputs, labels = next(test_iter)
        
        inputs = inputs.float()

        labels = labels.long()
        labels = labels.permute(1, 0) # seq_length x batch
        labels = labels.flatten() # seq_length * batch
        
        if inputs.shape[0] != batch_size:
            continue
    
        if use_gpu:
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
        else: 
            inputs, labels = Variable(inputs), Variable(labels)
        
        outputs, num_selections, selection_weights = model(inputs)
        outputs = torch.cat(outputs) # seq_length * batch x num_classes
        prediction = torch.argmax(outputs, dim=1)

        selection_weights = torch.cat(selection_weights)

        if args.plot:
            save_dir = os.path.join(args.weights, 'plots', str(index) + '.pdf')
            plot_single_trajectory(prediction, labels, selection_weights, save_dir=save_dir)
            index += 1

        errors += torch.sum(prediction != labels).cpu().detach().numpy()

        features_selected += num_selections.cpu().detach().numpy()

        num_samples += seq_length * batch_size
        
        tested_batch += 1
        if tested_batch % 10 == 0:
            cur_time = time.time()
            print('Tested #: {}, errors: {}, features selected: {}, time: {}'.format( \
                  tested_batch * batch_size, \
                  np.around([errors / num_samples], decimals=8), \
                  np.around([features_selected / (num_samples * num_features)], decimals=8), \
                  np.around([cur_time - pre_time], decimals=8) ) )
            pre_time = cur_time
        
        # Save predictions
        if args.save_pred:
            pred_dir = os.path.join(args.weights, 'pred')

            all_logits.append(outputs.cpu().detach().numpy())
            all_labels.append(labels.cpu().detach().numpy())
            all_selections.append(selection_weights.cpu().detach().numpy())
            np.save(pred_dir + '/logits', np.array(all_logits))
            np.save(pred_dir + '/labels', np.array(all_labels))
            np.save(pred_dir + '/selections', np.array(all_selections))
    
    print('Tested: errors: {}, features selected: {}'.format(errors / num_samples, features_selected / num_samples))

def create_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)

if __name__ == "__main__":
    torch.cuda.set_device(args.gpu)

    plot_dir = os.path.join(args.weights, 'plots')
    create_dir(plot_dir)

    pred_dir = os.path.join(args.weights, 'pred')
    create_dir(pred_dir)

    test_data = UciHarDataset('../UCI HAR Dataset/', split='test', segment_length=200)
    num_classes = test_data.num_classes

    test_loader = DataLoader(test_data, num_workers=1, shuffle=False, batch_size=1)

    inputs, labels = next(iter(test_loader))
    [batch_size, seq_length, input_size] = inputs.size()

    model = AdaptiveGRU(input_size, 5 * input_size, num_classes)
    model.load_state_dict(torch.load(args.weights + "/best_model.pt", map_location="cuda:{}".format(args.gpu)))
    model = model.to(args.gpu)

    test(model, test_loader, 6000)
