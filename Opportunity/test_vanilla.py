import argparse
import json
import os
import time

import numpy as np

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader

from VanillaGRU import VanillaGRU
from OpportunityDataset import OpportunityDataset

from plotting import plot_single_trajectory
import multiprocessing
multiprocessing.set_start_method("spawn", True)

# Arguments
parser = argparse.ArgumentParser(description='Adaptive Concrete')

# Settings
parser.add_argument('--gpu', type=int, default=3, help='init learning rate')
parser.add_argument('--weights', type=str, default="Test", help='weight name')
parser.add_argument('--n_hidden', type=int, default=256, help='hidden units')
parser.add_argument('--n_layers', type=int, default=2, help='GRU layers')
parser.add_argument('--seg_len', type=int, default=100, help='Segment length')
parser.add_argument('--interpolate', type=bool, default=False, help='linearly interpolate features')
parser.add_argument('--remove_feat', type=bool, default=False, help='remove features according to paper')
parser.add_argument('--plot', type=bool, default=False, help='Plot')
args = parser.parse_args()

def test(model, test_loader, label_type, epochs):
    
    inputs, labels = next(iter(test_loader))
    [batch_size, seq_length, num_features] = inputs.size()

    cur_time = time.time()
    pre_time = time.time()
    
    use_gpu = torch.cuda.is_available()
    
    tested_batch = 0
    errors = 0
    num_samples = 0

    index = 0
    while True:
        for inputs, labels in test_loader:
            
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
            
            outputs = model(inputs)
            outputs = torch.cat(outputs) # seq_length * batch x num_classes
            prediction = torch.argmax(outputs, dim=1)

            if args.plot:
                save_dir = os.path.join(args.weights, 'plots' + str(label_type), str(index) + '.png')
                plot_single_trajectory(prediction, labels, selection_weights, save_dir=save_dir)
                index += 1

            labelled_predictions = prediction[labels != -1]
            labelled_labels = labels[labels != -1]
            errors += torch.sum(labelled_predictions != labelled_labels).cpu().detach().numpy()

            num_samples += labelled_labels.shape[0]
            
            tested_batch += 1
            if tested_batch % 10 == 0:
                cur_time = time.time()
                print('Tested #: {}, errors: {}, time: {}'.format( \
                    tested_batch * batch_size, \
                    np.around([errors / num_samples], decimals=8), \
                    np.around([cur_time - pre_time], decimals=8) ) )
                pre_time = cur_time

        if tested_batch > epochs:
            break
    
    print('Label Type: {}, Tested: errors: {}'.format(label_type, errors / num_samples))

def create_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)

if __name__ == "__main__":
    torch.cuda.set_device(args.gpu)

    test_data = OpportunityDataset('../OpportunityUCIDataset/', split="test", segment_length=args.seg_len,
                        interpolate=args.interpolate, remove_features=args.remove_feat)
    num_classes = test_data.num_classes
    label_type = 6
    n_c = num_classes[label_type]

    print("Testing Label Type: {}".format(label_type))
    plot_dir = os.path.join(args.weights, 'plots' + str(label_type))
    create_dir(plot_dir)

    test_data.select_label_type(label_type)
    test_loader = DataLoader(test_data, num_workers=1, shuffle=False, batch_size=1)

    inputs, labels = next(iter(test_loader))
    [batch_size, seq_length, input_size] = inputs.size()

    model = VanillaGRU(input_size, args.n_hidden, n_c, args.n_layers)
    model.load_state_dict(torch.load(args.weights + "/best_model" + str(label_type) + ".pt", 
                            map_location="cuda:{}".format(args.gpu)))
    model = model.to(args.gpu)

    test(model, test_loader, label_type, 6000)
