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

import multiprocessing
multiprocessing.set_start_method("spawn", True)

# Arguments
parser = argparse.ArgumentParser(description='VanillaGRU Train')

# General Settings
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--save', type=str, default='Train', help='experiment name')
parser.add_argument('--n_hidden', type=int, default=256, help='hidden units')
parser.add_argument('--n_layers', type=int, default=2, help='GRU layers')
parser.add_argument('--seg_len', type=int, default=100, help='Segment length')
parser.add_argument('--batch', type=int, default=100, help='batch size')
parser.add_argument('--v_batch', type=int, default=200, help='validation batch size')
parser.add_argument('--interpolate', type=bool, default=False, help='linearly interpolate features')
parser.add_argument('--remove_feat', type=bool, default=False, help='remove features according to paper')

args = parser.parse_args()

def compute_loss(model, loss, inputs, labels, use_gpu=True):
    [batch_size, seq_length, input_size] = inputs.size()

    inputs = inputs.float()
    labels = labels.long()
    labels = labels.permute(1, 0) # seq_length x batch
    labels = labels.flatten() # seq_length * batch

    if use_gpu:
        inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
    else: 
        inputs, labels = Variable(inputs), Variable(labels)

    outputs = model(inputs)
    outputs = torch.cat(outputs) # seq_length * batch x num_classes

    labelled_outputs = outputs[labels != -1, :]
    labelled_labels = labels[labels != -1]
    performance = loss(labelled_outputs, labelled_labels)

    loss_computed = performance

    return loss_computed

def save_best(model, epoch, avg_valid_loss, min_valid_loss, label_type):
    if epoch == 0:
        is_best = 1
        best_model = model
        min_valid_loss = avg_valid_loss
    else:
        if avg_valid_loss < min_valid_loss:
            is_best = 1
            best_model = model
            min_valid_loss = avg_valid_loss 

            torch.save(model.state_dict(), args.save + "/best_model" + str(label_type) + ".pt")
        else:
            is_best = 0
    return is_best, min_valid_loss

def train(model, train_loader, valid_loader, label_type, num_epochs=5000):
    
    print('Model Structure: ', model)
    print('Start Training ... ')
    
    model.cuda()
    
    # Loss and optimizer
    loss = torch.nn.CrossEntropyLoss()
    learning_rate = 0.0001
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Runtime
    cur_time = time.time()
    pre_time = time.time()
    
    # Variables for Early Stopping
    min_valid_loss = np.Inf
    is_best = 0
    for epoch in range(num_epochs):
        
        valid_iter = iter(valid_loader)
        
        training_losses = []
        validation_losses = []
        
        for data in train_loader:
            inputs, labels = data

            model.zero_grad()
            loss_train = compute_loss(model, loss, inputs, labels)

            optimizer.zero_grad()
            loss_train.backward()
            optimizer.step()
            
            # Validation
            try:
                inputs_val, labels_val = next(valid_iter)
            except StopIteration:
                valid_iter = iter(valid_loader)
                inputs_val, labels_val = next(valid_iter)

            model.zero_grad()
            loss_valid = compute_loss(model, loss, inputs_val, labels_val)

            # Logging
            training_losses.append(loss_train.data)
            validation_losses.append(loss_valid.data)
        
        torch.save(model.state_dict(), args.save + "/latest_model" + str(label_type) + ".pt")
            
        avg_train_loss = sum(training_losses).cpu().numpy() / float(len(training_losses))
        avg_valid_loss = sum(validation_losses).cpu().numpy() / float(len(validation_losses))
        
        # Save best model
        is_best, min_valid_loss = save_best(model, epoch, avg_valid_loss, min_valid_loss, label_type)
        
        # Print training stats
        cur_time = time.time()
        print('Epoch: {}, train_loss: {}, valid_loss: {}, '
                'time: {}, best model: {}'.format( \
                    epoch, \
                    np.around(avg_train_loss, decimals=8),\
                    np.around(avg_valid_loss, decimals=8),\
                    np.around([cur_time - pre_time] , decimals=2),\
                    is_best) )
        pre_time = cur_time

def create_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)

if __name__ == "__main__":
    # Create directories
    print(args)
    
    args.save = '{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
    create_dir(args.save)
    torch.cuda.set_device(args.gpu)

    train_data = OpportunityDataset('../OpportunityUCIDataset/', 
                    split='train', segment_length=args.seg_len, 
                    interpolate=args.interpolate, remove_features=args.remove_feat)
    valid_data = OpportunityDataset('../OpportunityUCIDataset/', 
                    split='valid', segment_length=args.seg_len, 
                    interpolate=args.interpolate, remove_features=args.remove_feat)
    num_classes = train_data.num_classes
    label_type = 6
    n_c = num_classes[label_type]

    train_data.select_label_type(label_type)
    valid_data.select_label_type(label_type)

    train_loader = DataLoader(train_data, num_workers=1, shuffle=True, batch_size=args.batch)
    valid_loader = DataLoader(valid_data, num_workers=1, shuffle=True, batch_size=args.v_batch)

    inputs, labels = next(iter(train_loader))
    [batch_size, seq_length, num_features] = inputs.size()

    model = VanillaGRU(num_features, args.n_hidden, n_c, args.n_layers)
    train(model, train_loader, valid_loader, label_type)
