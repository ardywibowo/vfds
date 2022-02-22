import argparse
import json
import os
import time

import numpy as np

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader

from AdaptiveArmGRU import AdaptiveGRU
from OpportunityDataset import OpportunityDataset

import multiprocessing
multiprocessing.set_start_method("spawn", True)

# Arguments
parser = argparse.ArgumentParser(description='AdaptiveGRU Train')

# General Settings
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--save', type=str, default='Train', help='experiment name')
parser.add_argument('--n_hidden', type=int, default=256, help='hidden units')
parser.add_argument('--n_layers', type=int, default=2, help='GRU layers')
parser.add_argument('--seg_len', type=int, default=100, help='Segment length')
parser.add_argument('--batch', type=int, default=100, help='batch size')
parser.add_argument('--v_batch', type=int, default=200, help='validation batch size')
parser.add_argument('--reg_weight', type=float, default=1.0, help='regularization weight')
parser.add_argument('--interpolate', type=bool, default=False, help='linearly interpolate features')
parser.add_argument('--remove_feat', type=bool, default=False, help='remove features according to paper')
parser.add_argument('--st', type=bool, default=False, help='straight through')
parser.add_argument('--sigscale', type=float, default=1.0, help='sigmoid scale')

args = parser.parse_args()

def compute_arm_loss(model, loss, inputs, labels, reg_weight, use_gpu=True):
    [batch_size, seq_length, input_size] = inputs.size()

    inputs = inputs.float()
    labels = labels.long()
    labels = labels.permute(1, 0) # seq_length x batch
    labels = labels.flatten() # seq_length * batch

    if use_gpu:
        inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
    else: 
        inputs, labels = Variable(inputs), Variable(labels)

    outputs1, outputs2, num_selections1, num_selections2, all_hidden, all_u,all_selected_1, all_selected_2 = model(inputs)
    all_u = torch.cat(all_u) # seq_length * batch x feature_size
    outputs1 = torch.cat(outputs1) # seq_length * batch x num_classes
    outputs2 = torch.cat(outputs2) # seq_length * batch x num_classes
    all_hidden = torch.cat(all_hidden) # seq_length * batch x hidden_size
    all_selected_1 = torch.cat(all_selected_1) # seq_length * batch x feature_size
    all_selected_2 = torch.cat(all_selected_2) # seq_length * batch x feature_size

    # Get labelled outputs
    labelled_outputs1 = outputs1[labels != -1, :]
    labelled_outputs2 = outputs2[labels != -1, :]
    labelled_labels = labels[labels != -1]

    labelled_perf1 = loss(labelled_outputs1, labelled_labels)
    labelled_perf2 = loss(labelled_outputs2, labelled_labels)
    num_labels = labelled_labels.shape[0]

    perf1 = torch.zeros_like(labels, dtype=torch.float)
    perf2 = torch.zeros_like(labels, dtype=torch.float)

    perf1[labels != -1] = labelled_perf1
    perf2[labels != -1] = labelled_perf2
    
    # 2 Forward passes
    performance_train1_vec = perf1 * (batch_size * seq_length / num_labels)
    performance_train2_vec = perf2 * (batch_size * seq_length / num_labels)

    scaled_performance1 = torch.mean(performance_train1_vec)
    scaled_performance2 = torch.mean(performance_train2_vec)

    scaled_selections1 = num_selections1 / (batch_size * seq_length * input_size)
    scaled_selections2 = num_selections2 / (batch_size * seq_length * input_size)

    loss_train1 = scaled_performance1 + reg_weight * scaled_selections1
    loss_train2 = scaled_performance2 + reg_weight * scaled_selections2

    delta_loss = performance_train2_vec.unsqueeze(-1) + reg_weight * torch.sum(all_selected_2, 1, keepdim=True) \
        - performance_train1_vec.unsqueeze(-1) - reg_weight * torch.sum(all_selected_1, 1, keepdim=True)
    
    # Returns
    loss_computed = loss_train1
    weights_grad = model.scale_sigmoid * torch.matmul((all_u.transpose(1, 0) - 0.5), delta_loss * all_hidden)
    biases_grad = model.scale_sigmoid * torch.sum((delta_loss * (all_u-0.5)).transpose(1, 0), dim=1)

    return loss_computed, weights_grad, biases_grad, scaled_performance1, scaled_selections1

def compute_loss(model, loss, inputs, labels, reg_weight, use_gpu=True):
    [batch_size, seq_length, input_size] = inputs.size()

    inputs = inputs.float()
    labels = labels.long()
    labels = labels.permute(1, 0) # seq_length x batch
    labels = labels.flatten() # seq_length * batch

    if use_gpu:
        inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
    else: 
        inputs, labels = Variable(inputs), Variable(labels)

    outputs, _, num_selections, _, _, _, _, _ = model(inputs)
    outputs = torch.cat(outputs) # seq_length * batch x num_classes

    scaled_selected = num_selections / (batch_size * seq_length * input_size)

    labelled_outputs = outputs[labels != -1, :]
    labelled_labels = labels[labels != -1]
    num_labels = labelled_labels.shape[0]
    performance = torch.mean(loss(labelled_outputs, labelled_labels))
    scaled_performance = (batch_size * seq_length / num_labels) * performance

    loss_computed = scaled_performance + reg_weight * scaled_selected

    return loss_computed, scaled_performance, scaled_selected

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
    loss = torch.nn.CrossEntropyLoss(reduction='none')
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
        training_performances = []
        training_selections = []

        validation_losses = []
        validation_performances = []
        validation_selections = []
        
        for data in train_loader:
            inputs, labels = data

            model.zero_grad()
            loss_train, weights_grad, biases_grad, performance_train, selected_train = \
                 compute_arm_loss(model, loss, inputs, labels, args.reg_weight)

            optimizer.zero_grad()
            loss_train.backward()
            model.selector_layer.weight.grad = weights_grad
            model.selector_layer.bias.grad = biases_grad
            optimizer.step()
            
            # Validation
            try:
                inputs_val, labels_val = next(valid_iter)
            except StopIteration:
                valid_iter = iter(valid_loader)
                inputs_val, labels_val = next(valid_iter)

            model.zero_grad()
            loss_valid, performance_valid, selected_valid = \
                compute_loss(model, loss, inputs_val, labels_val, args.reg_weight)

            # Logging
            training_losses.append(loss_train.data)
            training_performances.append(performance_train.data)
            training_selections.append(selected_train.data)

            validation_losses.append(loss_valid.data)
            validation_performances.append(performance_valid.data)
            validation_selections.append(selected_valid.data)
        
        torch.save(model.state_dict(), args.save + "/latest_model" + str(label_type) + ".pt")
            
        avg_train_loss = sum(training_losses).cpu().numpy() / float(len(training_losses))
        avg_train_performances = sum(training_performances).cpu().numpy() / float(len(training_performances))
        avg_train_selections = sum(training_selections).cpu().numpy() / float(len(training_selections))

        avg_valid_loss = sum(validation_losses).cpu().numpy() / float(len(validation_losses))
        avg_valid_performances = sum(validation_performances).cpu().numpy() / float(len(validation_performances))
        avg_valid_selections = sum(validation_selections).cpu().numpy() / float(len(validation_selections))
        
        # Save best model
        is_best, min_valid_loss = save_best(model, epoch, avg_valid_loss, min_valid_loss, label_type)
        
        # Print training stats
        cur_time = time.time()
        print('Epoch: {}, train_loss: {}, valid_loss: {}, train_selections: {}, '
                'valid_selections: {}, time: {}, best model: {}'.format( \
                    epoch, \
                    np.around(avg_train_loss, decimals=8),\
                    np.around(avg_valid_loss, decimals=8),\
                    np.around(avg_train_selections, decimals=8),\
                    np.around(avg_valid_selections, decimals=8),\
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

    model = AdaptiveGRU(num_features, args.n_hidden, n_c, args.n_layers,
                    straight_through=args.st, scale_sigmoid=args.sigscale)
    train(model, train_loader, valid_loader, label_type)
