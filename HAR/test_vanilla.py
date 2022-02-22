import argparse
import json
import os
import time

import numpy as np

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader

from VanillaGRU import VanillaGRU
from uci_har import UciHarDataset

import multiprocessing
multiprocessing.set_start_method("spawn", True)

# Arguments
parser = argparse.ArgumentParser(description='Adaptive Concrete')

# Settings
parser.add_argument('--gpu', type=float, default=2, help='init learning rate')
args = parser.parse_args()

def test(model, test_loader, epochs):
    
    inputs, labels = next(iter(test_loader))
    [batch_size, seq_length, num_features] = inputs.size()

    cur_time = time.time()
    pre_time = time.time()
    
    use_gpu = torch.cuda.is_available()
    
    tested_batch = 0
    errors = 0
    num_samples = 0

    test_iter = iter(test_loader)
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
        
        outputs = model(inputs)
        outputs = torch.cat(outputs) # seq_length * batch x num_classes
        prediction = torch.argmax(outputs, dim=1)

        errors += torch.sum(prediction != labels).cpu().detach().numpy()
        num_samples += seq_length * batch_size
        
        tested_batch += 1
        if tested_batch % 10 == 0:
            cur_time = time.time()
            print('Tested #: {}, errors: {}, time: {}'.format( \
                  tested_batch * batch_size, \
                  np.around([errors / num_samples], decimals=8), \
                  np.around([cur_time - pre_time], decimals=8) ) )
            pre_time = cur_time
    
    print('Tested: errors: {}'.format(errors / num_samples))


if __name__ == "__main__":
    torch.cuda.set_device(args.gpu)

    test_data = UciHarDataset('../UCI HAR Dataset/', split='test', segment_length=20)
    num_classes = test_data.num_classes

    test_loader = DataLoader(test_data, num_workers=1, shuffle=False, batch_size=1)

    inputs, labels = next(iter(test_loader))
    [batch_size, seq_length, input_size] = inputs.size()

    model = VanillaGRU(input_size, 5 * input_size, num_classes)
    model.load_state_dict(torch.load("best_model.pt", map_location="cuda:{}".format(args.gpu)))
    model = model.to(args.gpu)

    test(model, test_loader, 6000)
