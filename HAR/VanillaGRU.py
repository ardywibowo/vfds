import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class VanillaGRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(VanillaGRU, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        use_gpu = torch.cuda.is_available()
        if use_gpu:
            self.identity = torch.eye(input_size).cuda()
            self.zeros = Variable(torch.zeros(input_size).cuda())
        else:
            self.identity = torch.eye(input_size)
            self.zeros = Variable(torch.zeros(input_size))

        self.cell = nn.GRUCell(self.input_size, self.hidden_size)
        self.output_layer = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input):
        batch_size = input.shape[0]
        seq_length = input.shape[1]
        num_features = input.shape[2]
        
        hidden_state = self.init_hidden(batch_size)
        
        outputs = []
        for i in range(seq_length):
            curr_in = torch.squeeze(input[:, i, :], dim=1) # batch x feature
            hidden_state = self.cell(curr_in, hidden_state)
            curr_out = self.output_layer(hidden_state)

            outputs.append(curr_out)

        return outputs
    
    def init_hidden(self, batch_size):
        use_gpu = torch.cuda.is_available()
        if use_gpu:
            hidden_state = Variable(torch.zeros(batch_size, self.hidden_size).cuda())
            hidden_state = hidden_state.float()
            return hidden_state
        else:
            hidden_state = Variable(torch.zeros(batch_size, self.hidden_size))
            hidden_state = hidden_state.float()
            return hidden_state
