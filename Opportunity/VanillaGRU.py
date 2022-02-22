import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class VanillaGRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers=1):
        super(VanillaGRU, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        
        use_gpu = torch.cuda.is_available()
        if use_gpu:
            self.identity = torch.eye(input_size).cuda()
            self.zeros = Variable(torch.zeros(input_size).cuda())
        else:
            self.identity = torch.eye(input_size)
            self.zeros = Variable(torch.zeros(input_size))
        
        self.cells = []

        cell = nn.GRUCell(self.input_size, self.hidden_size)
        self.cells.append(cell)
        for i in range(self.n_layers-1):
            cell = nn.GRUCell(self.hidden_size, self.hidden_size)
            self.cells.append(cell)
        self.cells = nn.ModuleList(self.cells)

        self.output_layer = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input):
        batch_size = input.shape[0]
        seq_length = input.shape[1]
        num_features = input.shape[2]
        
        hidden_state = self.init_hidden(batch_size, self.n_layers)
        
        outputs = []
        for i in range(seq_length):
            curr_in = torch.squeeze(input[:, i, :], dim=1) # batch x feature

            hidden_state[0] = self.cells[0](curr_in, hidden_state[0])
            for l in range(self.n_layers - 1):
                hidden_state[l+1] = self.cells[l+1](hidden_state[l], hidden_state[l+1])

            curr_out = self.output_layer(hidden_state[-1])

            outputs.append(curr_out)

        return outputs
    
    def init_hidden(self, batch_size, n_layers):
        use_gpu = torch.cuda.is_available()
        hidden_states = []
        if use_gpu:
            for i in range(n_layers):
                hidden_state = Variable(torch.zeros(batch_size, self.hidden_size).cuda())
                hidden_state = hidden_state.float()
                hidden_states.append(hidden_state)
            return hidden_states
        else:
            for i in range(n_layers):
                hidden_state = Variable(torch.zeros(batch_size, self.hidden_size))
                hidden_state = hidden_state.float()
                hidden_states.append(hidden_state)
            return hidden_states
