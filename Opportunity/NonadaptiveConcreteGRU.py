import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class NonadaptiveGRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers=1, hard=False):
        super(NonadaptiveGRU, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.hard = hard
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
        self.sel_logits = nn.Parameter(torch.empty(self.input_size).normal_(mean=0, std=1e-1), requires_grad=True)

    def sample_gumbel(self, shape, eps=1e-20):
        U = torch.rand(shape).cuda()
        return -Variable(torch.log(-torch.log(U + eps) + eps))

    def gumbel_softmax_sample(self, logits, temperature, hard=False, deterministic=False, eps=1e-20):
        
        if deterministic:
            if logits.shape[-1] == 1:
                return F.sigmoid(logits)
            else:
                return F.softmax(logits, dim=-1)
        
        # Stochastic
        if logits.shape[-1] == 1:
            noise = torch.rand_like(logits)
            y = (logits + torch.log(noise + eps) - torch.log(1 - noise + eps))            
            y = torch.sigmoid(y / temperature)
            if hard:
                return (y > 0.5).float()
            else:
                return y
        else:
            y = logits + self.sample_gumbel(logits.size())
            y = F.softmax(y / temperature, dim=-1)
            if hard:
                return (y > 0.5).float()
            else:
                return y

    def forward(self, input):
        batch_size = input.shape[0]
        seq_length = input.shape[1]
        num_features = input.shape[2]
        
        hidden_state = self.init_hidden(batch_size, self.n_layers)
        
        outputs = []
        selection_weights = []
        num_selections = 0
        for i in range(seq_length):
            if i == 0:
                curr_in = torch.squeeze(input[:, i, :], dim=1) # batch x feature

                weights = torch.ones((batch_size, num_features)).cuda()
                selection_weights.append(weights)
            else:
                curr_in = torch.squeeze(input[:, i, :], dim=1) # batch x feature
                
                # Feature selection
                temp = 0.05
                weights = self.gumbel_softmax_sample(self.sel_logits.unsqueeze(-1), temp, hard=self.hard)
                weights = weights.squeeze(-1) # batch x feature
                selection_weights.append(weights.unsqueeze(0))

                curr_in = weights * curr_in
                num_selections += torch.sum(weights)
            
            hidden_state[0] = self.cells[0](curr_in, hidden_state[0])
            for l in range(self.n_layers - 1):
                hidden_state[l+1] = self.cells[l+1](hidden_state[l], hidden_state[l+1])

            curr_out = self.output_layer(hidden_state[-1])
            outputs.append(curr_out)

        return outputs, num_selections, selection_weights
    
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
