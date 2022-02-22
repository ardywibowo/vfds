import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class AdaptiveGRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, hard=False):
        super(AdaptiveGRU, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.hard = hard
        
        use_gpu = torch.cuda.is_available()
        if use_gpu:
            self.identity = torch.eye(input_size).cuda()
            self.zeros = Variable(torch.zeros(input_size).cuda())
        else:
            self.identity = torch.eye(input_size)
            self.zeros = Variable(torch.zeros(input_size))

        self.cell = nn.GRUCell(self.input_size, self.hidden_size)
        self.output_layer = nn.Linear(self.hidden_size, self.output_size)
        self.selector_layer = nn.Linear(self.hidden_size, self.input_size)

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
        
        hidden_state = self.init_hidden(batch_size)
        
        outputs = []
        selection_logits = []
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
                sel_log = selection_logits[-1].unsqueeze(-1) # batch x feature x 1
                weights = self.gumbel_softmax_sample(sel_log, temp, hard=self.hard)
                weights = weights.squeeze(-1) # batch x feature
                selection_weights.append(weights)

                curr_in = weights * curr_in
                num_selections += torch.sum(weights)
            
            hidden_state = self.cell(curr_in, hidden_state)

            curr_out = self.output_layer(hidden_state)
            curr_logits = self.selector_layer(hidden_state)

            outputs.append(curr_out)
            selection_logits.append(curr_logits)

        return outputs, num_selections, selection_weights
    
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
