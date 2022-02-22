import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class AttentionGRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers=1, hard=False, threshold=0.5):
        super(AttentionGRU, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.hard = hard
        self.n_layers = n_layers
        self.threshold = threshold
        
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
        self.selector_layer = nn.Linear(self.hidden_size, self.input_size)

    def attention_weights(self, logits, hard=False, threshold=0.5):
        if logits.shape[-1] == 1:
            y = F.sigmoid(logits)
        else:
            y = F.softmax(logits, dim=-1)

        if hard:
            return (y > threshold).float()
        else:
            return y

    def forward(self, input):
        batch_size = input.shape[0]
        seq_length = input.shape[1]
        num_features = input.shape[2]
        
        hidden_state = self.init_hidden(batch_size, self.n_layers)
        
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
                sel_log = selection_logits[-1].unsqueeze(-1) # batch x feature x 1
                weights = self.attention_weights(sel_log, hard=self.hard, threshold=self.threshold)
                weights = weights.squeeze(-1) # batch x feature
                selection_weights.append(weights)

                curr_in = weights * curr_in / (1 - self.threshold)
                num_selections += torch.sum(weights)
            
            hidden_state[0] = self.cells[0](curr_in, hidden_state[0])
            for l in range(self.n_layers - 1):
                hidden_state[l+1] = self.cells[l+1](hidden_state[l], hidden_state[l+1])

            curr_out = self.output_layer(hidden_state[-1])
            curr_logits = self.selector_layer(hidden_state[-1])

            outputs.append(curr_out)
            selection_logits.append(curr_logits)

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
