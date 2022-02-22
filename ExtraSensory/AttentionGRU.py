import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class AttentionGRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, hard=False, threshold=0.5):
        super(AttentionGRU, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.hard = hard
        self.threshold = threshold
        
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
                sel_log = selection_logits[-1].unsqueeze(-1) # batch x feature x 1
                weights = self.attention_weights(sel_log, hard=self.hard, threshold=self.threshold)
                weights = weights.squeeze(-1) # batch x feature
                selection_weights.append(weights)

                curr_in = weights * curr_in / (1 - self.threshold)
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
