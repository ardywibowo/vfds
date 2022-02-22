import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class AdaptiveGRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, scale_sigmoid=1.0):
        super(AdaptiveGRU, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        use_gpu = torch.cuda.is_available()
        if use_gpu:
            self.identity = torch.eye(input_size).cuda()
            self.zeros = Variable(torch.zeros(input_size).cuda())
            self.scale_sigmoid = torch.tensor(scale_sigmoid,requires_grad=False).cuda()
        else:
            self.identity = torch.eye(input_size)
            self.zeros = Variable(torch.zeros(input_size))
            self.scale_sigmoid = torch.tensor(scale_sigmoid,requires_grad=False)

        self.cell = nn.GRUCell(self.input_size, self.hidden_size)
        self.output_layer = nn.Linear(self.hidden_size, self.output_size)
        
        self.selector_layer = nn.Linear(self.hidden_size, self.input_size)

    def sample_uniform(self, shape):
        U = torch.rand(shape).cuda()
        return U

    def categorical_sample(self, logits, u, anti=True, deterministic=False):
        
        if deterministic:
            if logits.shape[-1] == 1:
                return F.sigmoid(logits)
            else:
                return F.softmax(logits, dim=-1)
        
        # Stochastic
        if logits.shape[-1] == 1:
            p = F.sigmoid(logits)
            q = F.sigmoid(-logits)
            mask1 = p > u
            mask2 = u > q

            if not anti:
                return mask1.float()
            else:
                return mask2.float()
        else:
            p = F.softmax(logits, dim=-1)
            q = F.softmax(-logits, dim=-1)
            mask1 = p > u
            mask2 = u > q

            if not anti:
                return mask1.float()
            else:
                return mask2.float()

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

                u = self.sample_uniform((batch_size, num_features, 1)) # batch x feature x 1
                u = u.squeeze(-1) # batch x feature
                u = 0.5*torch.ones(u.shape,dtype=torch.float)
                u = u.cuda()
                weights = torch.ones(u.shape,dtype=torch.float).cuda()

                selection_weights.append(weights)
            else:
                curr_in = torch.squeeze(input[:, i, :], dim=1) # batch x feature
                
                # Feature selection
                sel_log = selection_logits[-1].unsqueeze(-1) # batch x feature x 1

                u = self.sample_uniform(sel_log.size()) # batch x feature x 1
                weights = self.categorical_sample(sel_log, u, anti=False)

                u = u.squeeze(-1) # batch x feature

                weights = weights.squeeze(-1).cuda() # batch x feature

                st = (weights - F.sigmoid(sel_log).squeeze(-1)).detach() + F.sigmoid(sel_log).squeeze(-1)

                curr_in = st * curr_in

                selection_weights.append(weights)

                num_selections += (torch.sum(weights) - \
                    torch.sum(F.sigmoid(sel_log).squeeze(-1))).detach() + \
                        torch.sum(F.sigmoid(sel_log).squeeze(-1))
            
            hidden_state = self.cell(curr_in, hidden_state)
            
            curr_out = self.output_layer(hidden_state)

            curr_logits = self.selector_layer(hidden_state) * self.scale_sigmoid

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