import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class AdaptiveGRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, straight_through=False, scale_sigmoid=5.0):
        super(AdaptiveGRU, self).__init__()
        
        self.straight_through = straight_through
        
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
        self.selector_layer.bias.requires_grad = False
        self.selector_layer.weight.requires_grad = False

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
        
        hidden_state1 = self.init_hidden(batch_size)
        hidden_state2 = self.init_hidden(batch_size)
        
        outputs1 = []
        outputs2 = []
        selection_logits1 = []
        selection_logits2 = []
        selection_weights1 = []
        selection_weights2 = []
        all_u = []
        all_hidden = []

        num_selections1 = 0
        num_selections2 = 0
        for i in range(seq_length):
            if i == 0:
                curr_in = torch.squeeze(input[:, i, :], dim=1) # batch x feature

                curr_in1 = curr_in
                curr_in2 = curr_in

                u = self.sample_uniform((batch_size, num_features, 1)) # batch x feature x 1
                u = u.squeeze(-1) # batch x feature
                u = 0.5*torch.ones(u.shape,dtype=torch.float)
                u = u.cuda()
                all_u.append(u)
                weights1 = torch.ones(u.shape,dtype=torch.float).cuda()
                weights2 = torch.ones(u.shape,dtype=torch.float).cuda()

                selection_weights1.append(weights1)
                selection_weights2.append(weights2)
            else:
                curr_in = torch.squeeze(input[:, i, :], dim=1) # batch x feature
                
                # Feature selection
                sel_log1 = selection_logits1[-1].unsqueeze(-1) # batch x feature x 1
                sel_log2 = selection_logits2[-1].unsqueeze(-1) # batch x feature x 1

                u = self.sample_uniform(sel_log1.size()) # batch x feature x 1
                weights1 = self.categorical_sample(sel_log1, u, anti=False)
                weights2 = self.categorical_sample(sel_log2, u, anti=True)

                u = u.squeeze(-1) # batch x feature
                all_u.append(u)

                weights1 = weights1.squeeze(-1).cuda() # batch x feature
                weights2 = weights2.squeeze(-1).cuda() # batch x feature

                if self.straight_through:
                    st1 = (weights1 - F.sigmoid(sel_log1).squeeze(-1)).detach() + F.sigmoid(sel_log1).squeeze(-1)
                    st2 = (weights2 - F.sigmoid(sel_log2).squeeze(-1)).detach() + F.sigmoid(sel_log2).squeeze(-1)

                    curr_in1 = st1 * curr_in
                    curr_in2 = st2 * curr_in
                else:
                    curr_in1 = weights1 * curr_in
                    curr_in2 = weights2 * curr_in

                selection_weights1.append(weights1)
                selection_weights2.append(weights2)

                num_selections1 += torch.sum(weights1)
                num_selections2 += torch.sum(weights2)
            
            hidden_state1 = self.cell(curr_in1, hidden_state1)
            hidden_state2 = self.cell(curr_in2, hidden_state2)
            all_hidden.append(hidden_state1)
            
            curr_out1 = self.output_layer(hidden_state1)
            curr_out2 = self.output_layer(hidden_state2)

            curr_logits1 = self.selector_layer(hidden_state1) * self.scale_sigmoid
            curr_logits2 = self.selector_layer(hidden_state2) * self.scale_sigmoid

            outputs1.append(curr_out1)
            outputs2.append(curr_out2)

            selection_logits1.append(curr_logits1)
            selection_logits2.append(curr_logits2)

        return outputs1, outputs2, num_selections1, num_selections2, all_hidden, all_u, selection_weights1, selection_weights2
    
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