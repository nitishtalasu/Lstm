import math
import torch
from torch.autograd import Variable
import torch.nn as nn
from torch.nn import functional, init

class LSTMCell(nn.Module):

    """A basic LSTM cell."""

    def __init__(self, input_size, hidden_size, use_bias=True):
        """
        Most parts are copied from torch.nn.LSTMCell.
        """
        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = use_bias
        self.x2h = nn.Linear(input_size, 4 * hidden_size, bias=self.bias)
        self.h2h = nn.Linear(hidden_size, 4 * hidden_size, bias=self.bias)
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)
    
    def forward(self, x, hidden):
        
        hx, cx = hidden        
        x = x.view(-1, x.size(1))        
        gates = self.x2h(x) + self.h2h(hx)    
        gates = gates.squeeze()        
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)        
        ingate = functional.sigmoid(ingate)
        forgetgate = functional.sigmoid(forgetgate)
        cellgate = functional.tanh(cellgate)
        outgate = functional.sigmoid(outgate)
        cy = torch.mul(cx, forgetgate) +  torch.mul(ingate, cellgate)        
        hy = torch.mul(outgate, functional.tanh(cy))
        
        return (hy, cy)

class LSTM(nn.Module):

    """A module that runs multiple steps of LSTM."""

    def __init__(self, cell_class, input_size, hidden_size, num_layers=1,
                 use_bias=True, batch_first=False, dropout=0, **kwargs):
        super(LSTM, self).__init__()
        self.cell_class = cell_class
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.use_bias = use_bias
        self.batch_first = batch_first
        self.dropout = dropout

        for layer in range(num_layers):
            layer_input_size = input_size if layer == 0 else hidden_size
            cell = cell_class(input_size=layer_input_size,
                              hidden_size=hidden_size,
                              **kwargs)
            setattr(self, 'cell_{}'.format(layer), cell)
        self.dropout_layer = nn.Dropout(dropout)
        self.reset_parameters()

    def get_cell(self, layer):
        return getattr(self, 'cell_{}'.format(layer))
    
    def reset_parameters(self):
        for layer in range(self.num_layers):
            cell = self.get_cell(layer)
            cell.reset_parameters()
    
    def init_hidden(self, batch_size):
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        weight = next(self.parameters()).data
        hidden = (weight.new(self.num_layers, batch_size, self.hidden_size).zero_().to(device),
                      weight.new(self.num_layers, batch_size, self.hidden_size).zero_().to(device))
        return hidden

    @staticmethod
    def _forward_lstm(cell, x, hx):
        output = []
        for time in range(x.size(0)):
            h_next, c_next = cell(x=x[time], hidden=hx)
            hx_next = (h_next, c_next)
            output.append(h_next)
            hx = hx_next
        output = torch.stack(output, 0)
        return output, hx

    def forward(self, x, hidden):
        if self.batch_first:
            x = x.transpose(0, 1)
                
        h_n = []
        c_n = []
        layer_output = None
        for layer in range(self.num_layers):
            cell = self.get_cell(layer)
            hx_layer = (hidden[0][layer,:,:], hidden[1][layer,:,:])
            
            if layer == 0:
                layer_output, (layer_h_n, layer_c_n) = LSTM._forward_lstm(
                    cell=cell, x=x, hx=hx_layer)
            else:
                layer_output, (layer_h_n, layer_c_n) = LSTM._forward_lstm(
                    cell=cell, x=layer_output, hx=hx_layer)
            
            input_ = self.dropout_layer(layer_output)
            h_n.append(layer_h_n)
            c_n.append(layer_c_n)
        output = layer_output
        h_n = torch.stack(h_n, 0)
        c_n = torch.stack(c_n, 0)
        return output, (h_n, c_n)


