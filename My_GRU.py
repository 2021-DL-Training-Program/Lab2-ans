import torch
import torch.nn as nn

class GRU(nn.Module):
    """
    simple version
    reference:https://pytorch.org/docs/stable/generated/torch.nn.GRU.html
    """
    def __init__(self, input_size, hidden_size, bias=True):
        super(GRU, self).__init__()
        self.W_ir = nn.Linear(input_size, input_size, bias=bias)
        self.W_hr = nn.Linear(hidden_size, hidden_size, bias=bias)
        
        self.W_iz = nn.Linear(input_size, hidden_size, bias=bias)
        self.W_hz = nn.Linear(hidden_size, hidden_size, bias=bias)
        
        self.W_in = nn.Linear(input_size, hidden_size, bias=bias)
        self.W_hn = nn.Linear(hidden_size, hidden_size, bias=bias)
    
    def forward(self, input, hidden):
        r_t = torch.sigmoid(self.W_ir(input) + self.W_hr(hidden))
        z_t = torch.sigmoid(self.W_iz(input) + self.W_hz(hidden))
        n_t = torch.tanh(self.W_in(input) + r_t * self.W_hn(hidden))
        h_t = (1 - z_t) * n_t + z_t * hidden
        
        return h_t, h_t
    
class GRU2(nn.Module):
    """
    speedup version
    """
    def __init__(self, input_size, hidden_size, bias=True):
        super(GRU2, self).__init__()
        self.W_i = nn.Linear(input_size, hidden_size * 3, bias=bias)
        self.W_h = nn.Linear(hidden_size, hidden_size * 3, bias=bias)
    
    def forward(self, input, hidden):
        y_ir, y_iz, y_in = torch.chunk(self.W_i(input), 3, 2)
        y_hr, y_hz, y_hn = torch.chunk(self.W_h(hidden), 3, 2)
        r_t = torch.sigmoid(y_ir + y_hr)
        z_t = torch.sigmoid(y_iz + y_hz)
        n_t = torch.tanh(y_in + r_t * y_hn)
        h_t = (1 - z_t) * n_t + z_t * hidden
        
        return h_t, h_t
    
    