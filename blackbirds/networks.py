import torch
from torch import nn


def count_pars(net):
    return sum(p.numel() for p in net.parameters() if p.requires_grad)


class MLP(nn.Module):
    
    def __init__(self, 
                 input_dim=64,
                 hidden_dims=[32, 32],
                 output_dim=1):
        
        super().__init__()
        
        self.relu = nn.ReLU()
        self._layers = nn.Sequential(nn.Linear(input_dim, hidden_dims[0]), self.relu)
        for i in range(len(hidden_dims) - 1):
            self._layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
            self._layers.append(self.relu)
        self._layers.append(nn.Linear(hidden_dims[-1], output_dim))

    def forward(self, x):

        return self._layers(x)
    

class RNN(nn.Module):
    
    """
    Elman RNN or GRU followed by feedforward
    """
    
    def __init__(self,
                 input_size=2,
                 hidden_size=64,
                 num_layers=1,
                 final_ff=nn.Identity(),
                 nonlinearity='tanh',
                 flavour='gru'):
        
        super().__init__()

        if flavour == 'gru':
            self._rnn = nn.GRU(input_size, hidden_size, num_layers,
                               batch_first=True)
        else:
            self._rnn = nn.RNN(input_size, hidden_size, num_layers, nonlinearity=nonlinearity,
                               batch_first=True)
        self._fff = final_ff
        self._rnn_n_pars = count_pars(self._rnn)
        self._fff_n_pars = count_pars(self._fff)

    def forward(self, x):

        out, _ = self._rnn(x)
        _x = out[:, -1, :]
        return self._fff(_x)
