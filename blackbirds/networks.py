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

        x = self._layers(x)
        return x
    

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
        _x = self._fff(_x)
        return _x

class CNN(nn.Module):

    """
    Convoultional neural network for images/data on grids.
    """

    def __init__(self, 
                 N=100,
                 n_channels=1, 
                 hidden_layer_channels=3,
                 conv_kernel_size=4,
                 pool_kernel_size=2,
                 final_ff=[32, 16]):

        assert len(final_ff) == 2

        super().__init__()

        conv1_out_dim = N - conv_kernel_size + 1
        pool1_out_dim = int((conv1_out_dim - pool_kernel_size) / pool_kernel_size + 1)
        #conv2_out_dim = pool1_out_dim - conv_kernel_size + 1
        #pool2_out_dim = conv2_out_dim - pool_kernel_size + 1

        self.conv1 = nn.Conv2d(n_channels, 
                               hidden_layer_channels, 
                               conv_kernel_size)
        self.pool = nn.MaxPool2d(pool_kernel_size,
                                 pool_kernel_size)
        #self.conv2 = nn.Conv2d(hidden_layer_channels,
        #                       2,
        #                       conv_kernel_size)
        self.fc1 = nn.Linear(hidden_layer_channels * pool1_out_dim**2,#(pool2_out_dim)**2, 
                             final_ff[0])
        self.fc2 = nn.Linear(final_ff[0],
                             final_ff[1])
        self.relu = nn.ReLU()

    def forward(self, x):
        # Output shape of this will be (B, hidden_layer_channels, )
        x = self.pool(self.relu(self.conv1(x)))
        #x = self.pool(self.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class CNN_RNN(nn.Module):

    """
    Implementation of a CNN-RNN model for sequences of images.
    """

    def __init__(self,
                 rnn,
                 *args,
                 **kwargs):

        super().__init__()
        self.rnn = rnn
        self.cnn = CNN(*args, **kwargs)

    def forward(self, x):

        # x will be shape (B, T, C, D0, D1) – batch, time, two spatial dims, number of channels
        # So need to reshape to (B * T, C, D0, D1) before CNN
        batch_size = x.shape[0]
        n_timesteps = x.shape[1]
        x = x.reshape(batch_size * n_timesteps, x.shape[2], x.shape[3], x.shape[-1])
        x = self.cnn(x)
        x = x.reshape(batch_size, n_timesteps, x.shape[-1])
        x = self.rnn(x)
        return x
