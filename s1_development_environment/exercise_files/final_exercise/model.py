from torch import nn
import torch.nn.functional as F
import numpy as np
import torch

class MyNetwork(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers, drop_p=0.5):
        ''' Builds a feedforward network with arbitrary hidden layers.
        
            Arguments
            ---------
            input_size: integer, size of the input layer
            output_size: integer, size of the output layer
            hidden_layers: list of integers, the sizes of the hidden layers
        
        '''
        super().__init__()
        # Input to first hidden layer
        self.hidden_layers = nn.ModuleList([nn.Linear(input_size, hidden_layers[0])])
        
        # Add the remaining hidden layers
        self.hidden_layers.extend([nn.Linear(hidden_layers[h], hidden_layers[h+1]) for h in range(len(hidden_layers)-1)])
        
        # Output layer
        self.output = nn.Linear(hidden_layers[-1], output_size)
        
        self.dropout = nn.Dropout(p=drop_p)
        
    def forward(self, x):
        ''' Forward pass through the network, returns the output logits '''
        x = x.to(torch.float32)
        for h_layer in self.hidden_layers:
            x = F.relu(h_layer(x))
            x = self.dropout(x)
        x = self.output(x)
        
        return F.log_softmax(x, dim=1)
