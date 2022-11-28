import torch
import torch.nn as nn

class dense_model(torch.nn.Module):
    
    """Dense neural network model.
    _init_:
        num_layers: scalar, number of hidden layers.
        layer_size: scalar, number of nodes per layer.
        input_size: scalar, number of expected features in the input x.
        output_size: scalar, size of each output sample.
        dropout: scalar within [0,1], dropout rate.
    """
    def __init__(self, num_layers = 1, layer_size = 16, input_size = 1, output_size = 1, dropout=0.25):
        super().__init__()
        
        # Initialize
        # self.input_size = input_size
        # self.hidden_size = hidden_size
        # self.num_layers = num_layers
        # self.out_features_lin = out_features_lin
        # self.out_features_end = out_features_end
        # self.dropout = dropout
        
        # Define activation function
        activation = nn.ReLU
        
        # Define layers according to input parameters
        modules = []
        
        modules.append(nn.Linear(input_size,layer_size)) # Input layer
        modules.append(nn.BatchNorm1d(layer_size)) # Batch normalization
        modules.append(nn.Dropout(dropout)) # Dropout
        modules.append(activation()) # Activation function
        
        for i in range(num_layers):
            if i != num_layers-1:
                modules.append(nn.Linear(layer_size, layer_size))
                modules.append(nn.BatchNorm1d(layer_size))
                modules.append(nn.Dropout(dropout))
                modules.append(activation())
            else:
                modules.append(nn.Linear(layer_size, output_size))
        
        self.net = nn.Sequential(*modules)
        

        
    def forward(self, Z):
        for mod in self.net:
            Z = mod(Z)
        
        return Z

