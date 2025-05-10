import torch
import torch.nn as nn


def swish(x):
    return x * torch.sigmoid(x)

# The pytorch model
class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, Nunits):
        super().__init__()
        self.layers = []
        input_dim_o = input_dim
        for Nunit in Nunits:
            self.layers.append(nn.Linear(input_dim, Nunit))
            input_dim = Nunit
        
        self.layers.append(nn.Linear(input_dim, output_dim))
        self.layers.append(nn.Linear(input_dim_o, output_dim))
        # Assigning the layers as class variables (PyTorch requirement). 
        for idx, layer in enumerate(self.layers):
            setattr(self, "fc{}".format(idx), layer)
            
    def forward(self, data):
        data_l = self.layers[-1](data)
        for layer in self.layers[:-2]:
            data = layer(data)
            data = swish(data)
        data = self.layers[-2](data)+data_l
        return data
 