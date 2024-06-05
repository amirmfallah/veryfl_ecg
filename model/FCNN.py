import torch
import torch.nn as nn
import torch.nn.functional as F

def createFCNNModel():
    return SimpleFCNN(input_size=128, hidden_sizes=[256, 128, 64], output_size=2)

class SimpleFCNN(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(SimpleFCNN, self).__init__()
        layers = []
        for i in range(len(hidden_sizes)):
            if i == 0:
                layers.append(nn.Linear(input_size, hidden_sizes[i]))
            else:
                layers.append(nn.Linear(hidden_sizes[i-1], hidden_sizes[i]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.5))
        
        layers.append(nn.Linear(hidden_sizes[-1], output_size))
        self.layers = nn.Sequential(*layers)
        
    def forward(self, x):
        x = self.layers(x)
        return x

#model = createModel()
