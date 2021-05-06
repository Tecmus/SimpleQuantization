from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from QuanzationLayer import  QuantizationLayer

class NetQuant(nn.Module):
    def __init__(self):
        super(NetQuant, self).__init__()
        # self.dropout1 = nn.Dropout(0.25)
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()

        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x=self.quant(x)
        
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        
        x=self.dequant(x)
        output = F.log_softmax(x, dim=1)
        
        return output

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        
        return output
    
class NetCustomQuant(nn.Module):
    def __init__(self):
        super(NetCustomQuant, self).__init__()
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.QuantizationLayer(784, 128)
        self.fc2 = nn.QuantizationLayer(128, 10)

    def forward(self, x):
        
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        
        return output