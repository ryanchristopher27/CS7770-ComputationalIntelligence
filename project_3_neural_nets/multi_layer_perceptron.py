# Imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.autograd import Function
import torch.optim as optim
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# Multi-Layer Preceptron Class
class MLP(nn.Module):
    def __init__(self, D_in, H_1, D_out):
        super(MLP, self).__init__()
        self.linear1 = nn.Linear(D_in, H_1) # input to hidden layer
        self.linear2 = nn.Linear(H_1, D_out) # hidden layer to output

    def forward(self, x):
        h_pred = F.relu(self.linear1(x)) # h = dot(input,w1) 
                                         #  and nonlinearity (relu)
        y_pred = self.linear2(h_pred) # network_output = dot(h,w2)
        return y_pred
    

# Multi-Layer Preceptron Class for Heart Disease Classification
class MLP_HD(nn.Module):
    def __init__(self, D_in, H1, H2, D_out):
        super(MLP_HD, self).__init__()
        # self.layers = nn.Sequential(
        #     nn.Linear(D_in, H1, dtype=float),
        #     nn.ReLU(),
        #     nn.Linear(H1,H2, dtype=float),
        #     nn.ReLU(),            
        #     nn.Linear(H2, D_out, dtype=float),
        #     nn.Sigmoid()
        # )  
        self.linear1 = nn.Linear(D_in, H1, dtype=float) # input to hidden layer
        self.linear2 = nn.Linear(H1, H2, dtype=float)
        self.linear3 = nn.Linear(H2, D_out, dtype=float) # hidden layer to output

    def forward(self, x):
        h1_pred = F.relu(self.linear1(x))
        h2_pred = F.relu(self.linear2(h1_pred))
        y_pred = F.sigmoid(self.linear3(h2_pred)) # network_output = dot(h,w2)
        # y_pred = self.layers(x)
        return y_pred
    
class HeartDiseaseNN(nn.Module):
    def __init__(self):
        super(HeartDiseaseNN, self).__init__()
        self.fc1 = nn.Linear(13, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 256)
        self.fc4 = nn.Linear(256, 5)
        self.act = nn.LeakyReLU()
        self.dropout = nn.Dropout(0.1)
    def forward(self, x):
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        x = self.act(self.fc3(x))
        x = self.dropout(x)
        return self.fc4(x)