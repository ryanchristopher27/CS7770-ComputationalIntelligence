# include our Python packages
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.autograd import Function
from torch.utils.data import random_split
import torch.optim as optim
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import pandas as pd

from multi_layer_perceptron import MLP, MLP_HD
from ucimlrepo import fetch_ucirepo 
from helpers import display_confusion_matrix


def main():
    example()


def example():
    ##############################################
    # create an instance and set up optimization
    ##############################################

    # here is a network with 2 inputs to 4 hidden neurons to one output neuron    
    D_in, H, D_out = 2, 4, 1    
    net = MLP(D_in, H, D_out)

    # now, optimization and draw stuff (look at perceptron Jupyter pages)

    def criterion(out,label):
        return (label - out)**2

    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.3)

    ##############################################
    # xor data set
    ##############################################

    data = torch.randn(4,2)
    data[0,0] = 0; data[0,1] = 0;
    data[1,0] = 1; data[1,1] = 1;
    data[2,0] = 0; data[2,1] = 1;
    data[3,0] = 1; data[3,1] = 0;

    L = torch.randn(4)
    L[0] = 0; L[1] = 0
    L[2] = 1; L[3] = 1

    ##############################################
    # training
    ##############################################

    for epoch in range(1500):
        for i in range(4):
            X = Variable(data[i,:])
            Y = Variable(L[i])
            optimizer.zero_grad()
            outputs = net(X)
            loss = criterion(outputs, Y)
            loss.backward()
            optimizer.step()

    print(net(Variable(torch.Tensor([[[0,0]]]))))
    print(net(Variable(torch.Tensor([[[1,0]]]))))
    print(net(Variable(torch.Tensor([[[0,1]]]))))
    print(net(Variable(torch.Tensor([[[1,1]]]))))

    # Plot decision boundary
    x_min, x_max = data[:, 0].min()-0.1, data[:, 0].max()+0.1
    y_min, y_max = data[:, 1].min()-0.1, data[:, 1].max()+0.1
    spacing = min(x_max - x_min, y_max - y_min) / 100
    XX, YY = np.meshgrid(np.arange(x_min, x_max, spacing),
                np.arange(y_min, y_max, spacing))
    datax = np.hstack((XX.ravel().reshape(-1,1), 
                    YY.ravel().reshape(-1,1)))
    data_t = torch.FloatTensor(datax)
    db_prob = net(data_t)
    clf = np.where(db_prob<0.5,0,1)
    Z = clf.reshape(XX.shape)
    plt.figure(figsize=(12,8))
    plt.contourf(XX, YY, Z, cmap=plt.cm.Accent, alpha=0.5)
    plt.scatter(data[:,0], data[:,1], c=L, cmap=plt.cm.Accent)
    plt.show()

if __name__=="__main__":
    main()