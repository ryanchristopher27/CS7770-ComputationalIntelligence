# include our Python packages
import torch
from torch.autograd import Variable
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Local Imports
from multi_layer_perceptron import MLP_Cal_House, MLP_Cal_House_Leaky, MLP_Cal_House_Sigmoid
from helpers import *


def main():
    activation = 'ReLu'
    momentum = 0.5
    loss_method = 'L1'
    # california_housing_regression(activation, momentum, loss_method)
    # loss_experiment()
    # momentum_experiment()
    activation_experiment()

def loss_experiment():
    # activations = ['ReLu', 'LReLu', 'Sigmoid']
    losses = ['MSE', 'MAE', 'Smooth L1']

    mses = {}
    r2s = {}

    for loss in losses:
        mse, r2 = california_housing_regression('ReLu', 0.5, loss)
        mses[loss] = mse
        r2s[loss] = r2

    for loss in losses:
        plt.plot(mses[loss], label=f'{loss}')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.legend(loc='lower right')
    plt.title(f'MSE Vs. Loss Method')
    plt.show()

    for loss in losses:
        plt.plot(r2s[loss], label=f'{loss}')
    plt.xlabel('Epoch')
    plt.ylabel('R2')
    plt.legend(loc='lower right')
    plt.title(f'R2 Vs. Loss Method')
    plt.show()


def momentum_experiment():
    # activations = ['ReLu', 'LReLu', 'Sigmoid']
    momentums = [0.2, 0.5, 0.8]

    mses = {}
    r2s = {}

    for momentum in momentums:
        mse, r2 = california_housing_regression('ReLu', momentum, 'MSE')
        mses[momentum] = mse
        r2s[momentum] = r2

    for momentum in momentums:
        plt.plot(mses[momentum], label=f'{momentum}')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.legend(loc='lower right')
    plt.title(f'MSE Vs. Momentum')
    plt.show()

    for momentum in momentums:
        plt.plot(r2s[momentum], label=f'{momentum}')
    plt.xlabel('Epoch')
    plt.ylabel('R2')
    plt.legend(loc='lower right')
    plt.title(f'R2 Vs. Momentum')
    plt.show()

def activation_experiment():
    activations = ['ReLu', 'LReLu', 'Sigmoid']

    mses = {}
    r2s = {}

    for activation in activations:
        mse, r2 = california_housing_regression(activation, 0.5, 'MSE')
        mses[activation] = mse
        r2s[activation] = r2

    for activation in activations:
        plt.plot(mses[activation], label=f'{activation}')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.legend(loc='lower right')
    plt.title(f'MSE Vs. Activation')
    plt.show()

    for activation in activations:
        plt.plot(r2s[activation], label=f'{activation}')
    plt.xlabel('Epoch')
    plt.ylabel('R2')
    plt.legend(loc='lower right')
    plt.title(f'R2 Vs. Activation')
    plt.show()

# Adult Dataset
    # 8 Features
    # 48842 Instances

def california_housing_regression(activation, momentum, loss_method):
    
    data = fetch_california_housing()
    
    X, y = pd.DataFrame(data.data), pd.DataFrame(data.target)

    for column in X.columns:
        X[column] = X[column] / X[column].abs().max()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    D_in, H1, H2, H3, H4, D_out = 8, 32, 64, 128, 256, 1
    if activation == 'ReLu':
        net = MLP_Cal_House(D_in, H1, H2, H3, H4, D_out) 
    elif activation == 'LReLu':
        net = MLP_Cal_House_Leaky(D_in, H1, H2, H3, H4, D_out)
    else:
        net = MLP_Cal_House_Sigmoid(D_in, H1, H2, H3, H4, D_out)

    X_train = X_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    # Declare Hyper-Parameters
    epochs = 20
    learning_rate = 0.001
    # momentum = 0.5

    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum)
    # optimizer = optim.Adam(net.parameters(), lr=learning_rate)

    if loss_method == 'MSE':
        criterion = torch.nn.MSELoss()
    elif loss_method == 'Smooth L1':
        criterion = torch.nn.SmoothL1Loss()
    else: # MAE
        criterion = torch.nn.L1Loss()

    training_loss = []
    
    training_mse = []
    training_r2 = []

    # Train
    for epoch in tqdm(range(epochs)):
        # print('==='*30)
        # print(f'Epoch {epoch+1}/{epochs}')
        number_correct = 0
        loss_total = 0
        y_pred = np.zeros(len(y_train), dtype=float)
        # for i in tqdm(range(len(X_train))):
        for i in range(len(X_train)):
            X_var = Variable(torch.tensor(X_train.loc[i,:], dtype=torch.float))
            Y_var = Variable(torch.tensor(y_train.loc[i,:], dtype=torch.float))
            optimizer.zero_grad()
            output = net(X_var)
            y_pred[i] = output.item()
            loss = criterion(output, Y_var)
            loss_total = loss_total + loss.item()
            loss.backward()
            optimizer.step()

        epoch_mse = mean_squared_error(y_train, y_pred)
        epoch_r2 = r2_score(y_train, y_pred)
        training_mse.append(epoch_mse)
        training_r2.append(epoch_r2)

        # print(f'Training MSE: {epoch_mse}, Training Loss: {loss_total}')

        training_loss.append(loss_total)

    # plot_data_line(training_mse, 'Training MSE', 'Epoch', 'MSE', 'lower right')
    # plt.plot(training_mse, label='Training MSE')
    # plt.plot(training_r2, label='Training R2')
    # plt.xlabel('Epoch')
    # plt.ylabel('MSE/R2')
    # plt.legend(loc='lower right')
    # plt.title(f'MSE & R2 with {loss_method} Loss')
    # plt.show()

    correct = 0
    y_pred_test = []
    predictions_classified = []
    with torch.no_grad():
        for i in range(len(X_test)):
            X_var = Variable(torch.tensor(X_test.loc[i, :], dtype=torch.float))
            output = net(X_var)  # Unsqueeze to add batch dimension
            # predicted = torch.argmax(output)
            y_pred_test.append(output.item())


    print(f'{activation} Activation, {loss_method} Loss, {momentum} Momentum')
    test_mse = mean_squared_error(y_test, y_pred_test)
    test_mae = mean_absolute_error(y_test, y_pred_test)
    test_r2 = r2_score(y_test, y_pred_test)
    print(f'Test MSE: {test_mse:.4f}')
    print(f'Test MAE: {test_mae:.4f}')
    print(f'Test R2: {test_r2:.4f}')

    # plt.plot(training_loss, label='Loss')
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.legend(loc='upper right')
    # plt.show()

    return training_mse, training_r2


if __name__=="__main__":
    main()