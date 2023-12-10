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
from sklearn.metrics import mean_squared_error 

# Local Imports
from multi_layer_perceptron import MLP_Cal_House
from helpers import *


def main():
    california_housing_regression()

# Adult Dataset
    # 8 Features
    # 48842 Instances

def california_housing_regression():
    
    data = fetch_california_housing()
    
    X, y = pd.DataFrame(data.data), pd.DataFrame(data.target)

    # Clean Data
    # nan_rows = X.isna().any(axis=1)
    # for row, isnan in enumerate(nan_rows):
    #     if isnan:
    #         X = X.drop(index=row)
    #         y = y.drop(index=row)
    
    # labelencoder=LabelEncoder()
    # X = X.astype('category')
    for column in X.columns:
        # X[column] = labelencoder.fit_transform(X[column])
        X[column] = X[column] / X[column].abs().max()


    # X['fnlwgt'] = X['fnlwgt'] / 1000

    # y = y.astype('category')
    # y = labelencoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    D_in, H1, H2, H3, H4, D_out = 8, 32, 64, 128, 256, 1
    net = MLP_Cal_House(D_in, H1, H2, H3, H4, D_out) 

    X_train = X_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    # Declare Hyper-Parameters
    epochs = 10
    learning_rate = 0.0001
    momentum = 0.5

    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum)
    # optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    # criterion = torch.nn.BCELoss()
    # criterion = torch.nn.CrossEntropyLoss()
    criterion = torch.nn.MSELoss()

    training_loss = []
    
    training_mse = []

    # Train
    for epoch in range(epochs):
        print('==='*30)
        print(f'Epoch {epoch+1}/{epochs}')
        number_correct = 0
        loss_total = 0
        y_pred = np.zeros(len(y_train), dtype=float)
        for i in tqdm(range(len(X_train))):
            X_var = Variable(torch.tensor(X_train.loc[i,:], dtype=torch.float))
            Y_var = Variable(torch.tensor([y_train.loc[i,:]], dtype=torch.float))
            optimizer.zero_grad()
            output = net(X_var)
            y_pred[i] = output.item()
            loss = criterion(output, Y_var)
            loss_total = loss_total + loss.item()
            loss.backward()
            optimizer.step()

        epoch_mse = mean_squared_error(y_train, y_pred)
        training_mse.append(epoch_mse)

        print(f'Training MSE: {epoch_mse}, Training Loss: {loss_total}')

        training_loss.append(loss_total)


    plot_data_line(training_mse, 'Training MSE', 'Epoch', 'MSE', 'lower right')

    correct = 0
    y_pred_test = []
    predictions_classified = []
    with torch.no_grad():
        for i in range(len(X_test)):
            X_var = Variable(torch.tensor(X_test.loc[i, :], dtype=torch.float))
            output = net(X_var)  # Unsqueeze to add batch dimension
            # predicted = torch.argmax(output)
            y_pred_test.append(output.item())


    test_mse = mean_squared_error(y_test, y_pred_test)
    print(f'Test MSE: {test_mse:.4f}')

    plt.plot(training_loss, label='Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.show()


if __name__=="__main__":
    main()