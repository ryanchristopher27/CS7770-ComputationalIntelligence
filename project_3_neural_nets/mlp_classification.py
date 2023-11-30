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
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from tqdm import tqdm
import pandas as pd

from multi_layer_perceptron import MLP, MLP_HD, HeartDiseaseNN
from ucimlrepo import fetch_ucirepo 
from helpers import display_confusion_matrix


def main():
    heart_disease_classification()
    # kaggle_example()

# Heart Disease Dataset
    # 13 Features
    # 303 Instances

def heart_disease_classification():
  
    # fetch dataset 
    heart_disease = fetch_ucirepo(id=45) 
    
    # data (as pandas dataframes) 
    X_HD = heart_disease.data.features 
    y_HD = heart_disease.data.targets 

    # Clean Data
    nan_rows = X_HD.isna().any(axis=1)
    for row, isnan in enumerate(nan_rows):
        if isnan:
            X_HD = X_HD.drop(index=row)
            y_HD = y_HD.drop(index=row)
    
    # metadata 
    # print(heart_disease.metadata) 
    
    # variable information 
    # print(heart_disease.variables) 

    X_train, X_test, y_train, y_test = train_test_split(X_HD, y_HD, test_size=0.2)

    D_in, H1, H2, H3, D_out = 13, 64, 128, 256, 5
    net = MLP_HD(D_in, H1, H2, H3, D_out)
    # net = HeartDiseaseNN()

    # def criterion(out, label):
    #     return (label - out)**2
    

    X_train = X_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    y_train_one_hot = np.eye(5, dtype='uint8')[y_train]
    y_train_one_hot = np.squeeze(y_train_one_hot, axis=1).astype(float)

    y_test_one_hot = np.eye(5, dtype='uint8')[y_test]
    y_test_one_hot = np.squeeze(y_test_one_hot, axis=1).astype(float)

    # net = net.float()

    # Declare Hyper-Parameters
    epochs = 1000
    learning_rate = 0.001
    momentum = 0.5

    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum)
    # optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    # criterion = torch.nn.BCELoss()
    criterion = torch.nn.CrossEntropyLoss()

    # Train
    for epoch in tqdm(range(epochs)):
        for i in range(len(X_train)):
            X = Variable(torch.tensor(X_train.loc[i,:]))
            Y = Variable(torch.tensor(y_train_one_hot[i,:]))
            optimizer.zero_grad()
            outputs = net(X)
            loss = criterion(outputs, Y)
            loss.backward()
            optimizer.step()

    correct = 0
    # predictions = np.zeros(len(y_test))
    # for j in range(len(X_test)):
    #     X = Variable(torch.tensor(X_test.loc[j,:]))
    #     pred = int(torch.argmax(net(X)))
    #     # predictions.append(pred)
    #     predictions[j] = pred
    # predictions = np.zeros((len(y_test), 5))
    # for j in range(len(X_test)):
    #     X = Variable(torch.tensor(X_test.loc[j,:]))
    #     pred = net(X)
    #     # predictions.append(pred)
    #     predictions[j,:] = pred

    X_test_tensor = torch.tensor(X_test.values, dtype=float)
    torch.set_grad_enabled(False)
    predictions = net(X_test_tensor)

    predictions_classified = np.zeros(len(predictions))

    for k in range(len(predictions)):
        predictions_classified[k] = torch.argmax(predictions[k])
        if y_test_one_hot[k,int(predictions_classified[k])] == 1:
            correct += 1

    print(f'Accuracy: {correct/len(predictions) * 100 :.4f}%')

    y_test = pd.Series(y_test.num)

    display_confusion_matrix(predictions_classified, y_test, 5)



def kaggle_example():
    # fetch dataset 
    heart_disease = fetch_ucirepo(id=45) 
    
    # data (as pandas dataframes) 
    X = heart_disease.data.features 
    y = heart_disease.data.targets 

    # Clean Data
    nan_rows = X.isna().any(axis=1)
    for row, isnan in enumerate(nan_rows):
        if isnan:
            X = X.drop(index=row)
            y = y.drop(index=row)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

    sc = MinMaxScaler()

    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    print(X_train.shape)

    X_train = torch.tensor(X_train).float()
    X_test = torch.tensor(X_test).float()
    y_train = torch.tensor(y_train.values).long()
    y_test = torch.tensor(y_test.values).long()

    net = HeartDiseaseNN()

    optimizer = optim.AdamW(net.parameters())
    criterion = nn.CrossEntropyLoss()   

    losses = []

    max_test = 0
    best_params = net.state_dict()
    for epoch in range(1, 50):
        optimizer.zero_grad()
        outputs = net(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        _, preds_y = torch.max(outputs, 1)
        train_acc = accuracy_score(y_train, preds_y)
        
        pred_test = net(X_test)
        _, preds_test_y = torch.max(pred_test, 1)
        test_acc = accuracy_score(y_test, preds_test_y)
        print("Epoch {}, Loss: {}, Acc:{:.2f}%, Test Acc: {:.2f}%".format(epoch, loss.item(), 
                                                                        train_acc*100, test_acc * 100))
        if test_acc > max_test:
            max_test = test_acc
            best_params = net.state_dict()
    net.load_state_dict(best_params)

    plt.plot(losses)

    pred_test = net(X_test)
    _, preds_y = torch.max(pred_test, 1)

    accuracy_score(y_test, preds_y)

    print(classification_report(y_test, preds_y))

    confusion_matrix(y_test, preds_y)

if __name__=="__main__":
    main()