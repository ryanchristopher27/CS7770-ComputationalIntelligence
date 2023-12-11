# include our Python packages
import torch
from torch.autograd import Variable
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import pandas as pd
from sklearn.preprocessing import LabelEncoder


from multi_layer_perceptron import MLP_Mushroom
from ucimlrepo import fetch_ucirepo 
from helpers import *


def main():
    learning_rate = 0.001
    epochs = 50
    optimizer = "SGD"
    # _, _, _, _ = mushroom_classification(learning_rate, epochs, optimizer)
    # epochs_experiment()
    # learning_rate_experiment()
    optimizer_experiment()

# Mushroom Dataset
    # 22 Features
    # 8124 Instances

def epochs_experiment():
    epochs = [5, 10, 50]

    accuracies = {}
    losses = {}
    y_preds = {}
    y_tests = {}
    
    for epoch in epochs:
        accuracy, loss, y_pred, y_test = mushroom_classification(0.001, epoch, "SGD")
        y_preds[epoch] = y_pred
        y_tests[epoch] = y_test

    labels = ['edible', 'poisonous']
    for epoch in epochs:
        display_confusion_matrix(pd.Series(y_preds[epoch]), pd.Series(y_tests[epoch]), 2, labels, f'{epoch} Epochs')



def learning_rate_experiment():
    learning_rates = [0.01, 0.001, 0.0001]

    accuracies = {}
    losses = {}
    
    for rate in learning_rates:
        accuracy, loss, y_pred, y_test = mushroom_classification(rate, 20, "SGD")
        accuracies[rate] = accuracy
        losses[rate] = loss

    for rate in learning_rates:
        plt.plot(accuracies[rate], label=f'{rate} Accuracy')

    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.title('Accuracy Vs. Learning Rate')
    plt.show()

    for rate in learning_rates:
        plt.plot(losses[rate], label=f'{rate} Loss')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.title('Loss Vs. Learning Rate')
    plt.show()


def optimizer_experiment():
    optimizers = ["SGD", "ADAM"]

    accuracies = {}
    losses = {}
    
    for op in optimizers:
        accuracy, loss, _, _ = mushroom_classification(0.001, 20, op)
        accuracies[op] = accuracy
        losses[op] = loss

    for op in optimizers:
        plt.plot(accuracies[op], label=f'{op} Accuracy')

    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.title('Accuracy Vs. Optimizer')
    plt.show()

    for op in optimizers:
        plt.plot(losses[op], label=f'{op} Loss')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.title('Loss Vs. Optimizer')
    plt.show()



def mushroom_classification(lr, epochs, op):
    
    # fetch dataset 
    mushroom = fetch_ucirepo(id=73) 
    
    # data (as pandas dataframes) 
    X = mushroom.data.features 
    y = mushroom.data.targets 

    # Clean Data
    nan_rows = X.isna().any(axis=1)
    for row, isnan in enumerate(nan_rows):
        if isnan:
            X = X.drop(index=row)
            y = y.drop(index=row)

    
    X = X.astype('category')
    labelencoder=LabelEncoder()
    for column in X.columns:
        X[column] = labelencoder.fit_transform(X[column])

    y = y.astype('category')
    y = labelencoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    D_in, H1, H2, H3, D_out = 22, 64, 128, 256, 1
    net = MLP_Mushroom(D_in, H1, H2, H3, D_out)

    X_train = X_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    # y_train = y_train.reset_index(drop=True)
    # y_test = y_test.reset_index(drop=True)

    # Declare Hyper-Parameters
    # epochs = 20
    learning_rate = lr
    # learning_rate = 0.001
    momentum = 0.5

    if op == "SGD":
        optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum)
    else:
        optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    # criterion = torch.nn.BCELoss()
    # criterion = torch.nn.CrossEntropyLoss()
    criterion = torch.nn.MSELoss()

    model_loss = []
    
    accuracies = []

    # Train
    for epoch in tqdm(range(epochs)):
        # print('==='*30)
        # print(f'Epoch {epoch+1}/{epochs}')
        number_correct = 0
        loss_total = 0
        # for i in tqdm(range(len(X_train))):
        for i in range(len(X_train)):
            X_var = Variable(torch.tensor(X_train.loc[i,:], dtype=torch.float))
            Y_var = Variable(torch.tensor([y_train[i]], dtype=torch.float))
            optimizer.zero_grad()
            output = net(X_var)
            prediction = binary_prediction_classifier(output.item())
            loss = criterion(output, Y_var)
            loss_total = loss_total + loss.item()
            loss.backward()
            optimizer.step()

            if prediction == y_train[i]:
                number_correct += 1

        accuracy = number_correct / len(y_train)
        accuracies.append(accuracy)

        model_loss.append(loss_total)

    # plt.plot(accuracies, label='Training Accuracy')
    # plt.xlabel('Epoch')
    # plt.ylabel('Accuracy')
    # plt.legend(loc='lower right')
    # plt.show()

    correct = 0
    predictions = []
    predictions_classified = []
    with torch.no_grad():
        for i in range(len(X_test)):
            X_var = Variable(torch.tensor(X_test.loc[i, :], dtype=torch.float))
            output = net(X_var)  # Unsqueeze to add batch dimension
            # predicted = torch.argmax(output)
            predictions.append(output.item())

            prediction = binary_prediction_classifier(output.item())
            predictions_classified.append(prediction)
            
            if prediction == y_test[i]:
                correct += 1

    accuracy = correct / len(y_test) * 100
    # print(f'Accuracy: {accuracy:.4f}%')

    # plt.plot(model_loss, label='Loss')
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.legend(loc='lower right')
    # plt.title('Loss per Epoch')
    # plt.show()

    print(f'Results for {op}')
    print_classification_stats(y_test, predictions_classified)
    print('==='*30)

    # labels = ['edible', 'poisonous']
    # display_confusion_matrix(pd.Series(predictions_classified), pd.Series(y_test), 2, labels)

    return accuracies, model_loss, predictions_classified, y_test

if __name__=="__main__":
    main()