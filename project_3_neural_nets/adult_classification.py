# include our Python packages
import torch
from torch.autograd import Variable
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import pandas as pd
from sklearn.preprocessing import LabelEncoder


from multi_layer_perceptron import MLP_Adult
from ucimlrepo import fetch_ucirepo 
from helpers import display_confusion_matrix, binary_prediction_classifier


def main():
    adult_classification()

# Adult Dataset
    # 14 Features
    # 48842 Instances

def adult_classification():
    # fetch dataset 
    adult = fetch_ucirepo(id=2) 
    
    # data (as pandas dataframes) 
    X = adult.data.features 
    y = adult.data.targets 

    # Clean Data
    nan_rows = X.isna().any(axis=1)
    for row, isnan in enumerate(nan_rows):
        if isnan:
            X = X.drop(index=row)
            y = y.drop(index=row)

    
    y.loc[y['income'].str.contains('<=50K.'), 'income'] = '<=50K'
    y.loc[y['income'].str.contains('>50K.'), 'income'] = '>50K'
    
    labelencoder=LabelEncoder()
    X = X.astype('category')
    for column in X.columns:
        X[column] = labelencoder.fit_transform(X[column])
        X[column] = X[column] / X[column].abs().max()


    # X['fnlwgt'] = X['fnlwgt'] / 1000

    y = y.astype('category')
    y = labelencoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    D_in, H1, H2, H3, H4, D_out = 14, 64, 128, 256, 512, 1
    net = MLP_Adult(D_in, H1, H2, H3, H4, D_out) 

    X_train = X_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    # y_train = y_train.reset_index(drop=True)
    # y_test = y_test.reset_index(drop=True)

    # Declare Hyper-Parameters
    epochs = 20
    learning_rate = 0.001
    momentum = 0.5

    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum)
    # optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    # criterion = torch.nn.BCELoss()
    # criterion = torch.nn.CrossEntropyLoss()
    criterion = torch.nn.MSELoss()

    model_loss = []
    
    accuracies = []

    # Train
    for epoch in range(epochs):
        print('==='*30)
        print(f'Epoch {epoch+1}/{epochs}')
        number_correct = 0
        loss_total = 0
        for i in tqdm(range(len(X_train))):
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

        print(f'Training Accuracy: {accuracy}, Training Loss: {loss_total}')

        model_loss.append(loss_total)

    plt.plot(accuracies, label='Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.show()

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
    print(f'Accuracy: {accuracy:.4f}%')

    plt.plot(model_loss, label='Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.show()

    labels = ['Less Than 50K', 'More Than 50K']
    display_confusion_matrix(pd.Series(predictions_classified), pd.Series(y_test), 2, labels)


if __name__=="__main__":
    main()