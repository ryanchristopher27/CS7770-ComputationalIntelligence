import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def display_confusion_matrix(predictions, y, size, labels, title):
    ConfusionMatrix = torch.zeros((size,size))
    for i in range(len(predictions)):
        prediction = int(predictions[i])
        label = int(y.iloc[i])
        ConfusionMatrix[label,prediction] += 1

    df_cm = pd.DataFrame(np.asarray(ConfusionMatrix), index = [i for i in range(size)],
                    columns = [i for i in range(size)])
    plt.figure(figsize = (10,7))
    sn.heatmap(df_cm, annot=True, fmt='g', xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.show()   

def binary_prediction_classifier(prediction :float):
    return 1 if prediction >= 0.5 else 0


def plot_data_line(data, plot_label, x_label, y_label, legend_loc) -> None:
    plt.plot(data, label=plot_label)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend(loc=legend_loc)
    plt.show()


def print_classification_stats(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    print(f'Accuracy: {accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1: {f1:.4f}')
