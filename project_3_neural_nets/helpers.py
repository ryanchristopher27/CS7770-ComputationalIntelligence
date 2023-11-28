import seaborn as sn  # yes, I had to "conda install seaborn"
import pandas as pd
import matplotlib.pyplot as plt
import torch
import numpy as np

def display_confusion_matrix(predictions, y, size):
    ConfusionMatrix = torch.zeros((size,size))
    for i in range(len(predictions)):
        prediction = int(predictions[i])
        label = int(y.iloc[i])
        ConfusionMatrix[label,prediction] += 1

    df_cm = pd.DataFrame(np.asarray(ConfusionMatrix), index = [i for i in range(size)],
                    columns = [i for i in range(size)])
    plt.figure(figsize = (10,7))
    sn.heatmap(df_cm, annot=True)
    plt.show()   