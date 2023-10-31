#Imports
from fuzzy_inference_system import FuzzyInferenceSystem
from sklearn import datasets, metrics
import matplotlib.pyplot as plt
import numpy as np
def main():
    iris_classification()

def iris_classification() -> None:
    iris = datasets.load_iris()

    iris_dictionary = {"sepal_length": 0, 
                       "sepal_width": 1,
                       "petal_length": 2,
                       "petal_width": 3}
    
    class_dictionary = {"Setosa": 0,
                        "Versicolor": 1,
                        "Virginica": 2}
    
    # Plot Iris Data
    # plot_iris_data(iris)
    # plot_iris_data_specific(iris, 3, 2)
            
    fis = FuzzyInferenceSystem()

    # Add domains
    fis.create_domain("PW", 0, 3, 0.01)
    fis.create_domain("PL", 0, 8, 0.01)
    fis.create_domain("Output", 0, 100, 1)

    # Create Petal Width Membership Functions
    fis.create_trapezoid_mf("PW", "PW_Low", 0, 0, 0.8, 1, "i")
    fis.create_trapezoid_mf("PW", "PW_Mid", 0.8, 0.9, 2, 2.2, "i")
    fis.create_trapezoid_mf("PW", "PW_High", 1.6, 2.3, 2.8, 3, "i")
    # fis.plot_membership_functions("PW")

    # Create Petal Length Membership Functions
    fis.create_trapezoid_mf("PL", "PL_Low", 0, 0, 2.3, 2.5, "i")
    fis.create_trapezoid_mf("PL", "PL_Mid", 2.5, 2.7, 4.4, 4.9, "i")
    fis.create_trapezoid_mf("PL", "PL_High", 4.5, 4.9, 8, 8, "i")
    # fis.plot_membership_functions("PL")

    # Create Output Membership Functions
    fis.create_trapezoid_mf("Output", "Setosa", 0, 10, 30, 40, "o")
    fis.create_trapezoid_mf("Output", "Versicolor", 30, 40, 60, 70, "o")
    fis.create_trapezoid_mf("Output", "Virginica", 60, 70, 90, 100, "o")
             
    # Rules
    # ------------------------------
        # 1) If Petal Width Low
        #    And Petal Length Low
                # Setosa
        # 2) If Petal Length Mid
        #    Petal Width Mid
                # Versicolor
        # 3) If Petal Length High
        #    Petal Width Hight
                # Virginica

    # Create Rules
    fis.create_rule("Rule1", ["PW_Low", "PL_Low"], "Setosa")
    fis.create_rule("Rule2", ["PW_Mid", "PL_Mid"], "Versicolor")
    fis.create_rule("Rule3", ["PW_High", "PL_High"], "Virginica")
    
    # Iris Evaluation
    X = iris.data
    Y = iris.target

    evaluations = []
    for i in range(X.shape[0]):
        x = X[i]
        data = {"SL": x[0], "SW": x[1], "PL": x[2], "PW": x[3]}
        evaluations.append(fis.evaluate_mamdani(data))
    
    iris_classification = fis.defuzzification_mamdani(evaluations)

    correct = 0
    # Compare labels to classification
    for i in range(X.shape[0]):
        if Y[i] == iris_classification[i]:
            correct += 1
    
    accuracy = correct / 150 * 100

    # print(f"Accuracy: {accuracy}%")

    plot_confusion_matrix(Y, iris_classification)


def plot_iris_data(iris) -> None:
    features = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    fig, axs = plt.subplots(4, 4, sharey=True, figsize=(10, 9))
    for i in range(4):
        for j in range(4):
            X1 = iris.data[:,i]
            X2 = iris.data[:,j]
            y = iris.target
            axs[i,j].scatter(X1, X2, c=y, cmap=plt.cm.Set1, edgecolor='k')
            # plt.xticks(())
            # plt.yticks(())
            axs[i,j].set(xlabel=features[i], ylabel=features[j])
            

    plt.show()

def plot_iris_data_specific(iris, x :int, y :int) -> None:
    features = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    legend = ["Setosa", "Versicolor", "Virginica"]
    X1 = iris.data[:,x]
    X2 = iris.data[:,y]
    c = iris.target
    # plt.scatter(X1, X2, c=c)
    # Create a scatter plot for each class
    for class_label in range(3):
        mask = (c == class_label)
        plt.scatter(X1[mask], X2[mask], label=legend[class_label])

    plt.title(f'{features[x]} vs. {features[y]}')
    plt.xlabel(f'{features[x]}')
    plt.ylabel(f'{features[y]}')
    plt.legend(legend)
    plt.show()

def plot_confusion_matrix(actual :[], predicted :[]) -> None:
    display_labels = ['Setosa', 'Versicolor', 'Virginica']
    confusion_matrix = metrics.confusion_matrix(actual, predicted)
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = display_labels)
    cm_display.plot()
    plt.show() 


if __name__=="__main__":
    main()