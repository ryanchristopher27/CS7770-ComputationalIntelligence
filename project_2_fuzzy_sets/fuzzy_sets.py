#Imports
from fuzzy_inference_system import FuzzyInferenceSystem
from sklearn import datasets, metrics
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

def main():
    iris_classification()
    # optimization_test()


def optimization_test():
    pw_vals = np.arange(1.4, 2.4, 0.1)
    pl_vals = np.arange(4.3, 5, 0.1)
    inc_vals = np.arange(0.1, 0.5, 0.1)

    best_acc = 0

    best_vals = (0, 0, 0, 0, 0, 0, 0, 0)

    for i, pw_mid in tqdm(enumerate(pw_vals)):
        print(f'{i+1}/{len(pw_vals)}')
        for pw_high in pw_vals:
            for pl_mid in pl_vals:
                for pl_high in pl_vals:
                    for pw_inc in inc_vals:
                        for pl_inc in inc_vals:
                            pw_mid_3 = pw_mid
                            pw_mid_4 = pw_mid + pw_inc
                            pw_high_1 = pw_high - pw_inc
                            pw_high_2 = pw_high
                            pl_mid_3 = pl_mid
                            pl_mid_4 = pl_mid + pl_inc
                            pl_high_1 = pl_high - pl_inc
                            pl_high_2 = pl_high

                            acc = iris_classification(pw_mid_3, pw_mid_4, pw_high_1, pw_high_2, pl_mid_3, pl_mid_4, pl_high_1, pl_high_2)
                            if acc > best_acc:
                                best_acc = acc
                                best_vals = (pw_mid_3, pw_mid_4, pw_high_1, pw_high_2, pl_mid_3, pl_mid_4, pl_high_1, pl_high_2)

    print(f'Best Accuracy: {best_acc}')
    print(f'Combo: {best_vals}')

# def iris_classification(pw_mid_3, pw_mid_4, pw_high_1, pw_high_2, pl_mid_3, pl_mid_4, pl_high_1, pl_high_2) -> float:
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
    fis.create_domain("PW", 0, 3, 0.1)
    fis.create_domain("PL", 0, 8, 0.1)
    fis.create_domain("Output", 0, 100, 1)

    # Create Petal Width Membership Functions
    fis.create_trapezoid_mf("PW", "PW_Low", 0, 0, 0.8, 1, "i")
    fis.create_trapezoid_mf("PW", "PW_Mid", 0.8, 0.9, 1.4, 1.7, "i")
    fis.create_trapezoid_mf("PW", "PW_High", 1.1, 1.4, 2.8, 3, "i")
    # fis.create_trapezoid_mf("PW", "PW_Mid", 0.8, 0.9, pw_mid_3, pw_mid_4, "i")
    # fis.create_trapezoid_mf("PW", "PW_High", pw_high_1, pw_high_2, 2.8, 3, "i")
    # fis.plot_membership_functions("PW")

    # Create Petal Length Membership Functions
    fis.create_trapezoid_mf("PL", "PL_Low", 0, 0, 2.3, 2.5, "i")
    fis.create_trapezoid_mf("PL", "PL_Mid", 2.5, 2.7, 4.8, 5, "i")
    fis.create_trapezoid_mf("PL", "PL_High", 4.8, 5, 8, 8, "i")
    # fis.create_trapezoid_mf("PL", "PL_Mid", 2.5, 2.7, pl_mid_3, pl_mid_4, "i")
    # fis.create_trapezoid_mf("PL", "PL_High", pl_high_1, pl_high_2, 8, 8, "i")
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
        #   Petal Width Hight
                # Virginica

    # Create Rules
        # Rule 1
    fis.create_rule("Rule1", ["PW_Low", "PL_Low"], "Setosa")
        # Rule 2
    fis.create_rule("Rule2", ["PW_Mid", "PL_Mid"], "Versicolor")
        # Rule 3
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

    print(f"Accuracy: {accuracy}%")

    plot_confusion_matrix(Y, iris_classification)

    # return accuracy


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


def test():
    start, end, step = 0, 10.5, 0.1
    fis = FuzzyInferenceSystem("Test", start, end, step)

    fis.create_trapezoid_mf("mf_1", 1, 2, 4, 5)
    # fis.create_trapezoid_mf("mf_2", 4, 5, 6, 8)
    # fis.create_trapezoid_mf("mf_3", 7, 8, 9, 10)

    # fis.create_triangle_mf("mf_7", 1, 2, 4)
    fis.create_triangle_mf("mf_8", 4, 5, 6)
    # fis.create_triangle_mf("mf_9", 7, 8, 9)

    # fis.create_gaussian_mf("mf_4", 2, 1)
    # fis.create_gaussian_mf("mf_5", 4, 1)
    fis.create_gaussian_mf("mf_6", 7, 1)

    fis.plot_membership_functions()


if __name__=="__main__":
    main()