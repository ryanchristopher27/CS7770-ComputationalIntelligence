#Imports
from mamdani_fis import FuzzyInferenceSystem
from sklearn import datasets, metrics
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import os

def main():
    # numeric_example()
    # iris_classification()
    iris_classification_multiple_output_domains()


def numeric_example() -> None:
    fis = FuzzyInferenceSystem()
    fis.set_aggregation_method("Sum")

    fis.create_domain("X", 0, 10, 0.1, "i")
    fis.create_domain("Y", 0, 10, 0.1, "o")

    fis.create_trapezoid_mf("X", "A1", 0, 1, 3, 4, "i")
    fis.create_trapezoid_mf("X", "A2", 3, 4, 6, 7, "i")
    fis.create_trapezoid_mf("X", "A3", 6, 7, 9, 10, "i")
    fis.plot_membership_functions("X", True, "plots/numeric_example/X_Domain_MFs.png")

    fis.create_triangle_mf("Y", "B1", 0, 2, 4, "o")
    fis.create_triangle_mf("Y", "B2", 3, 5, 7, "o")
    fis.create_triangle_mf("Y", "B3", 6, 8, 10, "o")
    fis.plot_membership_functions("Y", True, "plots/numeric_example/Y_Domain_MFs.png")

    fis.create_rule("Rule1", ["A1"], "B1")
    fis.create_rule("Rule2", ["A2"], "B2")
    fis.create_rule("Rule3", ["A3"], "B3")

    test_input_1 = {"X": 5}
    inf_1 = fis.evaluate_mamdani(test_input_1)
    class_1 = fis.defuzzification_mamdani([inf_1])
    plot_evaluations(inf_1, 0, 10, 0.1, "X", class_1[0][0], True, "plots/numeric_example/Test_1_Output.png")
    print(f"Centroid Output 1: {class_1}")

    test_input_2 = {"X": 3.5}
    inf_2 = fis.evaluate_mamdani(test_input_2)
    class_2 = fis.defuzzification_mamdani([inf_2])
    plot_evaluations(inf_2, 0, 10, 0.1, "X", class_2[0][0], True, "plots/numeric_example/Test_2_Output.png")
    print(f"Centroid Output 2: {class_2}")
    
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
    # plot_iris_data(iris, True, "project_2_fuzzy_sets/mamdani/plots/iris_single_output_domain/iris_data.png")
    # plot_iris_data_specific(iris, 2, 3, True, "project_2_fuzzy_sets/mamdani/plots/iris_single_output_domain/iris_data_pw_pl.png")
            
    fis = FuzzyInferenceSystem()
    fis.set_aggregation_method("Sum")


    # Add domains
    fis.create_domain("PW", 0, 3, 0.1, "i")
    fis.create_domain("PL", 0, 8, 0.1, "i")
    fis.create_domain("Output", 0, 100, 1, "o")

    # Create Petal Width Membership Functions
    fis.create_trapezoid_mf("PW", "PW_Low", 0, 0, 0.8, 1, "i")
    fis.create_trapezoid_mf("PW", "PW_Mid", 0.8, 0.9, 1.4, 1.7, "i")
    fis.create_trapezoid_mf("PW", "PW_High", 1.1, 1.4, 2.8, 3, "i")
    # fis.plot_membership_functions("PW", True, "project_2_fuzzy_sets/mamdani/plots/iris_single_output_domain/petal_width_mfs.png")

    # Create Petal Length Membership Functions
    fis.create_trapezoid_mf("PL", "PL_Low", 0, 0, 2.3, 2.5, "i")
    fis.create_trapezoid_mf("PL", "PL_Mid", 2.5, 2.7, 4.8, 5, "i")
    fis.create_trapezoid_mf("PL", "PL_High", 4.8, 5, 8, 8, "i")
    # fis.plot_membership_functions("PL", True, "project_2_fuzzy_sets/mamdani/plots/iris_single_output_domain/petal_length_mfs.png")

    # Create Output Membership Functions
    fis.create_trapezoid_mf("Output", "Setosa", 0, 10, 30, 40, "o")
    fis.create_trapezoid_mf("Output", "Versicolor", 30, 40, 60, 70, "o")
    fis.create_trapezoid_mf("Output", "Virginica", 60, 70, 90, 100, "o")
    # fis.plot_membership_functions("Output", True, "project_2_fuzzy_sets/mamdani/plots/iris_single_output_domain/output_mfs.png")
             
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
        if i == 106:
            print('check')
        evaluations.append(fis.evaluate_mamdani(data))
    
    iris_classification = fis.defuzzification_mamdani(evaluations)
    iris_classification = [class_dictionary[classify_iris(x[0])] for x in iris_classification]

    correct = 0
    # Compare labels to classification
    for i in range(X.shape[0]):
        if Y[i] == iris_classification[i]:
            correct += 1
    
    accuracy = correct / 150 * 100

    print(f"Accuracy: {accuracy}%")

    # plot_confusion_matrix(Y, iris_classification, True, "project_2_fuzzy_sets/mamdani/plots/iris_single_output_domain/confusion_matrix.png")


# Iris Classification With Multiple Output Domains
def iris_classification_multiple_output_domains() -> None:
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
    # plot_iris_data_specific(iris, 0, 1)
    # plot_iris_data_specific(iris, 2, 3)

            
    fis = FuzzyInferenceSystem()
    fis.set_aggregation_method("Sum")


    # Add domains
    fis.create_domain("PW", 0, 3, 0.1, "i")
    fis.create_domain("PL", 0, 8, 0.1, "i")
    fis.create_domain("SW", 0, 5, 0.1, "i") 
    fis.create_domain("SL", 0, 9, 0.1, "i")
    # fis.create_domain("Output", 0, 100, 1)
    fis.create_domain("Setosa", 0, 100, 1, "o")
    fis.create_domain("Versicolor", 0, 100, 1, "o")
    fis.create_domain("Virginica", 0, 100, 1, "o")


    # Create Petal Width Membership Functions
    fis.create_trapezoid_mf("PW", "PW_Low", 0, 0, 0.8, 1, "i")
    fis.create_trapezoid_mf("PW", "PW_Mid", 0.8, 0.9, 1.4, 1.7, "i")
    fis.create_trapezoid_mf("PW", "PW_High", 1.1, 1.4, 2.8, 3, "i")
    # fis.plot_membership_functions("PW", True, "project_2_fuzzy_sets/mamdani/plots/iris_single_output_domain/petal_width_mfs.png")

    # Create Petal Length Membership Functions
    fis.create_trapezoid_mf("PL", "PL_Low", 0, 0, 2.3, 2.5, "i")
    fis.create_trapezoid_mf("PL", "PL_Mid", 2.5, 2.7, 4.8, 5, "i")
    fis.create_trapezoid_mf("PL", "PL_High", 4.8, 5, 8, 8, "i")
    # fis.plot_membership_functions("PL", True, "project_2_fuzzy_sets/mamdani/plots/iris_single_output_domain/petal_length_mfs.png")

    # Create Sepal Width Membership Functions
    fis.create_trapezoid_mf("SW", "SW_Low", 0, 1.8, 3.5, 3.6, "i") # Versicolor
    fis.create_trapezoid_mf("SW", "SW_Mid", 2.2, 2.5, 3.5, 3.8, "i") # Virginica
    fis.create_trapezoid_mf("SW", "SW_High", 2.3, 3.0, 4.4, 4.5, "i") # Setosa
    # fis.plot_membership_functions("SW", True, "project_2_fuzzy_sets/mamdani/plots/iris_multiple_output_domains/sepal_width_mfs.png")

    # Create Sepal Length Membership Functions
    fis.create_trapezoid_mf("SL", "SL_Low", 4.0, 4.2, 5.7, 5.9, "i") # Setosa
    fis.create_trapezoid_mf("SL", "SL_Mid", 4.8, 5.0, 6.8, 7.1, "i") # Versicolor
    fis.create_trapezoid_mf("SL", "SL_High", 4.8, 5.6, 7.8, 8, "i") # Virginica
    # fis.plot_membership_functions("SL", True, "project_2_fuzzy_sets/mamdani/plots/iris_multiple_output_domains/sepal_length_mfs.png")

    # Create Output Membership Functions
    fis.create_trapezoid_mf("Setosa", "Setosa_Low", 0, 10, 30, 40, "o")
    fis.create_trapezoid_mf("Setosa", "Setosa_Mid", 30, 40, 60, 70, "o")
    fis.create_trapezoid_mf("Setosa", "Setosa_High", 60, 70, 90, 100, "o")
    # fis.plot_membership_functions("Setosa", True, "project_2_fuzzy_sets/mamdani/plots/iris_multiple_output_domains/setosa_mfs.png")
    fis.create_trapezoid_mf("Versicolor", "Versicolor_Low", 0, 10, 30, 40, "o")
    fis.create_trapezoid_mf("Versicolor", "Versicolor_Mid", 30, 40, 60, 70, "o")
    fis.create_trapezoid_mf("Versicolor", "Versicolor_High", 60, 70, 90, 100, "o")
    # fis.plot_membership_functions("Versicolor", True, "project_2_fuzzy_sets/mamdani/plots/iris_multiple_output_domains/versicolor_mfs.png")
    fis.create_trapezoid_mf("Virginica", "Virginica_Low", 0, 10, 30, 40, "o")
    fis.create_trapezoid_mf("Virginica", "Virginica_Mid", 30, 40, 60, 70, "o")
    fis.create_trapezoid_mf("Virginica", "Virginica_High", 60, 70, 90, 100, "o")
    # fis.plot_membership_functions("Virginica", True, "project_2_fuzzy_sets/mamdani/plots/iris_multiple_output_domains/virginica_mfs.png")

    # Create Specific Membership Functions for last four points
    fis.create_triangle_mf("PW", "PW_P70", 1.7, 1.8, 1.9, "i")
    fis.create_triangle_mf("PL", "PL_P70", 4.7, 4.8, 4.9, "i")
    fis.create_triangle_mf("SW", "SW_P70", 3.1, 3.2, 3.3, "i")
    fis.create_triangle_mf("SL", "SL_P70", 5.8, 5.9, 6.0, "i")
    fis.create_triangle_mf("PW", "PW_P77", 1.6, 1.7, 1.8, "i")
    fis.create_triangle_mf("PL", "PL_P77", 4.9, 5.0, 5.1, "i")
    fis.create_triangle_mf("SW", "SW_P77", 2.9, 3.0, 3.1, "i")
    fis.create_triangle_mf("SL", "SL_P77", 6.6, 6.7, 6.8, "i")
    fis.create_triangle_mf("PW", "PW_P83", 1.5, 1.6, 1.7, "i")
    fis.create_triangle_mf("PL", "PL_P83", 5.0, 5.1, 5.2, "i")
    fis.create_triangle_mf("SW", "SW_P83", 2.6, 2.7, 2.8, "i")
    fis.create_triangle_mf("SL", "SL_P83", 5.9, 6.0, 6.1, "i")
    fis.create_triangle_mf("PW", "PW_P106", 1.6, 1.7, 1.8, "i")
    fis.create_triangle_mf("PL", "PL_P106", 4.4, 4.5, 4.6, "i")
    fis.create_triangle_mf("SW", "SW_P106", 2.4, 2.5, 2.6, "i")
    fis.create_triangle_mf("SL", "SL_P106", 4.8, 4.9, 5.0, "i")

             
    # Create Rule Sets

    # Rule Set 1: 97.33 %
        # Only Petal Length and Width
    # fis.create_rule("Rule1", ["PW_Low", "PL_Low"], "Setosa_High")
    # fis.create_rule("Rule2", ["PW_Mid", "PL_Mid"], "Versicolor_High")
    # fis.create_rule("Rule3", ["PW_High", "PL_High"], "Virginica_High")
    # fis.create_rule("Rule4", ["PW_Low"], "Setosa_Mid")
    # fis.create_rule("Rule5", ["PL_Low"], "Setosa_Mid")
    # fis.create_rule("Rule6", ["PW_Mid"], "Versicolor_Mid")
    # fis.create_rule("Rule7", ["PL_Mid"], "Versicolor_Mid")
    # fis.create_rule("Rule8", ["PW_High"], "Virginica_Mid")
    # fis.create_rule("Rule9", ["PL_High"], "Virginica_Mid")

    # Try 2
    # fis.create_rule("Rule1", ["PW_Low", "PL_Low"], "Setosa_High")
    # fis.create_rule("Rule2", ["PW_Mid", "PL_Mid"], "Versicolor_High")
    # fis.create_rule("Rule3", ["PW_High", "PL_High"], "Virginica_High")
    # fis.create_rule("Rule4", ["PW_Low"], "Setosa_Mid")
    # fis.create_rule("Rule5", ["PL_Low"], "Setosa_Mid")
    # fis.create_rule("Rule6", ["PW_Mid"], "Versicolor_Mid")
    # fis.create_rule("Rule7", ["PL_Mid"], "Versicolor_Mid")
    # fis.create_rule("Rule8", ["PW_High"], "Virginica_Mid")
    # fis.create_rule("Rule9", ["PL_High"], "Virginica_Mid")
    # fis.create_rule("Rule10", ["PW_Low", "PL_Low"], "Versicolor_Low")
    # fis.create_rule("Rule11", ["PW_Low", "PL_Low"], "Virginica_Low")
    # fis.create_rule("Rule12", ["PW_Mid", "PL_Mid"], "Setosa_Low")
    # fis.create_rule("Rule13", ["PW_Mid", "PL_Mid"], "Virginica_Low")
    # fis.create_rule("Rule14", ["PW_High", "PL_High"], "Setosa_Low")
    # fis.create_rule("Rule15", ["PW_High", "PL_High"], "Versicolor_Low")
    # fis.create_rule("Rule16", ['SW_High', 'SL_Low'], "Versicolor_Low") # Setosa
    # fis.create_rule("Rule17", ['SW_High', 'SL_Low'], "Virginica_Low") # Setosa
    # fis.create_rule("Rule18", ['SW_Low', 'SL_Mid'], "Setosa_Low") # Versicolor
    # fis.create_rule("Rule19", ['SW_Low', 'SL_Mid'], "Virginica_Low") # Versicolor
    # fis.create_rule("Rule20", ['SW_Mid', 'SL_High'], "Setosa_Low") # Virginica
    # fis.create_rule("Rule21", ['SW_Mid', 'SL_High'], "Versicolor_Low") # Virginica

    # Try 3
    fis.create_rule("Rule1", ["PW_Low", "PL_Low"], "Setosa_High")
    fis.create_rule("Rule2", ["PW_Mid", "PL_Mid"], "Versicolor_High")
    fis.create_rule("Rule3", ["PW_High", "PL_High"], "Virginica_High")
    fis.create_rule("Rule4", ["PW_Low"], "Setosa_Mid")
    fis.create_rule("Rule5", ["PL_Low"], "Setosa_Mid")
    fis.create_rule("Rule6", ["PW_Mid"], "Versicolor_Mid")
    fis.create_rule("Rule7", ["PL_Mid"], "Versicolor_Mid")
    fis.create_rule("Rule8", ["PW_High"], "Virginica_Mid")
    fis.create_rule("Rule9", ["PL_High"], "Virginica_Mid")
    fis.create_rule("Rule10", ["PW_Low", "PL_Low", "SW_High", "SL_Low"], "Setosa_High")
    fis.create_rule("Rule11", ["PW_Mid", "PL_Mid", "SW_Low", "SL_Mid"], "Versicolor_High")
    fis.create_rule("Rule12", ["PW_High", "PL_High", "SW_Mid", "SL_High"], "Virginica_High")

    # Try 4 - Kind of Cheating
    # fis.create_rule("Rule1", ["PW_Low", "PL_Low"], "Setosa_High")
    # fis.create_rule("Rule2", ["PW_Mid", "PL_Mid"], "Versicolor_High")
    # fis.create_rule("Rule3", ["PW_High", "PL_High"], "Virginica_High")
    # fis.create_rule("Rule4", ["PW_Low"], "Setosa_Mid")
    # fis.create_rule("Rule5", ["PL_Low"], "Setosa_Mid")
    # fis.create_rule("Rule6", ["PW_Mid"], "Versicolor_Mid")
    # fis.create_rule("Rule7", ["PL_Mid"], "Versicolor_Mid")
    # fis.create_rule("Rule8", ["PW_High"], "Virginica_Mid")
    # fis.create_rule("Rule9", ["PL_High"], "Virginica_Mid")
    # fis.create_rule("Rule10", ["PW_P70", "PL_P70", "SW_P70", "SL_P70"], "Versicolor_High")
    # fis.create_rule("Rule11", ["PW_P77", "PL_P77", "SW_P77", "SL_P77"], "Versicolor_High")
    # fis.create_rule("Rule12", ["PW_P83", "PL_P83", "SW_P83", "SL_P83"], "Versicolor_High")
    # fis.create_rule("Rule13", ["PW_P106", "PL_P106", "SW_P106", "SL_P106"], "Virginica_High")

    # Iris Evaluation
    X = iris.data
    Y = iris.target

    evaluations = []
    for i in range(X.shape[0]):
        x = X[i]
        data = {"SL": x[0], "SW": x[1], "PL": x[2], "PW": x[3]}
        evaluations.append(fis.evaluate_mamdani(data))
    
    iris_classification = fis.defuzzification_mamdani(evaluations)

    iris_classification = [class_dictionary[x[1]] for x in iris_classification]

    correct = 0
    wrong_indices = []
    # Compare labels to classification
    for i in range(X.shape[0]):
        if Y[i] == iris_classification[i]:
            correct += 1
        else:
            wrong_indices.append(i)
    
    accuracy = correct / 150 * 100

    print(f"Accuracy: {accuracy}%")
    print(f'Wrong Indices: {wrong_indices}')

    plot_confusion_matrix(Y, iris_classification, True, "project_2_fuzzy_sets/mamdani/plots/iris_multiple_output_domains/confusion_matrix.png")


# Iris Helper Functions
def classify_iris(val :float) -> str:
    if val <= 33.3:
        return "Setosa"
    elif val > 33.3 and val <= 66.6:
        return "Versicolor"
    else:
        return "Virginica"
    

# Plotting Functions
# -----------------------------------------------------------------------------------
def plot_iris_data(iris, save :bool = False, save_path :str = "") -> None:
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
    
    if save:
        # os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path)

    plt.show()

def plot_iris_data_specific(iris, x :int, y :int, save :bool = False, save_path :str = "") -> None:
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

    if save:
        plt.savefig(save_path)

    plt.show()

def plot_confusion_matrix(actual :[], predicted :[], save :bool = False, save_path :str = "") -> None:
    display_labels = ['Setosa', 'Versicolor', 'Virginica']
    confusion_matrix = metrics.confusion_matrix(actual, predicted)
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = display_labels)
    cm_display.plot()

    if save:
        plt.savefig(save_path)

    plt.show() 

def plot_evaluations(evaluations :{}, d_start :float, d_stop :float, d_step :float, domain :str, centroid :float, save :bool = False, save_path :str = "") -> None:
    plt.figure(figsize=(8, 5))
    colors = ['skyblue', 'red', 'green', 'yellow', 'orange']
    color_i = 0
    for key, val in evaluations.items():
        plt.plot(np.arange(d_start, d_stop, d_step), val, 'k')
        plt.fill_between(np.arange(d_start, d_stop, d_step), val, color=colors[color_i], alpha=0.4, label=key)
        color_i += 1

    plt.axvline(x=centroid, color='black', linestyle='--', label=f'Centroid: {round(centroid, 1)}')
    plt.legend()  # Add a legend to show the centroid line
    plt.title(f"{domain} Plot")
    plt.ylabel('Fuzzy membership')
    plt.xlabel('The domain of interest')
    plt.ylim(-0.1, 1.1)

    if save:
        plt.savefig(save_path)

    plt.show()

if __name__=="__main__":
    main()