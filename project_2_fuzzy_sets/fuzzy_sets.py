#Imports
from fuzzy_inference_system import FuzzyInferenceSystem
from sklearn import datasets
import matplotlib.pyplot as plt

def main():
    iris_classification()


def iris_classification() -> None:
    iris = datasets.load_iris()
    # plot_iris_data(iris)

    iris_dictionary = {"sepal_length": 0, 
                       "sepal_width": 1,
                       "petal_length": 2,
                       "petal_width": 3}
    
    class_dictionary = {"Setosa": 0,
                        "Versicolor": 1,
                        "Virginica": 2}

    # Rules
    # ------------------------------
        # 1) If Petal Width < 0.9
            # Petal Length < 2.5
                # Setosa
        # 2) If Petal Length < 4.5
            # Petal Length > 2.5
                # Versicolor
             
    # Petal Width
    pw_fis = FuzzyInferenceSystem("Petal Width", 0, 3, 0.1)
    pw_fis.create_trapezoid_mf("Setosa", 0, 0, 0.8, 1)
    pw_fis.create_trapezoid_mf("Versicolor", 0.8, 0.9, 1.5, 1.8)
    pw_fis.create_trapezoid_mf("Virginica", 1.4, 1.8, 2.8, 3)
    # pw_fis.plot_membership_functions()

    # Petal Length
    pl_fis = FuzzyInferenceSystem("Petal Length", 0, 8, 0.1)
    pl_fis.create_trapezoid_mf("Setosa", 0, 0, 2.3, 2.5)
    pl_fis.create_trapezoid_mf("Versicolor", 2.5, 2.7, 4.2, 4.4)
    pl_fis.create_trapezoid_mf("Virginica", 4.1, 4.3, 8, 8)
    # pl_fis.plot_membership_functions()

    # Iris Evaluation
    X = iris.data
    Y = iris.target

    pw_eval = []
    pl_eval = []
    for i in range(X.shape[0]):
        x = X[i]
        
        # Petal Width Evaluation
        pw_eval.append(pw_fis.evaluate_mamdani(x[iris_dictionary["petal_width"]]))

        # Petal Length Evaluation
        pl_eval.append(pl_fis.evaluate_mamdani(x[iris_dictionary["petal_length"]]))

    pw_class = pw_fis.classify_mamdani(pw_eval)
    pl_class = pl_fis.classify_mamdani(pl_eval)

    iris_classification = []
    for i in range(X.shape[0]):
        # Take highest value from rules
        if pw_class[i][1] > pl_class[i][1]:
            iris_classification.append(class_dictionary[pw_class[i][0]])
        else:
            iris_classification.append(class_dictionary[pl_class[i][0]])

    correct = 0
    # Compare labels to classification
    for i in range(X.shape[0]):
        if Y[i] == iris_classification[i]:
            correct += 1
    
    accuracy = correct / 150 * 100



    print(f"Accuracy: {accuracy}%")


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