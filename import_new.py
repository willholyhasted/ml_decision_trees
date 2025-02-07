import numpy as np
import improvement
from classification import DecisionTreeClassifier

def load_data(filename):
    data = np.genfromtxt(filename, dtype = str, delimiter = ',')
    attributes = data[:, :-1]
    attributes = attributes.astype(int)

    label = data[:,-1]

    return attributes, label

def data_shape(data):
    rows = len(data)
    rows = data.shape[0] #An alternative way of counting rows
    print("There are ", rows, " observations")

    cols = data.shape[1]
    print("There are ", cols, " columns")

def unique_classes(data):
    classes, count = np.unique(data, return_counts = True)
    count = count/(np.count_nonzero(data))
    print(classes, count)


def attribute_analysis(attribute_data):
    for i in range(0, attribute_data.shape[1]):
        max = np.max(attribute_data[:,i])
        min = np.min(attribute_data[:,i])
        print("Min value of column", i," is:", min)
        print("Max value of column", i," is:", max)
        print("The range is:", max - min )


def compare_rows(d1, d2):
    new = (d1 != d2).any(axis = 1)
    return new

if __name__ == "__main__":
    x_validation, y_validation = load_data("data/validation.txt")
    datasets = ["train_full", "train_sub", "train_noisy", "validation"]
    x_test, y_test = load_data("data/test.txt")

    for dataset in datasets:
        x, y = load_data("data/" + dataset + ".txt")

        y_hat = improvement.train_and_predict(x, y, x_test, x_validation, y_validation)

        classifier = DecisionTreeClassifier()

        c = classifier.confusion_matrix(y_test, y_hat)
        
        print("***************************************************************************************")
        print("The dataset is: ", dataset)
        print("The confusion matrix is: ", c)
        acc = classifier.accuracy(y_test, y_hat)
        print("The accuracy is: ",acc)

        
        classes = np.unique(np.concatenate((y_test, y_hat)))

        for i in range(0, len(classes)):
            p = round(classifier.precision(y_test, y_hat, classes[i]), 2)
            r = round(classifier.recall(y_test, y_hat, classes[i]),2)
            f = round(classifier.f1_score(y_test, y_hat, classes[i]),2)
            print(f"Class {classes[i]} has Precision: ", p, "Recall: ", r, "F-measure: ", f)
        
        macro_acc = round(classifier.macro_calculator(classifier.precision, y_test, y_hat),2)
        print(f"With macro-averaged precision: {macro_acc}")
        macro_acc = round(classifier.macro_calculator(classifier.recall, y_test, y_hat),2)
        print(f"With macro-averaged recall: {macro_acc}")
        macro_acc = round(classifier.macro_calculator(classifier.f1_score, y_test, y_hat),2)
        print(f"With macro-averaged F1-score: {macro_acc}")
        



    



    


