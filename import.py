
import numpy as np
import classification
from scipy.stats import mode

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

def check_labels(x1, y1, x2, y2):
    if len(x1) != len(x2):
        print("The two datasets are not the same size")
        return
    count = 0
    for j in range(0, len(x1)):
        for i in range(0, len(x2)):
            if np.array_equal(x1[j], x2[i]):
                if y1[j] == y2[i]:
                    count += 1
                break
    perc_same = count / len(x1)
    return round(perc_same, 2)

def question_4():
    print("*** We now try to average across all trees from each cross-validation fold ***")

    x, y = load_data("data/train_full.txt")
    x_test, y_test = load_data("data/test.txt")

    classifier = classification.DecisionTreeClassifier(method = "mean")
    classifiers = classifier.cross_validation(x, y)

    # Collect predictions from all trees
    predictions = []
    for tree in classifiers:
        y_hat = tree.predict(x_test)
        predictions.append(y_hat)

    # Convert predictions to numpy array
    predictions = np.array(predictions)
    predictions = np.transpose(predictions)
    
    print(predictions)

    modes = []
    
    # Loop over each row.
    for row in predictions:
        # Find unique elements and their counts in the row.
        unique_elements, counts = np.unique(row, return_counts=True)
        # The mode is the element with the highest count.
        mode_value = unique_elements[np.argmax(counts)]
        modes.append(mode_value)
        
    ensemble_predictions = np.array(modes)

    # Calculate metrics...
    print("\nEnsemble Model Performance:")
    c = classifier.confusion_matrix(y_test, ensemble_predictions)
    print("The confusion matrix is: ", c)
    acc = classifier.accuracy(y_test, ensemble_predictions)
    print("The accuracy is: ", acc)

    classes = np.sort(np.unique(np.concatenate((y_test, ensemble_predictions))))
    for i in range(len(classes)):
        p = round(classifier.precision(y_test, ensemble_predictions, classes[i]), 2)
        r = round(classifier.recall(y_test, ensemble_predictions, classes[i]), 2)
        f = round(classifier.f1_score(y_test, ensemble_predictions, classes[i]), 2)
        print(f"Class {classes[i]} has Precision: ", p, "Recall: ", r, "F-measure: ", f)

    macro_acc = round(classifier.macro_calculator(classifier.precision, y_test, ensemble_predictions), 2)
    print(f"With macro-averaged precision: {macro_acc}")
    macro_acc = round(classifier.macro_calculator(classifier.recall, y_test, ensemble_predictions), 2)
    print(f"With macro-averaged recall: {macro_acc}")
    macro_acc = round(classifier.macro_calculator(classifier.f1_score, y_test, ensemble_predictions), 2)
    print(f"With macro-averaged F1-score: {macro_acc}")


if __name__ == "__main__":
    print("*** QUESTION 1.1 ***")
    print()
    x_full, y_full = load_data("data/train_full.txt")
    x_sub, y_sub = load_data("data/train_sub.txt")
    data_shape(x_full)
    data_shape(x_sub)
    unique_classes(y_full)
    unique_classes(y_sub)
    #attribute_analysis(x_full)

    print("*** QUESTION 1.2 ***")

    print("We now analyse the noisy dataset and compare to full dataset")
    x_noisy, y_noisy = load_data("data/train_noisy.txt")

    print("*** QUESTION 1.3 ***")
    
    #same = check_labels(x_full, y_full, x_noisy, y_noisy)
    #print("The percentage of same labels is: ", same)
    #unique_classes(y_noisy)

    print("*** QUESTION 3 ***")

    print("Training the decision tree...")


    datasets = ["full", "sub", "noisy"]

    for dataset in datasets:
        x, y = load_data("data/train_" + dataset + ".txt")
        print("***************************************************************************************")
        print("The dataset is: ", dataset)

        classifier = classification.DecisionTreeClassifier(max_depth=None, min_info_gain=0, method="mean")

        classifiers = classifier.cross_validation(x, y)

        classifier.fit(x, y)

        x_test, y_test = load_data("data/test.txt")

        y = y_test
        y_hat = classifier.predict(x_test)
        c = classifier.confusion_matrix(y, y_hat)
        
       
        print("The confusion matrix is: ", c)
        acc = classifier.accuracy(y, y_hat)
        print("The accuracy is: ",acc)

        classes = np.sort(np.unique(np.concatenate((y, y_hat))))

        for i in range(0, len(classes)):
            p = round(classifier.precision(y, y_hat, classes[i]), 2)
            r = round(classifier.recall(y, y_hat, classes[i]),2)
            f = round(classifier.f1_score(y, y_hat, classes[i]),2)
            print(f"Class {classes[i]} has Precision: ", p, "Recall: ", r, "F-measure: ", f)
        
        macro_acc = round(classifier.macro_calculator(classifier.precision, y, y_hat),2)
        print(f"With macro-averaged precision: {macro_acc}")
        macro_acc = round(classifier.macro_calculator(classifier.recall, y, y_hat),2)
        print(f"With macro-averaged recall: {macro_acc}")
        macro_acc = round(classifier.macro_calculator(classifier.f1_score, y, y_hat),2)
        print(f"With macro-averaged F1-score: {macro_acc}")

    question_4()
    



    


