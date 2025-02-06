##############################################################################
# Introduction to Machine Learning
# Coursework 1 Skeleton code
# Prepared by: Josiah Wang
#
# Your tasks: Complete the train_and_predict() function. 
#             You are free to add any other methods as needed. 
##############################################################################

import numpy as np
from classification_new import DecisionTreeClassifier

def load_data(filename):
    data = np.genfromtxt(filename, dtype = str, delimiter = ',')
    attributes = data[:, :-1]
    attributes = attributes.astype(int)

    label = data[:,-1]

    return attributes, label

def train_and_predict(x_train, y_train, x_test, y_test):
    """ Interface to train and test the new/improved decision tree.
    
    This function is an interface for training and testing the new/improved
    decision tree classifier. 

    x_train and y_train should be used to train your classifier, while 
    x_test should be used to test your classifier. 
    x_val and y_val may optionally be used as the validation dataset. 
    You can just ignore x_val and y_val if you do not need a validation dataset.

    Args:
    x_train (numpy.ndarray): Training instances, numpy array of shape (N, K) 
                       N is the number of instances
                       K is the number of attributes
    y_train (numpy.ndarray): Class labels, numpy array of shape (N, )
                       Each element in y is a str 
    x_test (numpy.ndarray): Test instances, numpy array of shape (M, K) 
                            M is the number of test instances
                            K is the number of attributes
    x_val (numpy.ndarray): Validation instances, numpy array of shape (L, K) 
                       L is the number of validation instances
                       K is the number of attributes
    y_val (numpy.ndarray): Class labels of validation set, numpy array of shape (L, )
    
    Returns:
    numpy.ndarray: A numpy array of shape (M, ) containing the predicted class label for each instance in x_test
    """

    #######################################################################
    #                 ** TASK 4.1: COMPLETE THIS FUNCTION **
    #######################################################################
       

    # TODO: Train new classifier

    classifier = DecisionTreeClassifier(max_depth=30, n_branches=5)

    classifier.cross_validation(x_train, y_train, k=5, max_depth=30)

    classifier.fit(x_train, y_train)

    #y_test = x_test[:,-1]
    #x_test = x_test[:,:-1]

    y = y_test
    y_hat = classifier.predict(x_test)
    print(y_hat)
    c = classifier.confusion_matrix(y, y_hat)
    print(c)
    acc = classifier.accuracy(y, y_hat)
    print(acc)       
        
    return y_hat

if __name__ == "__main__":
    x_train, y_train = load_data("data/train_full.txt")
    x_test, y_test = load_data("data/test.txt")

    y_hat = train_and_predict(x_train, y_train, x_test, y_test)


