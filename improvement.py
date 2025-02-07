##############################################################################
# Introduction to Machine Learning
# Coursework 1 Skeleton code
# Prepared by: Josiah Wang
#
# Your tasks: Complete the train_and_predict() function. 
#             You are free to add any other methods as needed. 
##############################################################################

import numpy as np
import classification_new as classification_new
import classification_new_new as classification_new_new

def train_and_predict(x_train, y_train, x_test, x_val = None, y_val = None):
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
    
    Returns:
    numpy.ndarray: A numpy array of shape (M, ) containing the predicted class label for each instance in x_test
    """

    #######################################################################
    #                 ** TASK 4.1: COMPLETE THIS FUNCTION **
    #######################################################################

    #Comment this out if you want the other improved model   
    #classifier = classification_new.DecisionTreeClassifier(max_depth=None, min_info_gain=0, method="information_gain",post_pruning_x=x_val, post_pruning_y=y_val, post_pruning_accuracy_gain_min=0.7)
    
    #Uncomment this to get the model with the multi-branches!!
    classifier = classification_new_new.DecisionTreeClassifier(max_depth=None, min_info_gain=0, max_branches=5)

    classifier.fit(x_train, y_train)

    y_hat = classifier.predict(x_test)
    return y_hat



