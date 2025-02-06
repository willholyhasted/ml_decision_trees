#############################################################################
# Introduction to Machine Learning
# Coursework 1 Skeleton code
# Prepared by: Josiah Wang
#
# Your tasks: Complete the fit() and predict() methods of DecisionTreeClassifier.
# You are free to add any other methods as needed. 
##############################################################################

import numpy as np


class DecisionTreeClassifier(object):
    """ Basic decision tree classifier
    
    Attributes:
    is_trained (bool): Keeps track of whether the classifier has been trained
    
    Methods:
    fit(x, y): Constructs a decision tree from data X and label y
    predict(x): Predicts the class label of samples X
    prune(x_val, y_val): Post-prunes the decision tree
    """

    def __init__(self, max_depth=None, min_info_gain=0, method="information_gain"):
        self.is_trained = False
        self.depth = 0
        self.max_depth = max_depth
        self.min_info_gain = min_info_gain
        self.method = method

    def entropy(self, y):
        """ Entropy calculation
        
        Parameters:
        y (numpy.ndarray):  Class labels, numpy array of shape (N, )
                            N is the number of instances   
                            Accepts a 2D array but only uses the last column in this case
        Returns: H (float): The entropy of the class labels
        """
        #We make sure to only select the labels
        if y.ndim > 1:
            y = y[:, -1] 

        # Calculate the probabilities for each label
        c, f = np.unique(y, return_counts=True)
        p = f / len(y)

        # Calculate the information
        H = -np.sum(p * (np.log2(p)))

        return H


    def fit(self, x, y, current_depth=0):
        """ Constructs a decision tree classifier from data
        
        Args:
        x (numpy.ndarray): Instances, numpy array of shape (N, K) 
                           N is the number of instances
                           K is the number of attributes
        y (numpy.ndarray): Class labels, numpy array of shape (N, )
                           Each element in y is a str 
        current_depth(int): The current depth of the tree. Should use 0 for the root node
        method(str): The method to use for the split. Can be "median", "mean", or leave blank for optimal split
        """
        
        assert x.shape[0] == len(y), \
            "Training failed. x and y must have the same number of instances."
        
        # Base case 1: If all labels are the same, return the label
        if np.unique(y).size == 1:
            return y[0]

        #Base case 2: If the depth goes beyond max, depth return the most common label
        if self.max_depth is not None and current_depth >= self.max_depth:
            values, counts = np.unique(y, return_counts = True)
            return values[np.argmax(counts)]

        #Calculate the parent entropy
        H_parent = self.entropy(y)
        gains = []

        #Calculate the information gain for each feature
        for i in range(x.shape[1]):
            if self.method == "median":
                m = np.median(x[:,i])
            elif self.method == "mean":
                m = np.mean(x[:,i])
            else: m = self.find_optimal_split(x[:,i], y)

            mask_left = x[:,i] < m
            mask_right = x[:,i] >= m
            
            # Skip if split doesn't separate the data
            if not np.any(mask_left) or not np.any(mask_right):
                gains.append(-float('inf'))
                continue

            #Split the data into two subsets, according to the optimal split
            y1 = y[mask_left]
            y2 = y[mask_right]
            
            #Calculate the entropy of the two subsets
            H1 = self.entropy(y1)
            H2 = self.entropy(y2)
            
            # Weighted average of child entropies
            H_child = (len(y1) * H1 + len(y2) * H2) / len(y)
            I_gain = H_parent - H_child
            gains.append(I_gain)

        # If no valid split found, return most common label
        if max(gains) == -float('inf'):
            values, counts = np.unique(y, return_counts=True)
            return values[np.argmax(counts)]

        #If the information gain is less than 0.05, return the most common label in interest of pruning  
        gain = np.max(gains)
        if gain <= self.min_info_gain:
            values, counts = np.unique(y, return_counts=True)
            return values[np.argmax(counts)]

        #Now split the data on the selected feature and create two new decision trees
        best_attr = np.argmax(gains)
        if self.method == "median":
                m = np.median(x[:,best_attr])
        elif self.method == "mean":
                m = np.mean(x[:,best_attr])
        else: m = self.find_optimal_split(x[:,best_attr], y)

        mask_left = x[:, best_attr] < m
        mask_right = x[:, best_attr] >= m
        
        x1 = x[mask_left]
        y1 = y[mask_left]
        x2 = x[mask_right]
        y2 = y[mask_right]

        self.left = DecisionTreeClassifier(max_depth=self.max_depth, min_info_gain=self.min_info_gain, method=self.method)
        self.right = DecisionTreeClassifier(max_depth=self.max_depth, min_info_gain=self.min_info_gain, method=self.method)
        
        #Recursively call fit on the two new decision trees
        self.left = self.left.fit(x1, y1, current_depth + 1)
        self.right = self.right.fit(x2, y2, current_depth + 1)

        #Set the attributes of the current node 
        self.best_attr = best_attr
        self.split_value = m
        self.depth = current_depth 
        self.is_trained = True
        
        return self


    def print_tree(self, depth=0):
        """Prints the decision tree structure"""
        if isinstance(self, str):  # If this is a leaf node (class label)
            print("\t" * depth + f"Leaf: {self}")
            return
            
        # If this is an internal node
        print("\t" * depth + f"[Attribute {self.best_attr} >= {self.split_value}]")
        
        print("\t" * depth + "Left:")
        if isinstance(self.left, str):
            print("\t" * (depth + 1) + f"Leaf: {self.left}")
        else:
            self.left.print_tree(depth + 1)
            
        print("\t" * depth + "Right:")
        if isinstance(self.right, str):
            print("\t" * (depth + 1) + f"Leaf: {self.right}")
        else:
            self.right.print_tree(depth + 1)

    def predict_one_row(self, x):
        if isinstance(self, str):
            return self
        elif x[self.best_attr] < self.split_value:
            if isinstance(self.left, str):
                return self.left
            return self.left.predict_one_row(x)
        else:
            if isinstance(self.right, str):
                return self.right
            return self.right.predict_one_row(x)

    def predict(self, x):
        """ Predicts a set of samples using the trained DecisionTreeClassifier.
        
        Assumes that the DecisionTreeClassifier has already been trained.
        
        Args:
        x (numpy.ndarray): Instances, numpy array of shape (M, K) 
                           M is the number of test instances
                           K is the number of attributes
        
        Returns:
        numpy.ndarray: A numpy array of shape (M, ) containing the predicted
                       class label for each instance in x
        """
        
        # make sure that the classifier has been trained before predicting
        if not self.is_trained:
            raise Exception("DecisionTreeClassifier has not yet been trained.")
        
        # set up an empty (M, ) numpy array to store the predicted labels 
        # feel free to change this if needed
        predictions = np.zeros((x.shape[0],), dtype=str)
        
        #######################################################################
        #                 ** TASK 2.2: COMPLETE THIS METHOD **
        #######################################################################
        for i in range(x.shape[0]):
            predictions[i] = self.predict_one_row(x[i])

        return predictions

    def find_optimal_split(self, x, y):
        """
        Finds the optimal split value for a given feature, based on information gain

        Parameters:
        x (numpy.ndarray): The feature values, shape (N, M)
        y (numpy.ndarray): The class labels, shape (N, )

        Returns:
        float: The optimal split value
        """
        x_range = np.arange(np.min(x), np.max(x), 1)
        entropies = {}

        for i in x_range:
            mask_right = x >= i
            mask_left = x < i

            # Skip if split doesn't separate the data
            if not np.any(mask_left) or not np.any(mask_right):
                continue

            y1 = y[mask_left]
            y2 = y[mask_right]
            h1 = self.entropy(y1)
            h2 = self.entropy(y2)

            h = (len(y1) * h1 + len(y2) * h2) / len(y)
            entropies[i] = h

        #If there are no valid splits, return the median of the feature
        if not entropies:
            return np.median(x)

        split_value = min(entropies, key=entropies.get)
        return split_value

    def confusion_matrix(self, y_gold, y_prediction, class_labels=None):

        """ Compute the confusion matrix.
        Args:
            y_gold (np.ndarray): the correct ground truth/gold standard labels
            y_prediction (np.ndarray): the predicted labels
            class_labels (np.ndarray): a list of unique class labels. 
                                Defaults to the union of y_gold and y_prediction.

        Returns:
            np.array : shape (C, C), where C is the number of classes. 
                    Rows are ground truth per class, columns are predictions
        """

        # if no class_labels are given, we obtain the set of unique class labels from
        # the union of the ground truth annotation and the prediction
        if class_labels is None:
            class_labels = np.unique(np.concatenate((y_gold, y_prediction)))

        confusion = np.zeros((len(class_labels), len(class_labels)), dtype=int)

        # TODO: Complete this
        # for each correct class (row), 
        # compute how many instances are predicted for each class (columns)

        for i in range(len(class_labels)):
            for j in range(len(class_labels)):
                confusion[i,j] = np.sum(np.logical_and(y_gold == class_labels[i], y_prediction == class_labels[j]))

        return confusion


    def accuracy(self, y_gold, y_prediction):
        """
        Compute the accuracy given the ground truth and predictions
        
        Args:
            y_gold (np.ndarray): the correct ground truth/gold standard labels
            y_prediction (np.ndarray): the predicted labels
        
        Returns:
            float : The accuracy score
        """
        # Get the confusion matrix
        conf_matrix = self.confusion_matrix(y_gold, y_prediction)
        # Accuracy is sum of diagonal divided by total sum
        return np.sum(np.diag(conf_matrix)) / np.sum(conf_matrix)

    def precision(self, y_gold, y_prediction, class_label):
        """
        Compute the precision for a given class
        
        Args:
            y_gold (np.ndarray): the correct ground truth/gold standard labels
            y_prediction (np.ndarray): the predicted labels
            class_label: the class label for which to compute precision
            
        Returns:
            float : The precision score for the given class
        """
        conf_matrix = self.confusion_matrix(y_gold, y_prediction)
        # Get index of class_label
        class_idx = np.where(np.unique(np.concatenate((y_gold, y_prediction))) == class_label)[0][0]
        # Precision is true positives divided by all predicted positives
        true_positives = conf_matrix[class_idx, class_idx]
        predicted_positives = np.sum(conf_matrix[:, class_idx])
        return true_positives / predicted_positives if predicted_positives > 0 else 0

    def recall(self, y_gold, y_prediction, class_label):
        """
        Compute the recall for a given class
        
        Args:
            y_gold (np.ndarray): the correct ground truth/gold standard labels
            y_prediction (np.ndarray): the predicted labels
            class_label: the class label for which to compute recall
            
        Returns:
            float : The recall score for the given class
        """
        conf_matrix = self.confusion_matrix(y_gold, y_prediction)
        # Get index of class_label
        class_idx = np.where(np.unique(np.concatenate((y_gold, y_prediction))) == class_label)[0][0]
        # Recall is true positives divided by all actual positives
        true_positives = conf_matrix[class_idx, class_idx]
        actual_positives = np.sum(conf_matrix[class_idx, :])
        return true_positives / actual_positives if actual_positives > 0 else 0

    def f1_score(self, y_gold, y_prediction, class_label):
        """
        Compute the F1 score for a given class
        
        Args:
            y_gold (np.ndarray): the correct ground truth/gold standard labels
            y_prediction (np.ndarray): the predicted labels
            class_label: the class label for which to compute F1 score
            
        Returns:
            float : The F1 score for the given class
        """
        prec = self.precision(y_gold, y_prediction, class_label)
        rec = self.recall(y_gold, y_prediction, class_label)
        return 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0

    def macro_calculator(self, metric, y_gold, y_prediction):
        classes = np.unique(np.concatenate((y_gold, y_prediction)))
        total = 0
        for i in range(len(classes)):
            total += metric(y_gold, y_prediction, classes[i])
        return total / len(classes)

    def cross_validation(self, x, y, k=10, max_depth = None, min_info_gain = 0, method = "information_gain"):
        """
        Perform k-fold cross-validation
        
        Args:
            x (numpy.ndarray): Input features
            y (numpy.ndarray): Labels
            k (int): Number of folds
            max_depth (int): The maximum depth of the decision tree. Defaults to None
        
        Returns:
            float: Average accuracy across all folds
        """
        # Shuffle the data
        indices = np.random.permutation(len(x))
        x = x[indices]
        y = y[indices]
        
        # Calculate fold size
        fold_size = len(x) // k
        accuracies = []
        
        for i in range(k):
            # Create validation fold
            start_idx = i * fold_size
            end_idx = start_idx + fold_size if i < k-1 else len(x)
            
            # Split data into training and validation
            x_val = x[start_idx:end_idx]
            y_val = y[start_idx:end_idx]
            x_train = np.concatenate([x[:start_idx], x[end_idx:]])
            y_train = np.concatenate([y[:start_idx], y[end_idx:]])
            
            # Train model
            classifier = DecisionTreeClassifier(max_depth=max_depth, min_info_gain=min_info_gain, method=method)
            classifier.fit(x_train, y_train)
            
            # Make predictions and calculate accuracy
            y_pred = classifier.predict(x_val)
            acc = classifier.accuracy(y_val, y_pred)
            accuracies.append(acc)
            
            print(f"Fold {i+1} Accuracy: {acc}")
        
        mean_accuracy = round(np.mean(accuracies),2)
        std_accuracy = round(np.std(accuracies),2)
        print(f"\nMean Accuracy: {mean_accuracy} ± {std_accuracy}")
        
        return mean_accuracy

        



#WE ARE NO LONGER IN THE DECISION TREE CLASS **************************************************************************





        
