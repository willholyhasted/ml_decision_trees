#############################################################################
# This is a improved version of the decision tree classifier
# The main difference is that it allows one to specify for n branches for each node
##############################################################################

import numpy as np
from itertools import combinations


class DecisionTreeClassifier(object):
    """ Basic decision tree classifier
    
    Attributes:
    is_trained (bool): Keeps track of whether the classifier has been trained
    
    Methods:
    fit(x, y): Constructs a decision tree from data X and label y
    predict(x): Predicts the class label of samples X
    prune(x_val, y_val): Post-prunes the decision tree
    """

    def __init__(self, max_depth=None, min_info_gain=0,max_branches=0):
        self.is_trained = False
        self.depth = 0
        self.max_depth = max_depth
        self.min_info_gain = min_info_gain
        self.max_branches = max_branches

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


    def fit(self, x, y):
        """ Constructs a decision tree classifier from data
        
        Args:
        x (numpy.ndarray): Instances, numpy array of shape (N, K) 
                           N is the number of instances
                           K is the number of attributes
        y (numpy.ndarray): Class labels, numpy array of shape (N, )
                           Each element in y is a str 
        """

        assert x.shape[0] == len(y), \
            "Training failed. x and y must have the same number of instances."
        
        self.fit_recursive(x, y)
        self.is_trained = True
        return self

    
    def fit_recursive(self, x, y, current_depth=0):
        """ Constructs a decision tree classifier from data
        Args:
            x (numpy.ndarray): Instances, numpy array of shape (N, K)
                            N is the number of instances
                            K is the number of attributes
            y (numpy.ndarray): Class labels, numpy array of shape (N, )
                            Each element in y is a str
            current_depth(int): The current depth of the tree. Should use 0 for the root node
        """
        if np.unique(y).size == 1:
            return y[0]
        
        if self.max_depth is not None and current_depth >= self.max_depth:
            values, counts = np.unique(y, return_counts=True)
            return values[np.argmax(counts)]

        entropy_parent = self.entropy(y)
        best_gain = 0
        best_attribute = None
        best_branches = None
        best_splits = None
        best_y_child = None  # Store the best y_child splits

        for i in range(x.shape[1]):
            unique_values = np.unique(x[:, i])
            if self.max_branches == 0:
                range_branches = len(unique_values)
            else:
                range_branches = min(len(unique_values), self.max_branches)
                
            if len(unique_values) <= 0:  # Skip features with a single unique value
                continue
            if range_branches <= -1:
                continue  # Skip if there aren't enough values for splitting
            for j in range(range_branches):
                if j > 0:
                    splits = np.array(list(combinations(unique_values, j)))
                    for k in range(splits.shape[0]):
                        # Create masks for each split
                        y_splits = []
                        for split_idx in range(len(splits[k])):
                            if split_idx == 0:
                                mask = x[:, i] < splits[k][split_idx]
                            else:
                                mask = (x[:, i] >= splits[k][split_idx-1]) & (x[:, i] < splits[k][split_idx])
                            y_splits.append(y[mask])
                        # Add the final split (greater than or equal to the last split value)
                        mask = x[:, i] >= splits[k][-1]
                        y_splits.append(y[mask])
                        y_child = np.array(y_splits, dtype=object)
                        entropy_child = np.array([self.entropy(part) for part in y_child]).flatten()
                        weights = np.array([len(part) for part in y_child]).flatten()
                        
                        entropy_child_weighted = np.average(entropy_child, weights=weights)
                        gain = entropy_parent - entropy_child_weighted
                        if gain > best_gain:
                            best_gain = gain
                            best_attribute = i
                            best_branches = j+1
                            best_splits = splits[k]
                            best_y_child = y_child

        if best_gain <= self.min_info_gain or best_attribute is None:
            values, counts = np.unique(y, return_counts=True)
            return values[np.argmax(counts)]

        self.children = []
        # Create x_child splits using the best attribute and splits
        x_child = []
        for split_idx in range(len(best_splits)):
            if split_idx == 0:
                mask = x[:, best_attribute] < best_splits[split_idx]
            else:
                mask = (x[:, best_attribute] >= best_splits[split_idx-1]) & (x[:, best_attribute] < best_splits[split_idx])
            x_child.append(x[mask])
        # Add the final split
        mask = x[:, best_attribute] >= best_splits[-1]
        x_child.append(x[mask])
        
        x_child = np.array(x_child, dtype=object)
        for i in range(best_branches):
            if len(best_y_child[i]) == 0:
                values, counts = np.unique(y, return_counts=True)
                self.children.append(values[np.argmax(counts)])
            else:
                child = DecisionTreeClassifier(max_depth=self.max_depth, 
                                            min_info_gain=self.min_info_gain,
                                            max_branches=self.max_branches)
                self.children.append(child.fit_recursive(x_child[i], best_y_child[i], current_depth + 1))

        self.best_attr = best_attribute
        self.best_branches = best_branches
        self.split_values = best_splits
        self.depth = current_depth
        return self

    def predict_one_row(self, x):
        # If self is a string (leaf node), return the class label
        if isinstance(self, str):
            return self
            
        # If this node is a leaf node (no splits), return the most common class
        if not hasattr(self, "split_values") or self.split_values is None:
            return self
            
        # Check which branch x belongs to based on split values
        for i, split_value in enumerate(self.split_values):
            if x[self.best_attr] < split_value:
                return (self.children[i] if isinstance(self.children[i], str) 
                    else self.children[i].predict_one_row(x))
        
        # If no splits matched, use the last branch (x >= last split value)
        return (self.children[-1] if isinstance(self.children[-1], str)
                else self.children[-1].predict_one_row(x))

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

    def cross_validation(self, x, y, k=10, max_depth = None, min_info_gain = 0, max_branches=0):
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
            classifier = DecisionTreeClassifier(max_depth=max_depth, min_info_gain=min_info_gain, max_branches=max_branches)
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