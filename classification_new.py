#############################################################################
# This is a improved version of the decision tree classifier
# The main difference is that it allows one to specify for n branches for each node
##############################################################################

import numpy as np
import itertools


class DecisionTreeClassifier(object):


    def __init__(self, max_depth=None, n_branches=2):
        self.is_trained = False
        self.max_depth = max_depth
        self.n_branches = n_branches  # Number of branches at each split
        self.children = []  # List to store child nodes
        

    def information(self, y):
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
        if np.unique(y).size == 1:
            return y[0]
            
        if self.max_depth is not None and current_depth >= self.max_depth:
            values, counts = np.unique(y, return_counts=True)
            return values[np.argmax(counts)]
            
        gains = []
        splits_per_feature = []
        
        for i in range(x.shape[1]):
            splits = self.find_optimal_split(x[:, i], y)
            splits_per_feature.append(splits)
            
            # Calculate information gain for these splits
            H_parent = self.information(y)
            H_children = 0
            total_samples = len(y)
            
            # Create regions based on splits
            splits = sorted(splits)
            regions_y = []
            
            # First region
            mask = x[:, i] < splits[0]
            if np.any(mask):
                regions_y.append(y[mask])
                
            # Middle regions
            for j in range(1, len(splits)):
                mask = (x[:, i] >= splits[j-1]) & (x[:, i] < splits[j])
                if np.any(mask):
                    regions_y.append(y[mask])
                    
            # Last region
            mask = x[:, i] >= splits[-1]
            if np.any(mask):
                regions_y.append(y[mask])
                
            # Calculate weighted entropy
            for region_y in regions_y:
                weight = len(region_y) / total_samples
                H_children += weight * self.information(region_y)
                
            gains.append(H_parent - H_children)
            
        best_feature = np.argmax(gains)
        best_splits = splits_per_feature[best_feature]
        
        self.feature = best_feature
        self.splits = best_splits
        self.children = []
        
        # Create child nodes for each region
        splits = sorted(best_splits)
        regions = []
        
        # First region
        mask = x[:, best_feature] < splits[0]
        if np.any(mask):
            regions.append((x[mask], y[mask]))
            
        # Middle regions
        for i in range(1, len(splits)):
            mask = (x[:, best_feature] >= splits[i-1]) & (x[:, best_feature] < splits[i])
            if np.any(mask):
                regions.append((x[mask], y[mask]))
                
        # Last region
        mask = x[:, best_feature] >= splits[-1]
        if np.any(mask):
            regions.append((x[mask], y[mask]))
            
        # Create and fit child nodes
        for region_x, region_y in regions:
            child = DecisionTreeClassifier(max_depth=self.max_depth, n_branches=self.n_branches)
            self.children.append(child.fit(region_x, region_y, current_depth + 1))
            
        self.is_trained = True
        return self


    def predict_one_row(self, x):
            if isinstance(self, str):
                return self
                
            # Find the appropriate child node
            splits = sorted(self.splits)
            
            if x[self.feature] < splits[0]:
                return self.children[0].predict_one_row(x) if not isinstance(self.children[0], str) else self.children[0]
                
            for i in range(1, len(splits)):
                if splits[i-1] <= x[self.feature] < splits[i]:
                    return self.children[i].predict_one_row(x) if not isinstance(self.children[i], str) else self.children[i]
                    
            return self.children[-1].predict_one_row(x) if not isinstance(self.children[-1], str) else self.children[-1]

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
        """Find optimal split points for n-way branching"""
        x_range = np.arange(np.min(x), np.max(x), 1)
        entropies = {}
        
        # For n-way splits, we need n-1 split points
        for splits in itertools.combinations(x_range, self.n_branches - 1):
            # Convert splits to tuple so it can be used as dictionary key
            splits = tuple(sorted(splits))  # Convert to tuple after sorting
            
            # Create masks for each region
            masks = []
            
            # First region: x < splits[0]
            masks.append(x < splits[0])
            
            # Middle regions: splits[i-1] <= x < splits[i]
            for i in range(1, len(splits)):
                masks.append((x >= splits[i-1]) & (x < splits[i]))
                
            # Last region: x >= splits[-1]
            masks.append(x >= splits[-1])
            
            # Calculate weighted entropy for this split
            total_entropy = 0
            total_samples = len(y)
            
            for mask in masks:
                if not np.any(mask):  # Skip if region is empty
                    continue
                subset_y = y[mask]
                weight = len(subset_y) / total_samples
                entropy = self.information(subset_y)
                total_entropy += weight * entropy
                
            entropies[splits] = total_entropy
            
        if not entropies:  # If no valid splits found
            return tuple(np.quantile(x, np.linspace(0, 1, self.n_branches + 1)[1:-1]))
            
        return min(entropies, key=entropies.get)


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

    def cross_validation(self, x, y, k=5, max_depth = None):
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
            classifier = DecisionTreeClassifier(max_depth)
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





        
