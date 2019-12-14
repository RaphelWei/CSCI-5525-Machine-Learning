import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier

# reading data, building training and test sets
data = pd.read_csv("./Mushroom.csv", header = None).values
X_train = data[0:6000,1:] # using the first 6000 examples as training set
y_train = data[0:6000,0].reshape(-1,1)
X_test = data[6000:,1:] # the remaining examples as test set
y_test = data[6000:,0].reshape(-1,1)

class RandomForest():
    """
    This is our implementation of random forest class.
    """
    def __init__(self, x, y, n_trees, n_features):
        """
        Parameters:
        ----------
        x : training set

        y : labels of training set

        n_trees : number of trees in the forest

        n_features : number of features in the feature set

        feature_idx : a list, each element in the list is the index of feature set
                      of each decison tree
                      e.g. if first decison tree use [0,1,2,3,4] column of sample set
                      as feature set, then feature_idx[0] = [0,1,2,3,4]

        trees : a list, each element in the list is an already trained decision tree
        """  
        # initialize the parameters
        self.x, self.y, self.n_trees, self.n_features  = x, y, n_trees, n_features
        self.feature_idx = []
        self.trees = [self.create_tree() for i in range(self.n_trees)]
    def create_tree(self):
        """
        this is a function to build a desion tree
        """
        # get the index of sample set for each tree
        idx_sample_set = np.random.choice(6000, 6000) # sample datasets for each decision tree classifier 
                                                      # from training set, with replacement.
        # get the index of feature set for each tree
        idx_feature_set = np.random.choice(22, self.n_features, replace=False) # subsample random k features from d features, without replacement
        # save feature index
        self.feature_idx.append(idx_feature_set)
        # train a decision tree
        dt = DecisionTreeClassifier(criterion='gini',splitter='random', max_depth=2) # training a decision tree classifier, using gini index as criterion and max depth of a tree is 2
        dt.fit(self.x[idx_sample_set, :][:,idx_feature_set], self.y[idx_sample_set]) # building our models
        return dt # return the trained tree
        
    def predict(self, x):
        """
        this is afunction to predict labels of given data set
        """
        y_pred = [] # a list to save results
        for i in range(self.n_trees):
            dt = self.trees[i]
            # here we need to subsample the data set, 
            # so that each tree will only split on given features
            y = dt.predict(x[:,self.feature_idx[i]]).reshape(1,-1)
            y_pred.append(y)
        y_pred = np.array(y_pred)
        # calculate the mean of all trees' predicts
        y_pred = np.mean(y_pred, axis=0)
        # the final result of random forest is the sign of predicts
        # note that I do not deal with 0's in the predicts, since I think 
        # half trees claasify them as label "1" and half trees "-1", so 
        # their labels could not be determined.
        y_pred = np.sign(y_pred[0]) 
        return np.array(y_pred).reshape(-1,1)

def vary_features():
    features = [5,10,15,20] # number of features
    # train random forests given different number of features
    forests=[RandomForest(X_train, y_train, 100, t) for t in features]
    # predict the training set
    y=[f.predict(X_train) for f in forests]
    # predict for the test set
    y2 = [f.predict(X_test) for f in forests]
    # calculate accuracy for training set
    acc = [sum(i==y_train)/len(y_train) for i in y]
    # calculate accuracy for test set
    acc2 = [sum(i==y_test)/len(y_test) for i in y2]
    # for plot purpose
    plt.figure(1)
    plt.plot(features, acc)
    plt.plot(features, acc2)
    plt.xlabel("Size of Feature Set", fontsize = 16)
    plt.ylabel("Accuracy", fontsize = 16)
    plt.legend(["train", "test"])
    plt.show()

vary_features()

def vary_estimators():
    estimators = [10,20,40,80,100] # number of trees
    # train random forests given different number of trees
    forests=[RandomForest(X_train, y_train, tree, 20) for tree in estimators]
    # predict the training set
    y=[f.predict(X_train) for f in forests]
    # predict for the test set
    y2 = [f.predict(X_test) for f in forests]
    # calculate accuracy for training set
    acc = [sum(i==y_train)/len(y_train) for i in y]
    # calculate accuracy for test set
    acc2 = [sum(i==y_test)/len(y_test) for i in y2]
    # for plot purpose
    plt.figure(2)
    plt.plot(estimators, acc)
    plt.plot(estimators, acc2)
    plt.xlabel("Number of Estimators", fontsize = 16)
    plt.ylabel("Accuracy", fontsize = 16)
    plt.legend(["train", "test"])
    plt.show()

vary_estimators()