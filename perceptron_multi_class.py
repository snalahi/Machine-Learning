#!/usr/bin/env python
# coding: utf-8

# In[50]:


import numpy as np
import sklearn.datasets as skdata
from sklearn.linear_model import Perceptron


'''
Name: Doe, John (Please write names in <Last Name, First Name> format)

Collaborators: Doe, Jane (Please write names in <Last Name, First Name> format)

Collaboration details: Discussed <function name> implementation details with Jane Doe.

Summary:

You should answer the questions:
1) What did you do in this assignment?
2) How did you do it?
3) What are the constants and hyper-parameters you used?

Scores:

Report your scores here. For example,

Results on the iris dataset using scikit-learn Perceptron model
Training set mean accuracy: 0.8512
Validation set mean accuracy: 0.7333
Testing set mean accuracy: 0.9286
Results on the iris dataset using our Perceptron model trained with 0 steps and tolerance of 0.0
Training set mean accuracy: 0.0000
Validation set mean accuracy: 0.0000
Results on the iris dataset using our Perceptron model trained with 0 steps and tolerance of 0.0
Training set mean accuracy: 0.0000
Validation set mean accuracy: 0.0000
Results on the iris dataset using our Perceptron model trained with 0 steps and tolerance of 0.0
Training set mean accuracy: 0.0000
Validation set mean accuracy: 0.0000
Using best model trained with 0 steps and tolerance of 0.0
Testing set mean accuracy: 0.0000
Results on the wine dataset using scikit-learn Perceptron model
Training set mean accuracy: 0.5625
Validation set mean accuracy: 0.4118
Testing set mean accuracy: 0.4706
Results on the wine dataset using our Perceptron model trained with 0 steps and tolerance of 0.0
Training set mean accuracy: 0.0000
Validation set mean accuracy: 0.0000
Results on the wine dataset using our Perceptron model trained with 0 steps and tolerance of 0.0
Training set mean accuracy: 0.0000
Validation set mean accuracy: 0.0000
Results on the wine dataset using our Perceptron model trained with 0 steps and tolerance of 0.0
Training set mean accuracy: 0.0000
Validation set mean accuracy: 0.0000
Using best model trained with 0 steps and tolerance of 0.0
Testing set mean accuracy: 0.0000
'''

'''
Implementation of Perceptron for multi-class classification
'''
class PerceptronMultiClass(object):

    def __init__(self):
        # Define private variables, weights and number of classes
        self.__weights = None
        self.__n_class = -1

    def __update(self, x, y):
        '''
        Update the weight vector during each training iteration

        Args:
            x : numpy
                d x N feature vector
            y : numpy
                1 x N ground-truth label
        '''
        # TODO: Implement the member update function
        #prediction_array = np.array([])
        threshold = 0.5 * np.ones([1, x.shape[1]])
        x = np.concatenate([threshold, x], axis=0)
        
        for n in range(x.shape[1]):
            x_n = np.expand_dims(x[:, n], axis=-1)
            for c in range (self.__n_class):
                weights_c = np.expand_dims(self.__weights[:, c], axis=-1)
                prediction_arg = np.matmul(weights_c.T, x_n)
                #prediction_array = np.append(prediction_array, prediction_arg)
                #prediction = np.argmax(prediction_arg)
            prediction = np.argmax(prediction_arg)
            if prediction != y[n]:
                self.__weights_c_hat = np.expand_dims(self.__weights[:, prediction], axis=-1)
                self.__weights_c_star = np.expand_dims(self.__weights[:, y[n]], axis=-1)
                self.__weights_c_hat = self.__weights_c_hat - x_n
                self.__weights_c_star = self.__weights_c_star + x_n
                    
    def fit(self, x, y, T=100, tol=1e-3):
        '''
        Fits the model to x and y by updating the weight vector
        based on mis-classified examples for t iterations until convergence

        Args:
            x : numpy
                d x N feature vector
            y : numpy
                1 x N ground-truth label
            t : int
                number of iterations to optimize perceptron
            tol : float
                change of loss tolerance, if greater than loss + tolerance, then stop
        '''
        # TODO: Implement the fit function
        self.__n_class = len(np.unique(y))
        self.__weights = np.zeros([x.shape[0] + 1, self.__n_class])

        self.__weights[0,0] = -1.0
        prev_loss = 2.0
        prev_weights = np.copy(self.__weights)
        for t in range(int(T)):
            predictions = self.predict(x)
            loss = np.mean(np.where(predictions != y, 1.0, 0.0))
            if loss == 0.0:
                break
            elif loss > prev_loss + tol and t > 2:
                self.__weights = prev_weights
                break
            prev_loss = loss
            prev_weights = np.copy(self.__weights)
            self.__update(x, y)

    def predict(self, x):
        '''
        Predicts the label for each feature vector x

        Args:
            x : numpy
                d x N feature vector

        Returns:
            numpy : 1 x N label vector
        '''
        # TODO: Implement the predict function
        #predictions_array = np.array([])
        threshold = 0.5 * np.ones([1, x.shape[1]])
        x = np.concatenate([threshold, x], axis=0)
        for c in range(self.__n_class):
            weights_c = np.expand_dims(self.__weights[:, c], axis=-1)
            predictions_arg = np.matmul(weights_c.T, x)
            #predictions_array = np.append(predictions_array, predictions_arg)
        return np.argmax(predictions_arg)

    def score(self, x, y):
        '''
        Predicts labels based on feature vector x and computes the mean accuracy
        of the predictions

        Args:
            x : numpy
                d x N feature vector
            y : numpy
                1 x N ground-truth label

        Returns:
            float : mean accuracy
        '''
        # TODO: Implement the score function
        predictions = self.predict(x)
        scores = np.where(predictions == y, 1.0, 0.0)
        return np.mean(scores)

def split_dataset(x, y, n_sample_train_to_val_test=8):
    '''
    Helper function to splits dataset into training, validation and testing sets

    Args:
        x : numpy
            d x N feature vector
        y : numpy
            1 x N ground-truth label
        n_sample_train_to_val_test : int
            number of training samples for every validation, testing sample

    Returns:
        x_train : numpy
            d x n feature vector
        y_train : numpy
            1 x n ground-truth label
        x_val : numpy
            d x m feature vector
        y_val : numpy
            1 x m ground-truth label
        x_test : numpy
            d x m feature vector
        y_test : numpy
            1 x m ground-truth label
    '''
    n_sample_interval = n_sample_train_to_val_test + 2

    train_idx = []
    val_idx = []
    test_idx = []
    for idx in range(x.shape[0]):
        if idx and idx % n_sample_interval == (n_sample_interval - 1):
            val_idx.append(idx)
        elif idx and idx % n_sample_interval == 0:
            test_idx.append(idx)
        else:
            train_idx.append(idx)

    x_train, x_val, x_test = x[train_idx, :], x[val_idx, :], x[test_idx, :]
    y_train, y_val, y_test = y[train_idx], y[val_idx], y[test_idx]

    return x_train, y_train, x_val, y_val, x_test, y_test


# In[51]:


if __name__ == '__main__':

    iris_data = skdata.load_iris()
    wine_data = skdata.load_wine()

    datasets = [iris_data, wine_data]
    tags = ['iris', 'wine']

    # TODO: Experiment with 3 different max training steps (T) for each dataset
    train_steps_iris = [0, 0, 0]
    train_steps_wine = [0, 0, 0]

    train_steps = [train_steps_iris, train_steps_wine]

    # TODO: Set a tolerance for each dataset
    tol_iris = 0.0
    tol_wine = 0.0

    tols = [tol_iris, tol_wine]

    for dataset, steps, tol, tag in zip(datasets, train_steps, tols, tags):
        # Split dataset into 80 training, 10 validation, 10 testing
        x = dataset.data
        y = dataset.target
        x_train, y_train, x_val, y_val, x_test, y_test = split_dataset(
            x=x,
            y=y,
            n_sample_train_to_val_test=8)

        '''
        Trains and tests Perceptron model from scikit-learn
        '''
        model = Perceptron(penalty=None, alpha=0.0, tol=1e-3)
        # Trains scikit-learn Perceptron model
        model.fit(x_train, y_train)

        print('Results on the {} dataset using scikit-learn Perceptron model'.format(tag))

        # Test model on training set
        scores_train = model.score(x_train, y_train)
        print('Training set mean accuracy: {:.4f}'.format(scores_train))

        # Test model on validation set
        scores_val = model.score(x_val, y_val)
        print('Validation set mean accuracy: {:.4f}'.format(scores_val))

        # Test model on testing set
        scores_test = model.score(x_test, y_test)
        print('Testing set mean accuracy: {:.4f}'.format(scores_test))

        '''
        Trains, validates, and tests our Perceptron model for multi-class classification
        '''
        # TODO: obtain dataset in correct shape (d x N)
        x_train = np.transpose(x_train, axes=(1, 0))
        x_val = np.transpose(x_val, axes=(1, 0))
        x_test = np.transpose(x_test, axes=(1, 0))
        
        # Initialize empty lists to hold models and scores
        models = []
        scores = []
        steps = [10, 20, 60]
        tol = 1e-3
        
        for T in steps:
            # TODO: Initialize PerceptronMultiClass model
            model = PerceptronMultiClass()

            print('Results on the {} dataset using our Perceptron model trained with {} steps and tolerance of {}'.format(tag, T, tol))
            # TODO: Train model on training set
            model.fit(x_train, y_train, tol)
            model.predict(x_train)

            # TODO: Test model on training set
            scores_train = model.score(x_train, y_train)
            print('Training set mean accuracy: {:.4f}'.format(scores_train))

            # TODO: Test model on validation set
            scores_val = model.score(x_val, y_val)
            print('Validation set mean accuracy: {:.4f}'.format(scores_val))

            # TODO: Save the model and its score
            models.append(model)
            scores.append(scores_val)

        # TODO: Select the best performing model on the validation set
        best_idx = 0
        for i in range(len(scores)):
            if scores[i] == max(scores):
                best_idx = i
                break
            else:
                continue

        print('Using best model trained with {} steps and tolerance of {}'.format(steps[best_idx], tol))

        # TODO: Test model on testing set
        scores_test = model.score(x_test, y_test)
        print('Testing set mean accuracy: {:.4f}'.format(scores_test))

