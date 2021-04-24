#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import sklearn.datasets as skdata
import sklearn.metrics as skmetrics
from sklearn.linear_model import LinearRegression


'''
Name: Alahi, Sk Nasimul (Please write names in <Last Name, First Name> format)

Collaborators: Wong, Alex (Please write names in <Last Name, First Name> format)

Collaboration details: The collaborator provided the skeleton and also all the major 
functions used in the following program.

Summary:
Report your scores here. For example,

Results using scikit-learn LinearRegression model
Training set mean squared error: 23.2560
Training set r-squared scores: 0.7323
Validation set mean squared error: 17.6111
Validation set r-squared scores: 0.7488
Testing set mean squared error: 17.1465
Testing set r-squared scores: 0.7805
Results using our linear regression model trained with normal_equation
Training set mean squared error: 25.9360
Training set r-squared scores: 0.7015
Validation set mean squared error: 18.4747
Validation set r-squared scores: 0.7365
Testing set mean squared error: 18.1262
Testing set r-squared scores: 0.7679
Results using our linear regression model trained with pseudoinverse
Training set mean squared error: 25.9360
Training set r-squared scores: 0.7015
Validation set mean squared error: 18.4747
Validation set r-squared scores: 0.7365
Testing set mean squared error: 18.1262
Testing set r-squared scores: 0.7679
'''

'''
Implementation of linear regression by directly solving normal equation or pseudoinverse
'''
class LinearRegressionClosedForm(object):

    def __init__(self):
        # Define private variables
        self.__weights = None

    def __fit_normal_equation(self, X, y):
        '''
        Fits the model to x and y via normal equation

        Args:
            X : numpy
                N x d feature vector
            y : numpy
                1 x N ground-truth label
        '''
        # TODO: Implement the __fit_normal_equation function

        X_trans = np.matmul(X.T, X)
        X_trans_X_inv = np.linalg.inv(X_trans)
        self.__weights = np.matmul(np.matmul(X_trans_X_inv, X.T), y)

    def __fit_pseudoinverse(self, X, y):
        '''
        Fits the model to x and y via pseudoinverse

        Args:
            X : numpy
                N x d feature vector
            y : numpy
                1 x N ground-truth label
        '''
        # TODO: Implement the __fit_pseudoinverse function

        U, S, V_t = np.linalg.svd(X)
        S_diag = np.diag(1.0 / S);
        
        padding = np.zeros([U.shape[0] - S.shape[0], S.shape[0]])
        S_pseudo = np.concatenate([S_diag, padding], axis=0)
        
        S_pseudo = S_pseudo.T
        X_pseudo = np.matmul(np.matmul(V_t.T, S_pseudo), U.T)
        
        self.__weights = np.matmul(X_pseudo, y)
        

    def fit(self, x, y, solver=''):
        '''
        Fits the model to x and y by solving the ordinary least squares
        using normal equation or pseudoinverse (SVD)

        Args:
            x : numpy
                d x N feature vector
            y : numpy
                1 x N ground-truth label
            solver : str
                solver types: normal_equation, pseudoinverse
        '''
        # TODO: Implement the fit function

        X = x.T
        
        if solver == 'normal_equation':
            self.__fit_normal_equation(X, y)
        elif solver == 'pseudoinverse':
            self.__fit_pseudoinverse(X, y)
        else:
            raise ValueError('Encountered unsupported solver: {}'.format(solver))
                
    def predict(self, x):
        '''
        Predicts the label for each feature vector x

        Args:
            x : numpy
                d x N feature vector

        Returns:
            numpy : d x 1 label vector
        '''
        # TODO: Implement the predict function
        self.__weights = np.expand_dims(self.__weights, axis=-1)

        predictions = np.matmul(self.__weights.T, x)
        return predictions

    def __score_r_squared(self, y_hat, y):
        '''
        Measures the r-squared score from groundtruth y

        Args:
            y_hat : numpy
                1 x N predictions
            y : numpy
                1 x N ground-truth label

        Returns:
            float : r-squared score
        '''
        # TODO: Implement the __score_r_squared function
        
        sum_squared_errors = np.sum((y_hat - y) ** 2)
        sum_variance = np.sum((y - np.mean(y)) ** 2)

        return 1.0 - (sum_squared_errors / sum_variance)

    def __score_mean_squared_error(self, y_hat, y):
        '''
        Measures the mean squared error (distance) from groundtruth y

        Args:
            y_hat : numpy
                1 x N predictions
            y : numpy
                1 x N ground-truth label

        Returns:
            float : mean squared error (mse)
        '''
        # TODO: Implement the __score_mean_squared_error function

        return np.mean((y_hat - y) ** 2)

    def score(self, x, y, scoring_func=''):
        '''
        Predicts real values from x and measures the mean squared error (distance)
        or r-squared from groundtruth y

        Args:
            x : numpy
                d x N feature vector
            y : numpy
                1 x N ground-truth label
            scoring_func : str
                scoring function: r_squared, mean_squared_error

        Returns:
            float : mean squared error (mse)
        '''
        # TODO: Implement the score function

        x = np.squeeze(x)
        
        if scoring_func == 'r_squared':
            return self.__score_r_squared(x, y)
        elif scoring_func == 'mean_squared_error':
            return self.__score_mean_squared_error(x, y)
        else:
            raise ValueError('Encountered unsupported scoring_func: {}'.format(scoring_func))


# In[2]:


if __name__ == '__main__':

    boston_housing_data = skdata.load_boston()
    x = boston_housing_data.data
    y = boston_housing_data.target

    # 80 percent train, 10 percent validation, 10 percent test split
    train_idx = []
    val_idx = []
    test_idx = []
    for idx in range(x.shape[0]):
        if idx and idx % 10 == 9:
            val_idx.append(idx)
        elif idx and idx % 10 == 0:
            test_idx.append(idx)
        else:
            train_idx.append(idx)

    x_train, x_val, x_test = x[train_idx, :], x[val_idx, :], x[test_idx, :]
    y_train, y_val, y_test = y[train_idx], y[val_idx], y[test_idx]

    '''
    Trains and tests linear regression model from scikit-learn
    '''
    # TODO: Initialize scikit-learn linear regression model
    model = LinearRegression()

    # TODO: Trains scikit-learn linear regression model
    model.fit(x_train, y_train)

    print('Results using scikit-learn LinearRegression model')

    # TODO: Test model on training set
    
    predictions_train = model.predict(x_train)
    
    scores_mse_train = skmetrics.mean_squared_error(predictions_train, y_train)
    print('Training set mean squared error: {:.4f}'.format(scores_mse_train))

    scores_r2_train = model.score(x_train, y_train)
    print('Training set r-squared scores: {:.4f}'.format(scores_r2_train))

    # TODO: Test model on validation set
    predictions_val = model.predict(x_val)

    scores_mse_val = skmetrics.mean_squared_error(predictions_val, y_val)
    print('Validation set mean squared error: {:.4f}'.format(scores_mse_val))

    scores_r2_val = model.score(x_val, y_val)
    print('Validation set r-squared scores: {:.4f}'.format(scores_r2_val))

    # TODO: Test model on testing set
    predictions_test = model.predict(x_test)

    scores_mse_test = skmetrics.mean_squared_error(predictions_test, y_test)
    print('Testing set mean squared error: {:.4f}'.format(scores_mse_test))

    scores_r2_test = model.score(x_test, y_test)
    print('Testing set r-squared scores: {:.4f}'.format(scores_r2_test))

    '''
    Trains and tests our linear regression model using different solvers
    '''
    # TODO: obtain dataset in correct shape (d x N) previously (N x d)
    x_train = np.transpose(x_train, axes=(1, 0))
    x_val = np.transpose(x_val, axes=(1, 0))
    x_test = np.transpose(x_test, axes=(1, 0))

    #Train 2 LinearRegressionClosedForm models using normal equation and pseudoinverse
    solvers = ['normal_equation', 'pseudoinverse']
    for solver in solvers:
        #TODO: Initialize linear regression model
        model = LinearRegressionClosedForm()

        print('Results using our linear regression model trained with {}'.format(solver))
        #TODO: Train model on training set
        model.fit(x_train, y_train, solver)

        #TODO: Test model on training set using mean squared error and r-squared
        predictions_train = model.predict(x_train)

        scores_mse_train = model.score(predictions_train, y_train, scoring_func='mean_squared_error')
        print('Training set mean squared error: {:.4f}'.format(scores_mse_train))

        scores_r2_train = model.score(predictions_train, y_train, scoring_func='r_squared')
        print('Training set r-squared scores: {:.4f}'.format(scores_r2_train))

        #TODO: Test model on validation set using mean squared error and r-squared
        predictions_val = model.predict(x_val)
        scores_mse_val = model.score(predictions_val, y_val, scoring_func='mean_squared_error')
        print('Validation set mean squared error: {:.4f}'.format(scores_mse_val))

        scores_r2_val = model.score(predictions_val, y_val, scoring_func='r_squared')
        print('Validation set r-squared scores: {:.4f}'.format(scores_r2_val))

        #TODO: Test model on testing set using mean squared error and r-squared
        predictions_test = model.predict(x_test)
        scores_mse_test = model.score(predictions_test, y_test, scoring_func='mean_squared_error')
        print('Testing set mean squared error: {:.4f}'.format(scores_mse_test))

        scores_r2_test = model.score(predictions_test, y_test, scoring_func='r_squared')
        print('Testing set r-squared scores: {:.4f}'.format(scores_r2_test))

