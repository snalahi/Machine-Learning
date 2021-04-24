#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import sklearn.datasets as skdata
import sklearn.metrics as skmetrics
import sklearn.preprocessing as skpreprocess
from sklearn.linear_model import Ridge as RidgeRegression
from matplotlib import pyplot as plt


'''
Name: Alahi, Sk Nasimul (Please write names in <Last Name, First Name> format)

Collaborators: Wong, Alex (Please write names in <Last Name, First Name> format)

Collaboration details: The collaborator provided the skeleton and also the plotting
functions used in the following program. Discussed with him the fit function, specifically
data fidelity and regularization losses.

Summary:

Please describe what you did for this assignment, what loss you minimized, and how you
minimized the loss function?

Ans: I implemented the RidgeRegressionClosedForm class using the closed form solution
w* = (Z(Trans)Z + lambdaI)(Inv)Z(Trans)y through the implementation of all sections
marked with TODO, namely:
1) fit function
2) predict function
3) __score_r_squared function
4) __score_mean_squared_error function
5) score function
6) train, validate, test loop in main function
7) plotting MSE and R-squared scores in main function

The losses that I minimized are data fidelity loss and regularization loss. While minimizing
the loss function, I used the Linear Regression loss function in Z space,
l(w) = 1/N((Zw - y)(Trans)(Zw - y)) for data fidelity loss and l2 norm loss function
l(w) = 1/N(lambda(w(Trans)w)) for regularization loss. Finally, the total loss was
loss = data fidelity loss + regularization loss.

Report your scores here. For example,

Results for scikit-learn RidgeRegression model with alpha=1.0
Training set mean squared error: 6.3724
Training set r-squared scores: 0.9267
Validation set mean squared error: 9.6293
Validation set r-squared scores: 0.8626
Testing set mean squared error: 19.2863
Testing set r-squared scores: 0.7531
Results for scikit-learn RidgeRegression model with alpha=10.0
Training set mean squared error: 6.9915
Training set r-squared scores: 0.9195
Validation set mean squared error: 10.5660
Validation set r-squared scores: 0.8493
Testing set mean squared error: 18.0993
Testing set r-squared scores: 0.7683
Results for scikit-learn RidgeRegression model with alpha=100.0
Training set mean squared error: 7.8843
Training set r-squared scores: 0.9093
Validation set mean squared error: 11.9197
Validation set r-squared scores: 0.8300
Testing set mean squared error: 18.5883
Testing set r-squared scores: 0.7620
Results for scikit-learn RidgeRegression model with alpha=1000.0
Training set mean squared error: 8.8610
Training set r-squared scores: 0.8980
Validation set mean squared error: 11.7491
Validation set r-squared scores: 0.8324
Testing set mean squared error: 15.2857
Testing set r-squared scores: 0.8043
Results for scikit-learn RidgeRegression model with alpha=10000.0
Training set mean squared error: 10.0741
Training set r-squared scores: 0.8841
Validation set mean squared error: 11.7167
Validation set r-squared scores: 0.8329
Testing set mean squared error: 13.5444
Testing set r-squared scores: 0.8266
Results for scikit-learn RidgeRegression model with alpha=100000.0
Training set mean squared error: 11.4729
Training set r-squared scores: 0.8680
Validation set mean squared error: 12.5270
Validation set r-squared scores: 0.8213
Testing set mean squared error: 10.8895
Testing set r-squared scores: 0.8606
Results for our RidgeRegression model with alpha=1.0
Training Loss: 2705.697
Data Fidelity Loss: 2603.550  Regularization Loss: 102.147
Training set mean squared error: 6.4127
Training set r-squared scores: 0.9262
Validation set mean squared error: 8.9723
Validation set r-squared scores: 0.8720
Testing set mean squared error: 18.4835
Testing set r-squared scores: 0.7633
Results for our RidgeRegression model with alpha=10.0
Training Loss: 3010.343
Data Fidelity Loss: 2852.474  Regularization Loss: 157.870
Training set mean squared error: 7.0258
Training set r-squared scores: 0.9191
Validation set mean squared error: 9.5386
Validation set r-squared scores: 0.8639
Testing set mean squared error: 16.1997
Testing set r-squared scores: 0.7926
Results for our RidgeRegression model with alpha=100.0
Training Loss: 3388.757
Data Fidelity Loss: 3219.627  Regularization Loss: 169.130
Training set mean squared error: 7.9301
Training set r-squared scores: 0.9087
Validation set mean squared error: 10.6471
Validation set r-squared scores: 0.8481
Testing set mean squared error: 16.3874
Testing set r-squared scores: 0.7902
Results for our RidgeRegression model with alpha=1000.0
Training Loss: 3827.986
Data Fidelity Loss: 3618.035  Regularization Loss: 209.950
Training set mean squared error: 8.9114
Training set r-squared scores: 0.8974
Validation set mean squared error: 11.2366
Validation set r-squared scores: 0.8397
Testing set mean squared error: 14.5313
Testing set r-squared scores: 0.8139
Results for our RidgeRegression model with alpha=10000.0
Training Loss: 4347.233
Data Fidelity Loss: 4077.059  Regularization Loss: 270.174
Training set mean squared error: 10.0420
Training set r-squared scores: 0.8844
Validation set mean squared error: 11.8909
Validation set r-squared scores: 0.8304
Testing set mean squared error: 13.8512
Testing set r-squared scores: 0.8226
Results for our RidgeRegression model with alpha=100000.0
Training Loss: 5271.383
Data Fidelity Loss: 4708.966  Regularization Loss: 562.418
Training set mean squared error: 11.5984
Training set r-squared scores: 0.8665
Validation set mean squared error: 13.1313
Validation set r-squared scores: 0.8127
Testing set mean squared error: 11.8234
Testing set r-squared scores: 0.8486
'''

'''
Implementation of ridge regression
'''
class RidgeRegressionClosedForm(object):

    def __init__(self):

        # Define private variables
        self.__weights = None

    def fit(self, z, y, alpha=0.0):
        '''
        Fits the model to x and y using closed form solution

        Args:
            z : numpy
                d x N feature vector
            y : numpy
                1 x N ground-truth label
            alpha : float
                weight (lambda) of regularization term
        '''

        # TODO: Implement the fit function
        z = z.T
        z_trans = np.matmul(z.T, z)
        alpha_identity = alpha * np.identity(z_trans.shape[0])
        z_trans_z_inv = np.linalg.inv(np.add(z_trans, alpha_identity))
        self.__weights = np.matmul(np.matmul(z_trans_z_inv, z.T), y)

        # TODO: Compute loss
        z_weights_y = np.subtract(np.matmul(z, self.__weights), y)
        loss_data_fidelity = np.mean(np.matmul(z_weights_y.T, z_weights_y))
        loss_regularization = np.mean(alpha * np.matmul(self.__weights.T, self.__weights))
        loss = loss_data_fidelity + loss_regularization

        print('Training Loss: {:.3f}'.format(loss))
        print('Data Fidelity Loss: {:.3f}  Regularization Loss: {:.3f}'.format(
            loss_data_fidelity, loss_regularization))

    def predict(self, z):
        '''
        Predicts the label for each feature vector x

        Args:
            z : numpy
                d x N feature vector

        Returns:
            numpy : d x 1 label vector
        '''

        # TODO: Implement the predict function
        self.__weights = np.expand_dims(self.__weights, axis=-1)
        predictions = np.matmul(self.__weights.T, z)
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

    def score(self, x, y, scoring_func='r_squared'):
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
        if scoring_func == 'r_squared':
            return self.__score_r_squared(x, y)
        elif scoring_func == 'mean_squared_error':
            return self.__score_mean_squared_error(x, y)
        else:
            raise ValueError('Encountered unsupported scoring_func: {}'.format(scoring_func))


'''
Utility functions to compute error and plot
'''
def score_mean_squared_error(model, x, y):
    '''
    Scores the model on mean squared error metric

    Args:
        model : object
            trained model, assumes predict function returns N x d predictions
        x : numpy
            N x d numpy array of features
        y : numpy
            N x 1 groundtruth vector
    Returns:
        float : mean squared error
    '''

    # Implement the score mean squared error function
    predictions = model.predict(x)
    mse = skmetrics.mean_squared_error(predictions, y)
    return mse

def plot_results(axis,
                 x_values,
                 y_values,
                 labels,
                 colors,
                 x_limits,
                 y_limits,
                 x_label,
                 y_label):
    '''
    Plots x and y values using line plot with labels and colors

    Args:
        axis :  pyplot.ax
            matplotlib subplot axis
        x_values : list[numpy]
            list of numpy array of x values
        y_values : list[numpy]
            list of numpy array of y values
        labels : str
            list of names for legend
        colors : str
            colors for each line
        x_limits : list[float]
            min and max values of x axis
        y_limits : list[float]
            min and max values of y axis
        x_label : list[float]
            name of x axis
        y_label : list[float]
            name of y axis
    '''

    # Iterate through x_values, y_values, labels, and colors and plot them
    # with associated legend
    for x, y, label, color in zip(x_values, y_values, labels, colors):
        axis.plot(x, y, marker='o', color=color, label=label)
        axis.legend(loc='best')

    # Set x and y limits
    axis.set_xlim(x_limits)
    axis.set_ylim(y_limits)

    # Set x and y labels
    axis.set_xlabel(x_label)
    axis.set_ylabel(y_label)


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
    Trains and tests Ridge regression model from scikit-learn
    '''
    # Initialize polynomial expansion of degree 2
    poly_transform = skpreprocess.PolynomialFeatures(degree=2)

    # Compute the polynomial terms needed for the data
    poly_transform.fit(x_train)

    # Transform the data by nonlinear mapping
    x_poly_train = poly_transform.transform(x_train)
    x_poly_val = poly_transform.transform(x_val)
    x_poly_test = poly_transform.transform(x_test)

    # Initialize empty lists to store scores for MSE and R-squared
    scores_mse_ridge_scikit_train = []
    scores_r2_ridge_scikit_train = []
    scores_mse_ridge_scikit_val = []
    scores_r2_ridge_scikit_val = []
    scores_mse_ridge_scikit_test = []
    scores_r2_ridge_scikit_test = []

    alphas = [1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0]

    for alpha in alphas:

        # Initialize scikit-learn ridge regression model
        model_ridge_scikit = RidgeRegression(alpha=alpha)

        # Trains scikit-learn ridge regression model
        model_ridge_scikit.fit(x_poly_train, y_train)

        print('Results for scikit-learn RidgeRegression model with alpha={}'.format(alpha))

        # Test model on training set
        score_mse_ridge_scikit_train = score_mean_squared_error(model_ridge_scikit, x_poly_train, y_train)
        print('Training set mean squared error: {:.4f}'.format(score_mse_ridge_scikit_train))

        score_r2_ridge_scikit_train = model_ridge_scikit.score(x_poly_train, y_train)
        print('Training set r-squared scores: {:.4f}'.format(score_r2_ridge_scikit_train))

        # Save MSE and R-squared training scores
        scores_mse_ridge_scikit_train.append(score_mse_ridge_scikit_train)
        scores_r2_ridge_scikit_train.append(score_r2_ridge_scikit_train)

        # Test model on validation set
        score_mse_ridge_scikit_val = score_mean_squared_error(model_ridge_scikit, x_poly_val, y_val)
        print('Validation set mean squared error: {:.4f}'.format(score_mse_ridge_scikit_val))

        score_r2_ridge_scikit_val = model_ridge_scikit.score(x_poly_val, y_val)
        print('Validation set r-squared scores: {:.4f}'.format(score_r2_ridge_scikit_val))

        # Save MSE and R-squared validation scores
        scores_mse_ridge_scikit_val.append(score_mse_ridge_scikit_val)
        scores_r2_ridge_scikit_val.append(score_r2_ridge_scikit_val)

        # Test model on testing set
        score_mse_ridge_scikit_test = score_mean_squared_error(model_ridge_scikit, x_poly_test, y_test)
        print('Testing set mean squared error: {:.4f}'.format(score_mse_ridge_scikit_test))

        score_r2_ridge_scikit_test = model_ridge_scikit.score(x_poly_test, y_test)
        print('Testing set r-squared scores: {:.4f}'.format(score_r2_ridge_scikit_test))

        # Save MSE and R-squared testing scores
        scores_mse_ridge_scikit_test.append(score_mse_ridge_scikit_test)
        scores_r2_ridge_scikit_test.append(score_r2_ridge_scikit_test)

    # Convert each scores to NumPy arrays
    scores_mse_ridge_scikit_train = np.array(scores_mse_ridge_scikit_train)
    scores_mse_ridge_scikit_val = np.array(scores_mse_ridge_scikit_val)
    scores_mse_ridge_scikit_test = np.array(scores_mse_ridge_scikit_test)
    scores_r2_ridge_scikit_train = np.array(scores_r2_ridge_scikit_train)
    scores_r2_ridge_scikit_val = np.array(scores_r2_ridge_scikit_val)
    scores_r2_ridge_scikit_test = np.array(scores_r2_ridge_scikit_test)

    # Clip each set of MSE scores between 0 and 40
    scores_mse_ridge_scikit_train = np.clip(scores_mse_ridge_scikit_train, 0.0, 40.0)
    scores_mse_ridge_scikit_val = np.clip(scores_mse_ridge_scikit_val, 0.0, 40.0)
    scores_mse_ridge_scikit_test = np.clip(scores_mse_ridge_scikit_test, 0.0, 40.0)

    # Clip each set of R-squared scores between 0 and 1
    scores_r2_ridge_scikit_train = np.clip(scores_r2_ridge_scikit_train, 0.0, 1.0)
    scores_r2_ridge_scikit_val = np.clip(scores_r2_ridge_scikit_val, 0.0, 1.0)
    scores_r2_ridge_scikit_test = np.clip(scores_r2_ridge_scikit_test, 0.0, 1.0)

    # Create figure for training, validation and testing scores for different features
    n_experiments = scores_mse_ridge_scikit_train.shape[0]
    fig = plt.figure()

    labels = ['Training', 'Validation', 'Testing']
    colors = ['blue', 'red', 'green']

    # Create the first subplot of a 1 by 2 figure to plot MSE for training, validation, testing
    ax = fig.add_subplot(1, 2, 1)

    # TODO: Set x (alpha in log scale) and y values (MSE)
    x_values = [np.log(np.asarray(alphas))] * n_experiments
    y_values = [
        scores_mse_ridge_scikit_train,
        scores_mse_ridge_scikit_val,
        scores_mse_ridge_scikit_test
    ]

    # TODO: Plot MSE scores for training, validation, testing sets
    # Set x limits to 0 to max of x_values + 1 and y limits between 0 and 40
    # Set x label to 'alpha (log scale)' and y label to 'MSE',
    plot_results(
        axis=ax,
        x_values=x_values,
        y_values=y_values,
        labels=labels,
        colors=colors,
        x_limits=[0.0, np.log(alphas[-1]) + 1],
        y_limits=[0.0, 40.0],
        x_label='alpha (log scale)',
        y_label='MSE')

    # Create the second subplot of a 1 by 2 figure to plot R-squared for training, validation, testing
    ax = fig.add_subplot(1, 2, 2)

    # TODO: Set x (alpha in log scale) and y values (R-squared)
    n_r2_experiments = scores_r2_ridge_scikit_train.shape[0]
    x_values = [np.log(np.asarray(alphas))] * n_r2_experiments
    y_values = [
        scores_r2_ridge_scikit_train,
        scores_r2_ridge_scikit_val,
        scores_r2_ridge_scikit_test
    ]

    # TODO: Plot R-squared scores for training, validation, testing sets
    # Set x limits to 0 to max of x_values + 1 and y limits between 0 and 1
    # Set x label to 'alpha (log scale)' and y label to 'R-squared',
    plot_results(
        axis=ax,
        x_values=x_values,
        y_values=y_values,
        labels=labels,
        colors=colors,
        x_limits=[0.0, np.log(alphas[-1]) + 1],
        y_limits=[0.0, 1.0],
        x_label='alpha (log scale)',
        y_label='R-squared')

    # TODO: Create super title 'Scikit-Learn Ridge Regression on Training, Validation and Testing Sets'
    plt.suptitle('Scikit-Learn Ridge Regression on Training, Validation and Testing Sets')

    '''
    Trains and tests our ridge regression model using different alphas
    '''

    # Initialize empty lists to store scores for MSE and R-squared
    scores_mse_ridge_ours_train = []
    scores_r2_ridge_ours_train = []
    scores_mse_ridge_ours_val = []
    scores_r2_ridge_ours_val = []
    scores_mse_ridge_ours_test = []
    scores_r2_ridge_ours_test = []

    # TODO: convert dataset (N x d) to correct shape (d x N)
    
    x_poly_train = np.transpose(x_poly_train, axes=(1, 0))
    x_poly_val = np.transpose(x_poly_val, axes=(1, 0))
    x_poly_test = np.transpose(x_poly_test, axes=(1, 0))

    # For each alpha, train a ridge regression model on degree 2 polynomial features
    for alpha in alphas:

        # TODO: Initialize our own ridge regression model
        model_ridge = RidgeRegressionClosedForm()

        print('Results for our RidgeRegression model with alpha={}'.format(alpha))

        # TODO: Train model on training set
        model_ridge.fit(x_poly_train, y_train, alpha=alpha)

        # TODO: Test model on training set using mean squared error and r-squared
        predictions_train = model_ridge.predict(x_poly_train)
        score_mse_ridge_ours_train = model_ridge.score(predictions_train, y_train, scoring_func='mean_squared_error')
        print('Training set mean squared error: {:.4f}'.format(score_mse_ridge_ours_train))

        score_r2_ridge_ours_train = model_ridge.score(predictions_train, y_train, scoring_func='r_squared')
        print('Training set r-squared scores: {:.4f}'.format(score_r2_ridge_ours_train))

        # TODO: Save MSE and R-squared training scores
        scores_mse_ridge_ours_train.append(score_mse_ridge_ours_train)
        scores_r2_ridge_ours_train.append(score_r2_ridge_ours_train)

        # TODO: Test model on validation set using mean squared error and r-squared
        predictions_val = model_ridge.predict(x_poly_val)
        score_mse_ridge_ours_val = model_ridge.score(predictions_val, y_val, scoring_func='mean_squared_error')
        print('Validation set mean squared error: {:.4f}'.format(score_mse_ridge_ours_val))

        score_r2_ridge_ours_val = model_ridge.score(predictions_val, y_val, scoring_func='r_squared')
        print('Validation set r-squared scores: {:.4f}'.format(score_r2_ridge_ours_val))

        # TODO: Save MSE and R-squared validation scores
        scores_mse_ridge_ours_val.append(score_mse_ridge_ours_val)
        scores_r2_ridge_ours_val.append(score_r2_ridge_ours_val)

        # TODO: Test model on testing set using mean squared error and r-squared
        predictions_test = model_ridge.predict(x_poly_test)
        score_mse_ridge_ours_test = model_ridge.score(predictions_test, y_test, scoring_func='mean_squared_error')
        print('Testing set mean squared error: {:.4f}'.format(score_mse_ridge_ours_test))

        score_r2_ridge_ours_test = model_ridge.score(predictions_test, y_test, scoring_func='r_squared')
        print('Testing set r-squared scores: {:.4f}'.format(score_r2_ridge_ours_test))

        # TODO: Save MSE and R-squared testing scores
        scores_mse_ridge_ours_test.append(score_mse_ridge_ours_test)
        scores_r2_ridge_ours_test.append(score_r2_ridge_ours_test)

    # Convert each scores to NumPy arrays
    scores_mse_ridge_ours_train = np.array(scores_mse_ridge_ours_train)
    scores_mse_ridge_ours_val = np.array(scores_mse_ridge_ours_val)
    scores_mse_ridge_ours_test = np.array(scores_mse_ridge_ours_test)
    scores_r2_ridge_ours_train = np.array(scores_r2_ridge_ours_train)
    scores_r2_ridge_ours_val = np.array(scores_r2_ridge_ours_val)
    scores_r2_ridge_ours_test = np.array(scores_r2_ridge_ours_test)

    # TODO: Clip each set of MSE scores between 0 and 40
    scores_mse_ridge_ours_train = np.clip(scores_mse_ridge_ours_train, 0.0, 40.0)
    scores_mse_ridge_ours_val = np.clip(scores_mse_ridge_ours_val, 0.0, 40.0)
    scores_mse_ridge_ours_test = np.clip(scores_mse_ridge_ours_test, 0.0, 40.0)

    # TODO: Clip each set of R-squared scores between 0 and 1
    scores_r2_ridge_ours_train = np.clip(scores_r2_ridge_ours_train, 0.0, 1.0)
    scores_r2_ridge_ours_val = np.clip(scores_r2_ridge_ours_val, 0.0, 1.0)
    scores_r2_ridge_ours_test = np.clip(scores_r2_ridge_ours_test, 0.0, 1.0)

    # Create figure for training, validation and testing scores for different features
    n_experiments = scores_mse_ridge_ours_train.shape[0]
    fig = plt.figure()

    labels = ['Training', 'Validation', 'Testing']
    colors = ['blue', 'red', 'green']

    # Create the first subplot of a 1 by 2 figure to plot MSE for training, validation, testing
    ax = fig.add_subplot(1, 2, 1)

    # TODO: Set x (alpha in log scale) and y values (MSE)
    x_values = [np.log(np.asarray(alphas))] * n_experiments
    y_values = [
        scores_mse_ridge_ours_train,
        scores_mse_ridge_ours_val,
        scores_mse_ridge_ours_test
    ]

    # TODO: Plot MSE scores for training, validation, testing sets
    # Set x limits to 0 to max of x_values + 1 and y limits between 0 and 40
    # Set x label to 'alpha (log scale)' and y label to 'MSE',
    plot_results(
        axis=ax,
        x_values=x_values,
        y_values=y_values,
        labels=labels,
        colors=colors,
        x_limits=[0.0, np.log(alphas[-1]) + 1],
        y_limits=[0.0, 40.0],
        x_label='alpha (log scale)',
        y_label='MSE')

    # Create the second subplot of a 1 by 2 figure to plot R-squared for training, validation, testing
    ax = fig.add_subplot(1, 2, 2)

    # TODO: Set x (alpha in log scale) and y values (R-squared)
    n_r2_experiments = scores_r2_ridge_ours_train.shape[0]
    x_values = [np.log(np.asarray(alphas))] * n_r2_experiments
    y_values = [
        scores_r2_ridge_ours_train,
        scores_r2_ridge_ours_val,
        scores_r2_ridge_ours_test
    ]

    # TODO: Plot R-squared scores for training, validation, testing sets
    # Set x limits to 0 to max of x_values + 1 and y limits between 0 and 1
    # Set x label to 'alpha (log scale)' and y label to 'R-squared',
    plot_results(
        axis=ax,
        x_values=x_values,
        y_values=y_values,
        labels=labels,
        colors=colors,
        x_limits=[0.0, np.log(alphas[-1]) + 1],
        y_limits=[0.0, 1.0],
        x_label='alpha (log scale)',
        y_label='R-squared')

    # TODO: Create super title 'Our Ridge Regression on Training, Validation and Testing Sets'
    plt.suptitle('Our Ridge Regression on Training, Validation and Testing Sets')

    plt.show()

