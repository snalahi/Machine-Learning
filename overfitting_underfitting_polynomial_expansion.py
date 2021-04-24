#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import sklearn.datasets as skdata
import sklearn.metrics as skmetrics
import sklearn.preprocessing as skpreprocess
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge as RidgeRegression
from matplotlib import pyplot as plt


'''
Name: Alahi, Sk Nasimul (Please write names in <Last Name, First Name> format)

Collaborators: Wong, Alex (Please write names in <Last Name, First Name> format)

Collaboration details: The collaborator provided the skeleton and also all the major 
functions used in the following program.

Summary:
Report your scores here. For example,

Experiment 1: Overfitting Linear Regression with Polynomial Expansion
Results for linear regression model with degree 1 polynomial expansion
Training set mean squared error: 23.2560
Training set r-squared scores: 0.7323
Validation set mean squared error: 17.6111
Validation set r-squared scores: 0.7488
Testing set mean squared error: 17.1465
Testing set r-squared scores: 0.7805
Results for linear regression model with degree 2 polynomial expansion
Training set mean squared error: 6.6954
Training set r-squared scores: 0.9229
Validation set mean squared error: 9.5225
Validation set r-squared scores: 0.8642
Testing set mean squared error: 28.2841
Testing set r-squared scores: 0.6378
Results for linear regression model with degree 3 polynomial expansion
Training set mean squared error: 0.0000
Training set r-squared scores: 1.0000
Validation set mean squared error: 131226.9521
Validation set r-squared scores: -1870.9926
Testing set mean squared error: 119705.1131
Testing set r-squared scores: -1531.7115

Experiment 2: Underfitting Ridge Regression with alpha/lambda
Results for scikit-learn RidgeRegression model with alpha=0.0
Training set mean squared error: 23.2560
Training set r-squared scores: 0.7323
Validation set mean squared error: 17.6111
Validation set r-squared scores: 0.7488
Testing set mean squared error: 17.1465
Testing set r-squared scores: 0.7805
Results for scikit-learn RidgeRegression model with alpha=1.0
Training set mean squared error: 23.4415
Training set r-squared scores: 0.7302
Validation set mean squared error: 18.2272
Validation set r-squared scores: 0.7400
Testing set mean squared error: 17.3485
Testing set r-squared scores: 0.7779
Results for scikit-learn RidgeRegression model with alpha=10.0
Training set mean squared error: 24.0140
Training set r-squared scores: 0.7236
Validation set mean squared error: 19.4869
Validation set r-squared scores: 0.7220
Testing set mean squared error: 18.1566
Testing set r-squared scores: 0.7675
Results for scikit-learn RidgeRegression model with alpha=100.0
Training set mean squared error: 25.1813
Training set r-squared scores: 0.7102
Validation set mean squared error: 20.4889
Validation set r-squared scores: 0.7077
Testing set mean squared error: 19.6372
Testing set r-squared scores: 0.7486
Results for scikit-learn RidgeRegression model with alpha=1000.0
Training set mean squared error: 29.6060
Training set r-squared scores: 0.6593
Validation set mean squared error: 21.3000
Validation set r-squared scores: 0.6961
Testing set mean squared error: 22.4310
Testing set r-squared scores: 0.7128
Results for scikit-learn RidgeRegression model with alpha=10000.0
Training set mean squared error: 40.2823
Training set r-squared scores: 0.5364
Validation set mean squared error: 30.1993
Validation set r-squared scores: 0.5692
Testing set mean squared error: 33.5127
Testing set r-squared scores: 0.5709

Experiment 3: Ridge Regression with alpha/lambda and Polynomial Expansion
Results for ridge regression model with alpha=0.0 using degree 2 polynomial expansion
Training set mean squared error: 5.6754
Training set r-squared scores: 0.9347
Validation set mean squared error: 9.4461
Validation set r-squared scores: 0.8652
Testing set mean squared error: 22.6050
Testing set r-squared scores: 0.7106
Results for ridge regression model with alpha=1.0 using degree 2 polynomial expansion
Training set mean squared error: 6.3724
Training set r-squared scores: 0.9267
Validation set mean squared error: 9.6293
Validation set r-squared scores: 0.8626
Testing set mean squared error: 19.2863
Testing set r-squared scores: 0.7531
Results for ridge regression model with alpha=10.0 using degree 2 polynomial expansion
Training set mean squared error: 6.9915
Training set r-squared scores: 0.9195
Validation set mean squared error: 10.5660
Validation set r-squared scores: 0.8493
Testing set mean squared error: 18.0993
Testing set r-squared scores: 0.7683
Results for ridge regression model with alpha=100.0 using degree 2 polynomial expansion
Training set mean squared error: 7.8843
Training set r-squared scores: 0.9093
Validation set mean squared error: 11.9197
Validation set r-squared scores: 0.8300
Testing set mean squared error: 18.5883
Testing set r-squared scores: 0.7620
Results for ridge regression model with alpha=1000.0 using degree 2 polynomial expansion
Training set mean squared error: 8.8610
Training set r-squared scores: 0.8980
Validation set mean squared error: 11.7491
Validation set r-squared scores: 0.8324
Testing set mean squared error: 15.2857
Testing set r-squared scores: 0.8043
Results for ridge regression model with alpha=10000.0 using degree 2 polynomial expansion
Training set mean squared error: 10.0741
Training set r-squared scores: 0.8841
Validation set mean squared error: 11.7167
Validation set r-squared scores: 0.8329
Testing set mean squared error: 13.5444
Testing set r-squared scores: 0.8266
'''

def score_mean_squared_error(model, x, y):
    '''
    Scores the model on mean squared error metric

    Args:
        model : object
            trained model, assumes predict function returns N x 1 predictions
        x : numpy
            N x d numpy array of features
        y : numpy
            N x 1 groundtruth vector
    Returns:
        float : mean squared error
    '''

    # TODO: Implement the score mean squared error function
    predictions = model.predict(x)
    score_mse = skmetrics.mean_squared_error(predictions, y)
    return score_mse

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
        x_label : str
            name of x axis
        y_label : str
            name of y axis
    '''

    # TODO: Iterate through x_values, y_values, labels, and colors and plot them
    # with associated legend
    for x, y, label, color in zip(x_values, y_values, labels, colors):
        axis.plot(x, y, marker='o', color=color, label=label)
        axis.legend(loc='best')

    # TODO: Set x and y limits
    axis.set_xlim(x_limits)
    axis.set_ylim(y_limits)

    # TODO: Set x and y labels
    axis.set_xlabel(x_label)
    axis.set_ylabel(y_label)


# In[2]:


if __name__ == '__main__':

    boston_data = skdata.load_boston()
    x = boston_data.data
    y = boston_data.target

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
    Experiment 1:
    Demonstrate that linear regression will overfit if we use polynomial expansion
    '''

    print('Experiment 1: Overfitting Linear Regression with Polynomial Expansion')

    # TODO: Initialize a list containing 1, 2, 3 as the degrees for polynomial expansion
    degrees = [1, 2, 3]

    # Initialize empty lists to store scores for MSE and R-squared
    scores_mse_linear_overfit_train = []
    scores_r2_linear_overfit_train = []
    scores_mse_linear_overfit_val = []
    scores_r2_linear_overfit_val = []
    scores_mse_linear_overfit_test = []
    scores_r2_linear_overfit_test = []

    for degree in degrees:

        # TODO: Initialize polynomial expansion
        poly_transform = skpreprocess.PolynomialFeatures(degree=degree)

        # TODO: Compute the polynomial terms needed for the data
        poly_transform.fit(x_train)

        # TODO: Transform the data by nonlinear mapping
        x_poly_train = poly_transform.transform(x_train)
        x_poly_val = poly_transform.transform(x_val)
        x_poly_test = poly_transform.transform(x_test)

        # TODO: Initialize scikit-learn linear regression model
        model_linear_overfit = LinearRegression()

        # TODO: Trains scikit-learn linear regression model
        model_linear_overfit.fit(x_poly_train, y_train)

        print('Results for linear regression model with degree {} polynomial expansion'.format(degree))

        # TODO: Test model on training set
        score_mse_linear_overfit_train = score_mean_squared_error(model_linear_overfit, x_poly_train, y_train)
        print('Training set mean squared error: {:.4f}'.format(score_mse_linear_overfit_train))

        score_r2_linear_overfit_train =  model_linear_overfit.score(x_poly_train, y_train)
        print('Training set r-squared scores: {:.4f}'.format(score_r2_linear_overfit_train))

        # TODO: Save MSE and R-squared training scores
        scores_mse_linear_overfit_train.append(score_mse_linear_overfit_train)
        scores_r2_linear_overfit_train.append(score_r2_linear_overfit_train)

        # TODO: Test model on validation set
        score_mse_linear_overfit_val = score_mean_squared_error(model_linear_overfit, x_poly_val, y_val)
        print('Validation set mean squared error: {:.4f}'.format(score_mse_linear_overfit_val))

        score_r2_linear_overfit_val = model_linear_overfit.score(x_poly_val, y_val)
        print('Validation set r-squared scores: {:.4f}'.format(score_r2_linear_overfit_val))

        # TODO: Save MSE and R-squared validation scores
        scores_mse_linear_overfit_val.append(score_mse_linear_overfit_val)
        scores_r2_linear_overfit_val.append(score_r2_linear_overfit_val)

        # TODO: Test model on testing set
        score_mse_linear_overfit_test = score_mean_squared_error(model_linear_overfit, x_poly_test, y_test)
        print('Testing set mean squared error: {:.4f}'.format(score_mse_linear_overfit_test))

        score_r2_linear_overfit_test = model_linear_overfit.score(x_poly_test, y_test)
        print('Testing set r-squared scores: {:.4f}'.format(score_r2_linear_overfit_test))

        # TODO: Save MSE and R-squared testing scores
        scores_mse_linear_overfit_test.append(score_mse_linear_overfit_test)
        scores_r2_linear_overfit_test.append(score_r2_linear_overfit_test)

    # Convert each scores to NumPy arrays
    scores_mse_linear_overfit_train = np.array(scores_mse_linear_overfit_train)
    scores_mse_linear_overfit_val = np.array(scores_mse_linear_overfit_val)
    scores_mse_linear_overfit_test = np.array(scores_mse_linear_overfit_test)
    scores_r2_linear_overfit_train = np.array(scores_r2_linear_overfit_train)
    scores_r2_linear_overfit_val = np.array(scores_r2_linear_overfit_val)
    scores_r2_linear_overfit_test = np.array(scores_r2_linear_overfit_test)

    # TODO: Clip each set of MSE scores between 0 and 40
    scores_mse_linear_overfit_train = np.clip(scores_mse_linear_overfit_train, 0.0, 40.0)
    scores_mse_linear_overfit_val = np.clip(scores_mse_linear_overfit_val, 0.0, 40.0)
    scores_mse_linear_overfit_test = np.clip(scores_mse_linear_overfit_test, 0.0, 40.0)

    # TODO: Clip each set of R-squared scores between 0 and 1
    scores_r2_linear_overfit_train = np.clip(scores_r2_linear_overfit_train, 0.0, 1.0)
    scores_r2_linear_overfit_val = np.clip(scores_r2_linear_overfit_val, 0.0, 1.0)
    scores_r2_linear_overfit_test = np.clip(scores_r2_linear_overfit_test, 0.0, 1.0)

    # Create figure for training, validation and testing scores for different features
    n_experiments = scores_mse_linear_overfit_train.shape[0]
    fig = plt.figure()

    labels = ['Training', 'Validation', 'Testing']
    colors = ['blue', 'red', 'green']

    # TODO: Create the first subplot of a 1 by 2 figure to plot MSE for training, validation, testing
    ax = fig.add_subplot(1, 2, 1)
    
    # TODO: Set x and y values
    x_values = [range(1, n_experiments + 1)] * n_experiments
    y_values = [
        scores_mse_linear_overfit_train,
        scores_mse_linear_overfit_val,
        scores_mse_linear_overfit_test
    ]

    # TODO: Plot MSE scores for training, validation, testing sets
    # Set x limits to 0 to number of experiments + 1 and y limits between 0 and 40
    # Set x label to 'p-degree' and y label to 'MSE',
    plot_results(
        axis=ax,
        x_values=x_values,
        y_values=y_values,
        labels=labels,
        colors=colors,
        x_limits=[0.0, n_experiments + 1],
        y_limits=[0.0, 40.0],
        x_label='p-degree',
        y_label='MSE')

    # TODO: Create the second subplot of a 1 by 2 figure to plot R-squared for training, validation, testing
    ax = fig.add_subplot(1, 2, 2)

    # TODO: Set x and y values
    n_r2_experiments = scores_r2_linear_overfit_train.shape[0]
    x_values = [range(1, n_r2_experiments + 1)] * n_r2_experiments
    y_values = [
        scores_r2_linear_overfit_train,
        scores_r2_linear_overfit_val,
        scores_r2_linear_overfit_test
    ]

    # TODO: Plot R-squared scores for training, validation, testing sets
    # Set x limits to 0 to number of experiments + 1 and y limits between 0 and 1
    # Set x label to 'p-degree' and y label to 'R-squared',
    plot_results(
        axis=ax,
        x_values=x_values,
        y_values=y_values,
        labels=labels,
        colors=colors,
        x_limits=[0.0, n_experiments + 1],
        y_limits=[0.0, 1.0],
        x_label='p-degree',
        y_label='R-squared')

    # TODO: Create super title 'Overfitted Linear Regression on Training, Validation and Testing Sets'
    plt.suptitle('Overfitted Linear Regression on Training, Validation and Testing Sets')

    plt.show()

    '''
    Experiment 2:
    Demonstrate that ridge regression will underfit with high weight (alpha/lambda) values
    '''

    print('Experiment 2: Underfitting Ridge Regression with alpha/lambda')

    # TODO: Initialize a list containing:
    # 0.0, 1.0, 10.0, 100.0, 1000.0, 10000.0 as the degrees for polynomial expansion
    alphas = [0.0, 1.0, 10.0, 100.0, 1000.0, 10000.0]

    # Initialize empty lists to store scores for MSE and R-squared
    scores_mse_ridge_underfit_train = []
    scores_r2_ridge_underfit_train = []
    scores_mse_ridge_underfit_val = []
    scores_r2_ridge_underfit_val = []
    scores_mse_ridge_underfit_test = []
    scores_r2_ridge_underfit_test = []

    for alpha in alphas:
        # TODO: Initialize scikit-learn ridge regression model
        model_ridge_underfit = RidgeRegression(alpha=alpha)

        # TODO: Trains scikit-learn ridge regression model
        model_ridge_underfit.fit(x_train, y_train)

        print('Results for scikit-learn RidgeRegression model with alpha={}'.format(alpha))

        # TODO: Test model on training set
        score_mse_ridge_underfit_train = score_mean_squared_error(model_ridge_underfit, x_train, y_train)
        print('Training set mean squared error: {:.4f}'.format(score_mse_ridge_underfit_train))

        score_r2_ridge_underfit_train = model_ridge_underfit.score(x_train, y_train)
        print('Training set r-squared scores: {:.4f}'.format(score_r2_ridge_underfit_train))

        # TODO: Save MSE and R-squared training scores
        scores_mse_ridge_underfit_train.append(score_mse_ridge_underfit_train)
        scores_r2_ridge_underfit_train.append(score_r2_ridge_underfit_train)

        # TODO: Test model on validation set
        score_mse_ridge_underfit_val = score_mean_squared_error(model_ridge_underfit, x_val, y_val)
        print('Validation set mean squared error: {:.4f}'.format(score_mse_ridge_underfit_val))

        score_r2_ridge_underfit_val = model_ridge_underfit.score(x_val, y_val)
        print('Validation set r-squared scores: {:.4f}'.format(score_r2_ridge_underfit_val))

        # TODO: Save MSE and R-squared validation scores
        scores_mse_ridge_underfit_val.append(score_mse_ridge_underfit_val)
        scores_r2_ridge_underfit_val.append(score_r2_ridge_underfit_val)

        # TODO: Test model on testing set
        score_mse_ridge_underfit_test = score_mean_squared_error(model_ridge_underfit, x_test, y_test)
        print('Testing set mean squared error: {:.4f}'.format(score_mse_ridge_underfit_test))

        score_r2_ridge_underfit_test = model_ridge_underfit.score(x_test, y_test)
        print('Testing set r-squared scores: {:.4f}'.format(score_r2_ridge_underfit_test))

        # TODO: Save MSE and R-squared testing scores
        scores_mse_ridge_underfit_test.append(score_mse_ridge_underfit_test)
        scores_r2_ridge_underfit_test.append(score_r2_ridge_underfit_test)

    # Convert each scores to NumPy arrays
    scores_mse_ridge_underfit_train = np.array(scores_mse_ridge_underfit_train)
    scores_mse_ridge_underfit_val = np.array(scores_mse_ridge_underfit_val)
    scores_mse_ridge_underfit_test = np.array(scores_mse_ridge_underfit_test)
    scores_r2_ridge_underfit_train = np.array(scores_r2_ridge_underfit_train)
    scores_r2_ridge_underfit_val = np.array(scores_r2_ridge_underfit_val)
    scores_r2_ridge_underfit_test = np.array(scores_r2_ridge_underfit_test)

    # TODO: Clip each set of MSE scores between 0 and 40
    scores_mse_ridge_underfit_train = np.clip(scores_mse_ridge_underfit_train, 0.0, 40.0)
    scores_mse_ridge_underfit_val = np.clip(scores_mse_ridge_underfit_val, 0.0, 40.0)
    scores_mse_ridge_underfit_test = np.clip(scores_mse_ridge_underfit_test, 0.0, 40.0)

    # TODO: Clip each set of R-squared scores between 0 and 1
    scores_r2_ridge_underfit_train = np.clip(scores_r2_ridge_underfit_train, 0.0, 1.0)
    scores_r2_ridge_underfit_val = np.clip(scores_r2_ridge_underfit_val, 0.0, 1.0)
    scores_r2_ridge_underfit_test = np.clip(scores_r2_ridge_underfit_test, 0.0, 1.0)

    # Create figure for training, validation and testing scores for different features
    n_experiments = scores_mse_ridge_underfit_train.shape[0]
    fig = plt.figure()

    labels = ['Training', 'Validation', 'Testing']
    colors = ['blue', 'red', 'green']

    # TODO: Create the first subplot of a 1 by 2 figure to plot MSE for training, validation, testing
    ax = fig.add_subplot(1, 2, 1)

    # TODO: Set x values (alphas in log scale )and y values (R-squared)
    x_values = [np.log(np.asarray(alphas) + 1.0)] * n_experiments
    y_values = [
        scores_mse_ridge_underfit_train,
        scores_mse_ridge_underfit_val,
        scores_mse_ridge_underfit_test
    ]

    # TODO: Plot MSE scores for training, validation, testing sets
    # Set x limits to 0 to log of highest alphas + 1 and y limits between 0 and 40
    # Set x label to 'alphas' and y label to 'MSE',
    plot_results(
        axis=ax,
        x_values=x_values,
        y_values=y_values,
        labels=labels,
        colors=colors,
        x_limits=[0.0, np.log(alphas[-1]) + 1],
        y_limits=[0.0, 40.0],
        x_label='alphas',
        y_label='MSE')

    # TODO: Create the second subplot of a 1 by 2 figure to plot R-squared for training, validation, testing
    ax = fig.add_subplot(1, 2, 2)

    # TODO: Set x values (alphas in log scale) and y values (R-squared)
    n_r2_experiments = scores_r2_ridge_underfit_train.shape[0]
    x_values = [np.log(np.asarray(alphas) + 1.0)] * n_r2_experiments
    y_values = [
        scores_r2_ridge_underfit_train,
        scores_r2_ridge_underfit_val,
        scores_r2_ridge_underfit_test
    ]

    # TODO: Plot R-squared scores for training, validation, testing sets
    # Set x limits to 0 to 1100 and y limits between 0 and 1
    # Set x label to 'alphas' and y label to 'R-squared',
    plot_results(
        axis=ax,
        x_values=x_values,
        y_values=y_values,
        labels=labels,
        colors=colors,
        x_limits=[0.0, np.log(alphas[-1]) + 1],
        y_limits=[0.0, 1.0],
        x_label='alphas',
        y_label='R-squared')

    # TODO: Create super title 'Underfitted Ridge Regression on Training, Validation and Testing Sets'
    plt.suptitle('Underfitted Ridge Regression on Training, Validation and Testing Sets')

    plt.show()

    '''
    Experiment 3:
    Demonstrate that ridge regression with various alpha/lambda prevents overfitting
    when using polynomial expansion of degree 2
    '''

    print('Experiment 3: Ridge Regression with alpha/lambda and Polynomial Expansion')

    degree = 2

    # TODO: Initialize a list containing:
    # 0.0, 1.0, 10.0, 100.0, 1000.0, 10000.0 as the degrees for polynomial expansion
    alphas = [0.0, 1.0, 10.0, 100.0, 1000.0, 10000.0]

    # TODO: Initialize polynomial expansion
    poly_transform = skpreprocess.PolynomialFeatures(degree=degree)

    # TODO: Compute the polynomial terms needed for the data
    poly_transform.fit(x_train)

    # TODO: Transform the data by nonlinear mapping
    x_poly_train = poly_transform.transform(x_train)
    x_poly_val = poly_transform.transform(x_val)
    x_poly_test = poly_transform.transform(x_test)

    # Initialize empty lists to store scores for MSE and R-squared
    scores_mse_ridge_poly_train = []
    scores_r2_ridge_poly_train = []
    scores_mse_ridge_poly_val = []
    scores_r2_ridge_poly_val = []
    scores_mse_ridge_poly_test = []
    scores_r2_ridge_poly_test = []

    for alpha in alphas:

        # TODO: Initialize scikit-learn linear regression model
        model_ridge_robust_fit = RidgeRegression(alpha=alpha)

        # TODO: Trains scikit-learn linear regression model
        model_ridge_robust_fit.fit(x_poly_train, y_train)

        print('Results for ridge regression model with alpha={} using degree {} polynomial expansion'.format(alpha, degree))

        # TODO: Test model on training set
        score_mse_ridge_poly_train = score_mean_squared_error(model_ridge_robust_fit, x_poly_train, y_train)
        print('Training set mean squared error: {:.4f}'.format(score_mse_ridge_poly_train))

        score_r2_ridge_poly_train = model_ridge_robust_fit.score(x_poly_train, y_train)
        print('Training set r-squared scores: {:.4f}'.format(score_r2_ridge_poly_train))

        # TODO: Save MSE and R-squared training scores
        scores_mse_ridge_poly_train.append(score_mse_ridge_poly_train)
        scores_r2_ridge_poly_train.append(score_r2_ridge_poly_train)

        # TODO: Test model on validation set
        score_mse_ridge_poly_val = score_mean_squared_error(model_ridge_robust_fit, x_poly_val, y_val)
        print('Validation set mean squared error: {:.4f}'.format(score_mse_ridge_poly_val))

        score_r2_ridge_poly_val = model_ridge_robust_fit.score(x_poly_val, y_val)
        print('Validation set r-squared scores: {:.4f}'.format(score_r2_ridge_poly_val))

        # TODO: Save MSE and R-squared validation scores
        scores_mse_ridge_poly_val.append(score_mse_ridge_poly_val)
        scores_r2_ridge_poly_val.append(score_r2_ridge_poly_val)

        # TODO: Test model on testing set
        score_mse_ridge_poly_test = score_mean_squared_error(model_ridge_robust_fit, x_poly_test, y_test)
        print('Testing set mean squared error: {:.4f}'.format(score_mse_ridge_poly_test))

        score_r2_ridge_poly_test = model_ridge_robust_fit.score(x_poly_test, y_test)
        print('Testing set r-squared scores: {:.4f}'.format(score_r2_ridge_poly_test))

        # TODO: Save MSE and R-squared testing scores
        scores_mse_ridge_poly_test.append(score_mse_ridge_poly_test)
        scores_r2_ridge_poly_test.append(score_r2_ridge_poly_test)

    # Convert each scores to NumPy arrays
    scores_mse_ridge_poly_train = np.array(scores_mse_ridge_poly_train)
    scores_mse_ridge_poly_val = np.array(scores_mse_ridge_poly_val)
    scores_mse_ridge_poly_test = np.array(scores_mse_ridge_poly_test)
    scores_r2_ridge_poly_train = np.array(scores_r2_ridge_poly_train)
    scores_r2_ridge_poly_val = np.array(scores_r2_ridge_poly_val)
    scores_r2_ridge_poly_test = np.array(scores_r2_ridge_poly_test)

    # TODO: Clip each set of MSE scores between 0 and 40
    scores_mse_ridge_poly_train = np.clip(scores_mse_ridge_poly_train, 0.0, 40.0)
    scores_mse_ridge_poly_val = np.clip(scores_mse_ridge_poly_val, 0.0, 40.0)
    scores_mse_ridge_poly_test = np.clip(scores_mse_ridge_poly_test, 0.0, 40.0)

    # TODO: Clip each set of R-squared scores between 0 and 1
    scores_r2_ridge_poly_train = np.clip(scores_r2_ridge_poly_train, 0.0, 1.0)
    scores_r2_ridge_poly_val = np.clip(scores_r2_ridge_poly_val, 0.0, 1.0)
    scores_r2_ridge_poly_test = np.clip(scores_r2_ridge_poly_test, 0.0, 1.0)

    # Create figure for training, validation and testing scores for different features
    n_experiments = scores_mse_ridge_poly_train.shape[0]
    fig = plt.figure()

    labels = ['Training', 'Validation', 'Testing']
    colors = ['blue', 'red', 'green']

    # TODO: Create the first subplot of a 1 by 2 figure to plot MSE for training, validation, testing
    ax = fig.add_subplot(1, 2, 1)

    # TODO: Set x values (alphas in log scale) and y values (R-squared)
    x_values = [np.log(np.asarray(alphas) + 1.0)] * n_experiments
    y_values = [
        scores_mse_ridge_poly_train,
        scores_mse_ridge_poly_val,
        scores_mse_ridge_poly_test
    ]

    # TODO: Plot MSE scores for training, validation, testing sets
    # Set x limits to 0 to 1100 and y limits between 0 and 40
    # Set x label to 'alphas' and y label to 'MSE',
    plot_results(
        axis=ax,
        x_values=x_values,
        y_values=y_values,
        labels=labels,
        colors=colors,
        x_limits=[0.0, np.log(alphas[-1]) + 1],
        y_limits=[0.0, 40.0],
        x_label='alphas',
        y_label='MSE')

    # TODO: Create the second subplot of a 1 by 2 figure to plot R-squared for training, validation, testing
    ax = fig.add_subplot(1, 2, 2)

    # TODO: Set x values (alphas in log scale )and y values (R-squared)
    n_r2_experiments = scores_r2_ridge_poly_train.shape[0]
    x_values = [np.log(np.asarray(alphas) + 1.0)] * n_r2_experiments
    y_values = [
        scores_r2_ridge_poly_train,
        scores_r2_ridge_poly_val,
        scores_r2_ridge_poly_test
    ]

    # TODO: Plot R-squared scores for training, validation, testing sets
    # Set x limits to 0 to 1100 and y limits between 0 and 1
    # Set x label to 'alphas' and y label to 'R-squared',
    plot_results(
        axis=ax,
        x_values=x_values,
        y_values=y_values,
        labels=labels,
        colors=colors,
        x_limits=[0.0, np.log(alphas[-1]) + 1],
        y_limits=[0.0, 1.0],
        x_label='alphas',
        y_label='R-squared')

    # TODO: Create super title 'Ridge Regression with various alphas on Training, Validation and Testing Sets'
    plt.suptitle('Ridge Regression with various alphas on Training, Validation and Testing Sets')

    plt.show()
