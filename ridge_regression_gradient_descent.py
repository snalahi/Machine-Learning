#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import sklearn.datasets as skdata
import sklearn.metrics as skmetrics
import sklearn.preprocessing as skpreprocess
from sklearn.linear_model import Ridge as RidgeRegression


'''
Name: Alahi, Sk Nasimul (Please write names in <Last Name, First Name> format)

Collaborators: Wong, Alex (Please write names in <Last Name, First Name> format)

Collaboration details: The collaborator provided the skeleton and also the
score_mean_squared_error function with application of scikit-learn Ridge Regression
model. Discussed with him the fit() and predict() functions and the overall
performance of the program output.

Summary:

TODO: Please answer the following questions and report your scores

1. What did you observe when using larger versus smaller momentum for
momentum gradient descent and momentum stochastic gradient descent?

Ans: When I used larger momentum or beta, the gradient converged to the global or
local minima faster. In other words, loss reduced faster. On the other hand, smaller
momentum or beta took longer to reduce the loss and converge.

2. What did you observe when using larger versus smaller batch size
for stochastic gradient descent?

Ans: In stochastic gradient descent, each iteration updates the weights by the
gradient with zero-mean noise and variance depending on size of batch. When I had
larger batch size, associated variance tended to be zero and the gradient moved to
the correct direction faster. Which eventually, reduced the loss faster. On the other
hand, when I used smaller batch size, associated variance was high and the gradient
took high number of steps to move to the actual direction. In other words, convergence
and reduction of loss became slower.

3. Explain the difference between gradient descent, momentum gradient descent,
stochastic gradient descent, and momentum stochastic gradient descent?

Ans: The main difference between gradient descent and stochastic gradient descent is
that, in case of gradient descent, the gradient of the whole dataset (full gradient)
is calculated once at a time which can be really expensive due to the high orders of a
real dataset. To get rid of that we use stochastic gradient descent where full gradient
is calculated in a batchwise manner. We sample out small batches or mini-batches from the
whole dataset, take their gradients and sum them all up to aproximate the full gradient.
That is why, in comparison to Gradient Descent, Stochastic Gradient Descent is:
--> cheaper per iteration computation
--> faster convergence in the beginning
**but**
--> is less stable
--> slower final convergence
--> difficult to tune step size (eta^t)

The above elaboration works accordingly to differentiate momentum gradient descent and
momentum stochastic gradient descent. The only thing that adds to the above is the
initiation of momentum while calculating gradient. And the momentum approach has the
following advantages:
--> Designed to accelerate learning, especially for functions with high curvatures or
    noisy gradients
--> Applied to both Gradient Descent and Stochastic Gradient Descent
--> Speeds up convergence for Gradient Descent
--> Reduces the variance introduced by stochastic sampling in case of Stochastic
    Gradient Descent


Report your scores here.

Results on using scikit-learn Ridge Regression model
Training set mean squared error: 2749.2155
Validation set mean squared error: 3722.5782
Testing set mean squared error: 3169.6860
Results on using Ridge Regression using gradient descent variants
Fitting with gradient_descent using learning rate=4.0E-01, t=8000
Training set mean squared error: 2753.2056
Validation set mean squared error: 3726.9486
Testing set mean squared error: 3170.4715
Fitting with momentum_gradient_descent using learning rate=5.0E-01, t=8000
Training set mean squared error: 2751.6429
Validation set mean squared error: 3725.1999
Testing set mean squared error: 3170.4279
Fitting with stochastic_gradient_descent using learning rate=1.2E+00, t=20000
Training set mean squared error: 2769.4864
Validation set mean squared error: 3740.1901
Testing set mean squared error: 3175.3508
Fitting with momentum_stochastic_gradient_descent using learning rate=1.8E+00, t=20000
Training set mean squared error: 2765.3814
Validation set mean squared error: 3735.1896
Testing set mean squared error: 3186.0410
'''


def score_mean_squared_error(model, x, y):
    '''
    Scores the model on mean squared error metric

    Args:
        model : object
            trained model, assumes predict function returns N x d predictions
        x : numpy
            d x N numpy array of features
        y : numpy
            N element groundtruth vector
    Returns:
        float : mean squared error
    '''

    # Computes the mean squared error
    predictions = model.predict(x)
    mse = skmetrics.mean_squared_error(predictions, y)
    return mse


'''
Implementation of our gradient descent optimizer for ridge regression
'''
class GradientDescentOptimizer(object):

    def __init__(self, learning_rate):
        self.__momentum = None
        self.__learning_rate = learning_rate

    def __compute_gradients(self, w, x, y, lambda_weight_decay):
        '''
        Returns the gradient of the mean squared loss with weight decay

        Args:
            w : numpy
                d x 1 weight vector
            x : numpy
                d x N feature vector
            y : numpy
                N element groundtruth vector
            lambda_weight_decay : float
                weight of weight decay

        Returns:
            numpy : 1 x d gradients
        '''

        # TODO: Implements the __compute_gradients function
        x = np.concatenate([np.ones([1, x.shape[1]]), x], axis=0)
        
        w_trans_x_sub_y = (np.matmul(w.T, x) - y)
        data_fidelity_gradient = 2.0 * np.mean(w_trans_x_sub_y * x, axis=1)
        regularization_gradient = 2.0 * (lambda_weight_decay / float(x.shape[1])) * w

        return data_fidelity_gradient + regularization_gradient

    def __cube_root_decay(self, time_step):
        '''
        Computes the cube root polynomial decay factor t^{-1/3}

        Args:
            time_step : int
                current step in optimization

        Returns:
            float : cube root decay factor to adjust learning rate
        '''

        # TODO: Implement cube root polynomial decay factor to adjust learning rate
        cube_root_decay = 1.0 / (np.power(float(time_step), (1.0 / 3.0)))
        
        return cube_root_decay

    def update(self,
               w,
               x,
               y,
               optimizer_type,
               lambda_weight_decay,
               beta,
               batch_size,
               time_step):
        '''
        Updates the weight vector based on

        Args:
            w : numpy
                1 x d weight vector
            x : numpy
                d x N feature vector
            y : numpy
                N element groundtruth vector
            optimizer_type : str
                'gradient_descent',
                'momentum_gradient_descent',
                'stochastic_gradient_descent',
                'momentum_stochastic_gradient_descent'
            lambda_weight_decay : float
                weight of weight decay
            beta : str
                momentum discount rate
            batch_size : int
                batch size for stochastic and momentum stochastic gradient descent
            time_step : int
                current step in optimization

        Returns:
            numpy : 1 x d weights
        '''

        # TODO: Implement the optimizer update function

        if self.__momentum is None:
            self.__momentum = np.zeros_like(w)

        if optimizer_type == 'gradient_descent':

            # TODO: Compute gradients
            gradients = self.__compute_gradients(w, x, y, lambda_weight_decay)

            # TODO: Update weights
            return w - (self.__learning_rate * gradients)

        elif optimizer_type == 'momentum_gradient_descent':

            # TODO: Compute gradients
            gradients = self.__compute_gradients(w, x, y, lambda_weight_decay)

            # TODO: Compute momentum
            self.__momentum = (beta * self.__momentum) + ((1.0 - beta) * gradients)

            # TODO: Update weights
            return w - (self.__learning_rate * self.__momentum)

        elif optimizer_type == 'stochastic_gradient_descent':

            # TODO: Implement stochastic gradient descent
            batch_idx = np.random.permutation(x.shape[1])[0:batch_size]

            # TODO: Sample batch from dataset
            x_batch = x[:, batch_idx]
            y_batch = y[batch_idx]

            # TODO: Compute gradients
            gradients = self.__compute_gradients(w, x_batch, y_batch, lambda_weight_decay)

            # TODO: Compute cube root decay factor and multiply by learning rate
            eta = self.__cube_root_decay(time_step) * self.__learning_rate
            
            # TODO: Update weights
            return w - (eta * gradients)

        elif optimizer_type == 'momentum_stochastic_gradient_descent':

            # TODO: Implement momentum stochastic gradient descent
            batch_idx = np.random.permutation(x.shape[1])[0:batch_size]

            # TODO: Sample batch from dataset
            x_batch = x[:, batch_idx]
            y_batch = y[batch_idx]

            # TODO: Compute gradients
            gradients = self.__compute_gradients(w, x_batch, y_batch, lambda_weight_decay)

            # TODO: Compute momentum
            self.__momentum = (beta * self.__momentum) + ((1.0 - beta) * gradients)

            # TODO: Compute cube root decay factor and multiply by learning rate
            eta = self.__cube_root_decay(time_step) * self.__learning_rate

            # TODO: Update weights
            return w - (eta * self.__momentum)


'''
Implementation of our Ridge Regression model trained using gradient descent variants
'''
class RidgeRegressionGradientDescent(object):

    def __init__(self):
        # Define private variables
        self.__weights = None

    def fit(self,
            x,
            y,
            optimizer_type,
            learning_rate,
            t,
            lambda_weight_decay,
            beta,
            batch_size):
        '''
        Fits the model to x and y by updating the weight vector
        using gradient descent variants

        Args:
            x : numpy
                d x N feature vector
            y : numpy
                N element groundtruth vector
            optimizer_type : str
                'gradient_descent',
                'momentum_gradient_descent',
                'stochastic_gradient_descent',
                'momentum_stochastic_gradient_descent'
            learning_rate : float
                learning rate
            t : int
                number of iterations to train
            lambda_weight_decay : float
                weight of weight decay
            beta : str
                momentum discount rate
            batch_size : int
                batch size for stochastic and momentum stochastic gradient descent
        '''

        # TODO: Implement the fit function

        # TODO: Initialize weights
        self.__weights = np.zeros([x.shape[0] + 1, 1])
        self.__weights[0] = 1.0

        # TODO: Initialize optimizer
        self.__optimizer = GradientDescentOptimizer(learning_rate)

        for time_step in range(1, t + 1):
            # TODO: Compute loss function
            loss, loss_data_fidelity, loss_regularization =                 self.__compute_loss(x, y, lambda_weight_decay)

            if (time_step % 500) == 0:
                print('Step={:5}  Loss={:.4f}  Data Fidelity={:.4f}  Regularization={:.4f}'.format(
                    time_step, loss, loss_data_fidelity, loss_regularization))

            # TODO: Update weights
            self.__weights = np.squeeze(self.__weights)
            w_i = self.__optimizer.update(
                self.__weights, x, y, optimizer_type, lambda_weight_decay,
                beta, batch_size, time_step)

            self.__weights = w_i

    def predict(self, x):
        '''
        Predicts the label for each feature vector x

        Args:
            x : numpy
                d x N feature vector

        Returns:
            numpy : N element vector
        '''

        # TODO: Implements the predict function
        # Add bias to x (d, N) -> (d + 1, N)
        x = np.concatenate([np.ones([1, x.shape[1]]), x], axis=0)
        
        self.__weights = np.expand_dims(self.__weights, axis=-1)
        predictions = np.matmul(self.__weights.T, x)
        predictions = np.squeeze(predictions)
        
        return predictions

    def __compute_loss(self, x, y, lambda_weight_decay):
        '''
        Returns the gradient of the mean squared loss with weight decay

        Args:
            x : numpy
                d x N feature vector
            y : numpy
                N element groundtruth vector
            lambda_weight_decay : float
                weight of weight decay

        Returns:
            float : loss
            float : loss data fidelity
            float : loss regularization
        '''

        # TODO: Implements the __compute_loss function
        y_hat = self.predict(x)
        
        loss_data_fidelity = np.mean((y_hat - y) ** 2)
        loss_regularization = lambda_weight_decay * np.mean(self.__weights ** 2)
        loss = loss_data_fidelity + loss_regularization

        return loss, loss_data_fidelity, loss_regularization


if __name__ == '__main__':

    # Loads dataset with 80% training, 10% validation, 10% testing split
    data = skdata.load_diabetes()
    x = data.data
    y = data.target

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

    x_train, x_val, x_test =         x[train_idx, :], x[val_idx, :], x[test_idx, :]
    y_train, y_val, y_test =         y[train_idx], y[val_idx], y[test_idx]

    # Initialize polynomial expansion

    poly_transform = skpreprocess.PolynomialFeatures(degree=2)

    # Compute the polynomial terms needed for the data
    poly_transform.fit(x_train)

    # Transform the data by nonlinear mapping
    x_train = poly_transform.transform(x_train)
    x_val = poly_transform.transform(x_val)
    x_test = poly_transform.transform(x_test)

    lambda_weight_decay = 0.1

    '''
    Trains and tests Ridge Regression model from scikit-learn
    '''

    # Trains scikit-learn Ridge Regression model on diabetes data
    ridge_scikit = RidgeRegression(alpha=lambda_weight_decay)
    ridge_scikit.fit(x_train, y_train)

    print('Results on using scikit-learn Ridge Regression model')

    # Test model on training set
    scores_mse_train_scikit = score_mean_squared_error(
        ridge_scikit, x_train, y_train)
    print('Training set mean squared error: {:.4f}'.format(scores_mse_train_scikit))

    # Test model on validation set
    scores_mse_val_scikit = score_mean_squared_error(
        ridge_scikit, x_val, y_val)
    print('Validation set mean squared error: {:.4f}'.format(scores_mse_val_scikit))

    # Test model on testing set
    scores_mse_test_scikit = score_mean_squared_error(
        ridge_scikit, x_test, y_test)
    print('Testing set mean squared error: {:.4f}'.format(scores_mse_test_scikit))

    '''
    Trains and tests our Ridge Regression model trained using gradient descent variants
    '''

    # Optimization types to use
    optimizer_types = [
        'gradient_descent',
        'momentum_gradient_descent',
        'stochastic_gradient_descent',
        'momentum_stochastic_gradient_descent'
    ]

    # TODO: Select learning rates for each optimizer
    learning_rates = [0.4, 0.5, 1.2, 1.8]

    # TODO: Select number of steps (t) to train
    T = [8000, 8000, 20000, 20000]

    # TODO: Select beta for momentum (do not replace None)
    betas = [None, 0.05, None, 0.5]

    # TODO: Select batch sizes for stochastic and momentum stochastic gradient descent (do not replace None)
    batch_sizes = [None, None, 300, 300]

    # TODO: Convert dataset (N x d) to correct shape (d x N)
    x_train = np.transpose(x_train, axes=(1, 0))
    x_val = np.transpose(x_val, axes=(1, 0))
    x_test = np.transpose(x_test, axes=(1, 0))

    print('Results on using Ridge Regression using gradient descent variants')

    hyper_parameters =         zip(optimizer_types, learning_rates, T, betas, batch_sizes)
    
    for optimizer_type, learning_rate, t, beta, batch_size in hyper_parameters:

        # Conditions on batch size and beta
        if batch_size is not None:
            assert batch_size <= 0.90 * x_train.shape[1]

        if beta is not None:
            assert beta >= 0.05

        # TODO: Initialize ridge regression trained with gradient descent variants
        ridge_grad_descent = RidgeRegressionGradientDescent()

        print('Fitting with {} using learning rate={:.1E}, t={}'.format(
            optimizer_type, learning_rate, t))

        # TODO: Train ridge regression using gradient descent variants
        ridge_grad_descent.fit(
            x=x_train,
            y=y_train,
            optimizer_type=optimizer_type,
            learning_rate=learning_rate,
            t=t,
            lambda_weight_decay=lambda_weight_decay,
            beta=beta,
            batch_size=batch_size)

        # TODO: Test model on training set
        score_mse_grad_descent_train = score_mean_squared_error(ridge_grad_descent, x_train, y_train)
        print('Training set mean squared error: {:.4f}'.format(score_mse_grad_descent_train))

        # TODO: Test model on validation set
        score_mse_grad_descent_val = score_mean_squared_error(ridge_grad_descent, x_val, y_val)
        print('Validation set mean squared error: {:.4f}'.format(score_mse_grad_descent_val))

        # TODO: Test model on testing set
        score_mse_grad_descent_test = score_mean_squared_error(ridge_grad_descent, x_test, y_test)
        print('Testing set mean squared error: {:.4f}'.format(score_mse_grad_descent_test))

        
