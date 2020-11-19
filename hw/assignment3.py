import numpy as np
import sklearn.datasets as skdata
import sklearn.metrics as skmetrics
import sklearn.preprocessing as skpreprocess
from sklearn.linear_model import Ridge as RidgeRegression


'''
Name: Doe, John (Please write names in <Last Name, First Name> format)

Collaborators: Doe, Jane (Please write names in <Last Name, First Name> format)

Collaboration details: Discussed <function name> implementation details with Jane Doe.

Summary:

TODO: Please answer the following questions and report your scores

1. What did you observe when using larger versus smaller momentum for
momentum gradient descent and momentum stochastic gradient descent?

2. What did you observe when using larger versus smaller batch size
for stochastic gradient descent?

3. Explain the difference between gradient descent, momentum gradient descent,
stochastic gradient descent, and momentum stochastic gradient descent?

Report your scores here.

Results on using scikit-learn Ridge Regression model
Training set mean squared error: 2749.2155
Validation set mean squared error: 3722.5782
Testing set mean squared error: 3169.6860
Results on using Ridge Regression using gradient descent variants
Fitting with gradient_descent using learning rate=0.0E+00,  t=1
Training set mean squared error: 0.0000
Validation set mean squared error: 0.0000
Testing set mean squared error: 0.0000
Fitting with momentum_gradient_descent using learning rate=0.0E+00,  t=1
Training set mean squared error: 0.0000
Validation set mean squared error: 0.0000
Testing set mean squared error: 0.0000
Fitting with stochastic_gradient_descent using learning rate=0.0E+00,  t=1
Training set mean squared error: 0.0000
Validation set mean squared error: 0.0000
Testing set mean squared error: 0.0000
Fitting with momentum_stochastic_gradient_descent using learning rate=0.0E+00,  t=1
Training set mean squared error: 0.0000
Validation set mean squared error: 0.0000
Testing set mean squared error: 0.0000
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

        return np.zeros_like(w)

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

        return 0.0

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

            # TODO: Update weights
            return np.zeros_like(w)

        elif optimizer_type == 'momentum_gradient_descent':

            # TODO: Compute gradients

            # TODO: Compute momentum

            # TODO: Update weights
            return np.zeros_like(w)

        elif optimizer_type == 'stochastic_gradient_descent':

            # TODO: Implement stochastic gradient descent

            # TODO: Sample batch from dataset

            # TODO: Compute gradients

            # TODO: Compute cube root decay factor and multiply by learning rate

            # TODO: Update weights
            return np.zeros_like(w)

        elif optimizer_type == 'momentum_stochastic_gradient_descent':

            # TODO: Implement momentum stochastic gradient descent

            # TODO: Sample batch from dataset

            # TODO: Compute gradients

            # TODO: Compute momentum

            # TODO: Compute cube root decay factor and multiply by learning rate

            # TODO: Update weights
            return np.zeros_like(w)


'''
Implementation of our Ridge Regression model trained using gradient descent variants
'''
class RidgeRegressionGradientDescent(object):

    def __init__(self):
        # Define private variables
        self.__weights = None
        self.__optimizer = None

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

        # TODO: Initialize optimizer

        for time_step in range(1, t + 1):

            # TODO: Compute loss function
            loss, loss_data_fidelity, loss_regularization = 0.0, 0.0, 0.0

            if (time_step % 500) == 0:
                print('Step={:5}  Loss={:.4f}  Data Fidelity={:.4f}  Regularization={:.4f}'.format(
                    time_step, loss, loss_data_fidelity, loss_regularization))

            # TODO: Update weights

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

        return np.zeros(x.shape[1])

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

        loss_data_fidelity = 0.0
        loss_regularization = 0.0
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

    x_train, x_val, x_test = \
        x[train_idx, :], x[val_idx, :], x[test_idx, :]
    y_train, y_val, y_test = \
        y[train_idx], y[val_idx], y[test_idx]

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
    learning_rates = [0.0, 0.0, 0.0, 0.0]

    # TODO: Select number of steps (t) to train
    T = [1, 1, 1, 1]

    # TODO: Select beta for momentum (do not replace None)
    betas = [None, 0.05, None, 0.05]

    # TODO: Select batch sizes for stochastic and momentum stochastic gradient descent (do not replace None)
    batch_sizes = [None, None, 1, 1]

    # TODO: Convert dataset (N x d) to correct shape (d x N)

    print('Results on using Ridge Regression using gradient descent variants')

    hyper_parameters = \
        zip(optimizer_types, learning_rates, T, betas, batch_sizes)

    for optimizer_type, learning_rate, t, beta, batch_size in hyper_parameters:

        # Conditions on batch size and beta
        if batch_size is not None:
            assert batch_size <= 0.90 * x_train.shape[1]

        if beta is not None:
            assert beta >= 0.05

        # TODO: Initialize ridge regression trained with gradient descent variants

        print('Fitting with {} using learning rate={:.1E}, t={}'.format(
            optimizer_type, learning_rate, t))

        # TODO: Train ridge regression using gradient descent variants

        # TODO: Test model on training set
        score_mse_grad_descent_train = 0.0
        print('Training set mean squared error: {:.4f}'.format(score_mse_grad_descent_train))

        # TODO: Test model on validation set
        score_mse_grad_descent_val = 0.0
        print('Validation set mean squared error: {:.4f}'.format(score_mse_grad_descent_val))

        # TODO: Test model on testing set
        score_mse_grad_descent_test = 0.0
        print('Testing set mean squared error: {:.4f}'.format(score_mse_grad_descent_test))
