import numpy as np
import sklearn.datasets as skdata
import sklearn.metrics as skmetrics
from sklearn.linear_model import LinearRegression


'''
Name: Doe, John (Please write names in <Last Name, First Name> format)

Collaborators: Doe, Jane (Please write names in <Last Name, First Name> format)

Collaboration details: Discussed <function name> implementation details with Jane Doe.

Summary:
Report your scores here.

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
Implementation of our Gradient Descent optimizer for mean squared loss and logistic loss
'''
class GradientDescentOptimizer(object):

    def __init__(self):
        pass

    def __compute_gradients(self, w, x, y, loss_func):
        '''
        Returns the gradient of the logistic, mean squared or half mean squared loss

        Args:
            w : numpy
                d x 1 weight vector
            x : numpy
                d x N feature vector
            y : numpy
                N element groundtruth vector
            loss_func : str
                loss type either mean_squared', or 'half_mean_squared'

        Returns:
            numpy : 1 x d gradients
        '''

        # TODO: Implements the __compute_gradients function

        if loss_func == 'mean_squared':
            # TODO: Implements gradients for mean squared loss

            return np.zeros_like(w)
        elif loss_func == 'half_mean_squared':
            # TODO: Implements gradients for half mean squared loss

            return np.zeros_like(w)
        else:
            raise ValueError('Supported losses: mean_squared, or half_mean_squared')

    def update(self, w, x, y, alpha, loss_func):
        '''
        Updates the weight vector based on logistic, mean squared or half mean squared loss

        Args:
            w : numpy
                1 x d weight vector
            x : numpy
                d x N feature vector
            y : numpy
                N element groundtruth vector
            alpha : numpy
                learning rate
            loss_func : str
                loss type either 'mean_squared', or 'half_mean_squared'

        Returns:
            numpy : 1 x d weights
        '''

        # TODO: Implement the optimizer update function

        return np.zeros_like(w)


'''
Implementation of our Linear Regression model trained using Gradient Descent
'''
class LinearRegressionGradientDescent(object):

    def __init__(self):
        # Define private variables
        self.__weights = None
        self.__optimizer = GradientDescentOptimizer()

    def fit(self, x, y, t, alpha, loss_func='mean_squared'):
        '''
        Fits the model to x and y by updating the weight vector
        using gradient descent

        Args:
            x : numpy
                N x d feature vector
            y : numpy
                N element groundtruth vector
            t : numpy
                number of iterations to train
            alpha : numpy
                learning rate
            loss_func : str
                loss function to use
        '''

        # TODO: Implement the fit function

        for i in range(1, t + 1):

            # TODO: Compute loss function
            loss = 0.0

            if (i % 500) == 0:
                print('Step={}  Loss={:.4f}'.format(i, loss))

            # TODO: Update weights

    def predict(self, x):
        '''
        Predicts the label for each feature vector x

        Args:
            x : numpy
                N x d feature vector

        Returns:
            numpy : N x 1 vector
        '''

        # TODO: Implements the predict function

        return np.zeros([x.shape[1]])

    def __compute_loss(self, x, y, loss_func):
        '''
        Returns the gradient of the logistic, mean squared or half mean squared loss

        Args:
            x : numpy
                N x d feature vector
            y : numpy
                N element groundtruth vector
            loss_func : str
                loss type either mean_squared', or 'half_mean_squared'

        Returns:
            float : loss
        '''

        # TODO: Implements the __compute_loss function

        if loss_func == 'mean_squared':
            # TODO: Implements loss for mean squared loss
            pass
        elif loss_func == 'half_mean_squared':
            # TODO: Implements loss for half mean squared loss
            pass
        else:
            raise ValueError('Supported losses: mean_squared, or half_mean_squared')

        return 0.0

if __name__ == '__main__':

    # Loads diabetes data with 80% training, 10% validation, 10% testing split
    diabetes_data = skdata.load_diabetes()
    x = diabetes_data.data
    y = diabetes_data.target

    split_idx = int(0.90 * x.shape[0])

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

    '''
    Trains and tests Linear Regression model from scikit-learn
    '''

    # Trains scikit-learn Linear Regression model on diabetes data
    linear_scikit = LinearRegression()
    linear_scikit.fit(x_train, y_train)

    print('Results on diabetes dataset using scikit-learn Linear Regression model')

    # Test model on training set
    scores_mse_train_scikit = score_mean_squared_error(linear_scikit, x_train, y_train)
    print('Training set mean accuracy: {:.4f}'.format(scores_mse_train_scikit))

    # Test model on validation set
    scores_mse_val_scikit = score_mean_squared_error(linear_scikit, x_val, y_val)
    print('Validation set mean accuracy: {:.4f}'.format(scores_mse_val_scikit))

    # Test model on testing set
    scores_mse_test_scikit = score_mean_squared_error(linear_scikit, x_test, y_test)
    print('Testing set mean accuracy: {:.4f}'.format(scores_mse_test_scikit))

    '''
    Trains and tests our Linear Regression model trained using Gradient Descent
    '''

    # Loss functions to minimize
    loss_funcs = ['mean_squared', 'half_mean_squared']

    # TODO: Select learning rates (alpha) for mean squared and half mean squared loss
    alphas = [0.0, 0.0]

    # TODO: Select number of steps (t) to train for mean squared and half mean squared loss
    T = [0, 0]

    # TODO: Convert dataset (N x d) to correct shape (d x N)

    print('Results on diabetes dataset using Linear Regression trained with gradient descent'.format())

    for loss_func, alpha, t in zip(loss_funcs, alphas, T):

        # TODO: Initialize linear regression trained with gradient descent

        print('Fitting with learning rate (alpha)={:.1E},  t={}'.format(alpha, t))

        # TODO: Train linear regression using gradient descent

        # TODO: Test model on training set
        score_mse_grad_descent_train = 0.0
        print('Training set mean accuracy: {:.4f}'.format(score_mse_grad_descent_train))

        # TODO: Test model on validation set
        score_mse_grad_descent_val = 0.0
        print('Validation set mean accuracy: {:.4f}'.format(score_mse_grad_descent_val))

        # TODO: Test model on testing set
        score_mse_grad_descent_test = 0.0
        print('Testing set mean accuracy: {:.4f}'.format(score_mse_grad_descent_test))
