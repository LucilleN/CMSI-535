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
                loss type either 'mean_squared', or 'half_mean_squared'

        Returns:
            numpy : 1 x d gradients
        '''
        N = x.shape[1]
        # TODO: Implements the __compute_gradients function

        # Add bias to x 
        # (1 x N) + (d x N) -> (d+1 x N)
        x = np.concatenate((np.ones((1, N)), x), axis=0)

        # gradients for all samples is the size of x, (d+1 x N)
        gradients = np.zeros(x.shape) 

        if loss_func == 'mean_squared':

            # MSE: f(w) = 1\N || Xw - y ||_2^2 = 1/N \sum_n^N (w^T x_n - y_n)^2
            # f'(w) = 1/N sum_n^N 2 * (w^T x - y_n) \nabla (w^T x_n - y)
            #       = 1/N sum_n^N 2 * (w^T x - y_n) * x_n
            # use this function to compute gradients for MSE loss function
            for n in range(N):
                x_n = x[:, n]
                prediction = np.matmul(w.T, x_n)
                gradient_n = 2 * (prediction - y[n]) * x_n
                gradients[:, n] = gradient_n

                # one-liner:
                # gradients[:, n] = 2 * (np.matmul(w.T, x_n) - y[n]) * x_n

            # Accumulated N of these gradients; we want average of all of them
            # Summing and dividing by N is same as taking the mean
            return np.mean(gradients, axis=1)

        elif loss_func == 'half_mean_squared':
            # TODO: Implements gradients for half mean squared loss

            return np.zeros_like(w)
        else:
            raise ValueError('Supported losses: mean_squared, or half_mean_squared')

    def update(self, w, x, y, alpha, loss_func):
        '''
        Updates the weight vector based on mean squared or half mean squared loss

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

        # Computes the gradients for a given loss function
        gradients = self.__compute_gradients(w, x, y, loss_func)

        # Update our weights using gradient descent
        # w^(t+1) = w^(t) - \alpha * \nabla \l(w^(t))
        #         = w^(t) - alpha * gradients
        w = w - alpha * gradients

        return w


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
                d x N feature vector
            y : numpy
                N element groundtruth vector
            t : numpy
                number of iterations to train
            alpha : numpy
                learning rate
            loss_func : str
                loss function to use
        '''
        d = x.shape[0]

        # TODO: Implement the fit function

        # Initialize weights (d+1 x 1)
        self.__weights = np.zeros((d+1, 1))
        self.__weights[0] = 1.0

        for i in range(1, t + 1):

            # TODO: Compute loss function
            loss = 0.0

            if (i % 500) == 0:
                print('Step={}  Loss={:.4f}'.format(i, loss))

            # Update weights
            w_i = self.__optimizer.update(self.__weights, x, y, alpha, loss_func)
            self.__weights = w_i

    def predict(self, x):
        '''
        Predicts the label for each feature vector x

        Args:
            x : numpy
                d x N feature vector

        Returns:
            numpy : 1 x N vector
        '''
        N = x.shape[1]

        # Add bias to x; (d x N) -> (d+1 x N)
        x = np.concatenate((np.ones((1, N)), x), axis=0)

        predictions = np.zeros((1, N))

        for n in range(N):
            x_n = x[:, n]
            # y_hat or h_x = w^T x
            # prediction = np.matmul(self.__weights.T, x_n)
            prediction = np.dot(self.__weights.T, x_n)
            predictions[:, n] = prediction

        return predictions

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
    alphas = [1.0, 1.0]

    # TODO: Select number of steps (t) to train for mean squared and half mean squared loss
    T = [100, 100]

    # Convert dataset (N x d) to correct shape (d x N) for train, val, and test
    x_train = np.transpose(x_train, axes=(1, 0))
    x_val = np.transpose(x_val, axes=(1, 0))
    x_test = np.transpose(x_test, axes=(1, 0))

    print('Results on diabetes dataset using Linear Regression trained with gradient descent'.format())

    for loss_func, alpha, t in zip(loss_funcs, alphas, T):

        # Initialize linear regression trained with gradient descent
        linear_grad_descent = LinearRegressionGradientDescent()

        print('Fitting with learning rate (alpha)={:.1E},  t={}'.format(alpha, t))

        # Train linear regression using gradient descent
        linear_grad_descent.fit(
            x=x_train,
            y=y_train,
            t=t,
            alpha=alpha, 
            loss_func=loss_func
        )

        # TODO: Test model on training set
        score_mse_grad_descent_train = score_mean_squared_error(linear_grad_descent, x_train, y_train)
        print('Training set mean accuracy: {:.4f}'.format(score_mse_grad_descent_train))

        # TODO: Test model on validation set
        score_mse_grad_descent_val = 0.0
        print('Validation set mean accuracy: {:.4f}'.format(score_mse_grad_descent_val))

        # TODO: Test model on testing set
        score_mse_grad_descent_test = 0.0
        print('Testing set mean accuracy: {:.4f}'.format(score_mse_grad_descent_test))
