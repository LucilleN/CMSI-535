import numpy as np
import sklearn.datasets as skdata
import sklearn.metrics as skmetrics
from sklearn.linear_model import LinearRegression


'''
Name: Njoo, Lucille

Collaborators: N/A

Collaboration details: N/A

Summary:

In this exercise, we implemented the Gradient Descent algorithm for optimization, and 
used it to train our own LinearRegressionGradientDescent model. We then compared the 
performance of the scikit-learn LinearRegression model, which uses the closed-form 
solution to find the optimal weights, to our own implementation of the 
LinearRegressionGradientDescent model, and trained our own model on different 
combinations of hyperparameters T (timesteps), alpha (learning rate), and loss function 
(MSE or half-MSE). The result is that our model performs approximately the same (though 
a little bit worse) than the scikit-learn model. Also, notably, when we use the 
half-MSE loss function, the results are exactly the same as using half the learning 
rate with the regular MSE loss function. This makes sense because the weights are 
updated using the equation: 
    w^(t + 1) = w^(t) - alpha * (gradient of loss)
And so when alpha is halved but the gradient is doubled, the result is exactly the same.

Report your scores here.

Results on diabetes dataset using scikit-learn Linear Regression model
Training set mean accuracy: 2750.0952
Validation set mean accuracy: 3665.1559
Testing set mean accuracy: 3114.5817
Results on diabetes dataset using Linear Regression trained with gradient descent
Fitting with learning rate (alpha)=5.0E-01,  t=2000
Step=500  Loss=2897.2545
Step=1000  Loss=2783.0061
Step=1500  Loss=2766.2741
Step=2000  Loss=2762.8744
Training set mean accuracy: 2762.8712
Validation set mean accuracy: 3741.2679
Testing set mean accuracy: 3113.1787
Fitting with learning rate (alpha)=1.0E+00,  t=4000
Step=500  Loss=1448.6273
Step=1000  Loss=1391.5030
Step=1500  Loss=1383.1371
Step=2000  Loss=1381.4372
Step=2500  Loss=1380.9459
Step=3000  Loss=1380.7327
Step=3500  Loss=1380.5947
Step=4000  Loss=1380.4796
Training set mean accuracy: 2760.9587
Validation set mean accuracy: 3763.9527
Testing set mean accuracy: 3111.1478

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
    predictions = np.squeeze(model.predict(x))
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
        Returns the gradient of the mean squared or half mean squared loss

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

        # Add bias to x so that its shape is (d, N) -> (d + 1, N)
        x = np.concatenate([np.ones([1, x.shape[1]]), x], axis=0)

        # Gradients for all samples will be (d + 1, N)
        gradients = np.zeros(x.shape)

        # Regular MSE loss function:
        #   f(w) = 1/N || Xw - y ||^2_2 
        #        = 1/N sum_n^N ^w^T x^n - y^n)^2

        # Derivative of MSE loss function:
        #   f'(w) = 1/N sum_n^N 2 * (w^T x^n - y^n) \nabla (w^T x^n - y^n)
        #         = 1/N sum_n^N 2 * (w^T x^n - y^n) x^n

        for n in range(x.shape[1]):
            # x_n : (d + 1 , 1)
            x_n = np.expand_dims(x[:, n], axis=1)

            # w.T (d + 1, 1)^T \times x_n (d + 1, 1)
            prediction = np.matmul(w.T, x_n)
            gradient = 2 * (prediction - y[n]) * x_n
            gradients[:, n] = np.squeeze(gradient)

            # The same thing as the lines above, but in a one-liner
            # gradients[:, n] = 2 * np.squeeze((np.matmul(w.T, x_n) - y[n]) * x_n)

        # Retain the last dimension so that we have (d + 1, 1)
        gradient_with_mse_loss = np.mean(gradients, axis=1, keepdims=True)

        if loss_func == 'mean_squared':
            return gradient_with_mse_loss

        elif loss_func == 'half_mean_squared':
            # Half Mean Squared Error is just Mean Squared Error divided by 2
            # so that the derivative is not multiplied by that constant 2

            # Half MSE loss function: 
            #   f(w) = 1/2N || Xw - y ||^2_2 
            #        = 1/2N sum_n^N (w^T x^n - y^n)^2

            # Derivative of MSE loss function:
            #   f'(w) = 1/2N sum_n^N 2 * (w^T x^n - y^n) \nabla (w^T x^n - y^n)
            #         = 1/2N sum_n^N 2 * (w^T x^n - y^n) x^n
            return gradient_with_mse_loss / 2

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
        #   w^(t + 1) = w^(t) - \alpha \nabla \ell(w^(t))
        # Which is the same as: 
        #   w = w - alpha * self.__compute_gradients(w, x, y, loss_func)
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

        # Initialize the weights (d + 1, 1)
        self.__weights = np.zeros([x.shape[0] + 1, 1])
        self.__weights[0] = 1.0

        for i in range(1, t + 1):

            loss = self.__compute_loss(x, y, loss_func)

            if (i % 500) == 0:
                print('Step={}  Loss={:.4f}'.format(i, loss))

            # Update weights
            w_i = self.__optimizer.update(
                self.__weights, x, y, alpha, loss_func=loss_func)

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

        # Add bias to x (d, N) -> (d + 1, N)
        x = np.concatenate([np.ones([1, x.shape[1]]), x], axis=0)

        # predictions should be (1 , N)
        predictions = np.zeros([1, x.shape[1]])

        for n in range(x.shape[1]):
            # x_n : (d + 1, 1)
            x_n = np.expand_dims(x[:, n], axis=1)

            # y_hat or h_x = w^T x
            # w^T (d + 1, 1)^T \times x_n (d + 1, 1)
            prediction = np.matmul(self.__weights.T, x_n)
            predictions[:, n] = np.squeeze(prediction)

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
        N = x.shape[1]

        # Add bias to x so that its shape is (d, N) -> (d + 1, N)
        x = np.concatenate([np.ones([1, x.shape[1]]), x], axis=0)

        # Regular MSE loss function:
        # f(w) = 1/N || Xw - y ||^2_2 
        #      = 1/N sum_n^N (w^T x^n - y^n)^2

        losses = []

        for n in range(N):
            # x_n : (d + 1 , 1)
            x_n = np.expand_dims(x[:, n], axis=1)

            prediction = np.matmul(self.__weights.T, x_n)
            loss = (prediction - y[n]) ** 2
            losses = np.append(losses, loss)

        mse_loss = np.mean(losses)

        if loss_func == 'mean_squared':
            return mse_loss
        elif loss_func == 'half_mean_squared':
            return mse_loss / 2
        else:
            raise ValueError('Supported losses: mean_squared, or half_mean_squared')
        

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

    # Select learning rates (alpha) for mean squared and half mean squared loss
    alphas = [0.5, 1.0]

    # Select number of steps (t) to train for mean squared and half mean squared loss
    T = [2000,4000]

    # Convert dataset (N x d) to correct shape (d x N)
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
            loss_func=loss_func)

        # Test model on training set
        score_mse_grad_descent_train = score_mean_squared_error(linear_grad_descent, x_train, y_train)
        print('Training set mean accuracy: {:.4f}'.format(score_mse_grad_descent_train))

        # Test model on validation set
        score_mse_grad_descent_val = score_mean_squared_error(linear_grad_descent, x_val, y_val)
        print('Validation set mean accuracy: {:.4f}'.format(score_mse_grad_descent_val))

        # Test model on testing set
        score_mse_grad_descent_test = score_mean_squared_error(linear_grad_descent, x_test, y_test)
        print('Testing set mean accuracy: {:.4f}'.format(score_mse_grad_descent_test))
