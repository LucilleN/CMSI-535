import numpy as np
import sklearn.datasets as skdata
import sklearn.metrics as skmetrics
import sklearn.preprocessing as skpreprocess
from sklearn.linear_model import Ridge as RidgeRegression


'''
Name: Njoo, Lucille

Collaborators: N/A

Collaboration details: N/A

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
Fitting with gradient_descent using learning rate=1.0E-01, t=10
Training set mean squared error: 5706.2191
Validation set mean squared error: 5539.8455
Testing set mean squared error: 7371.7854
Fitting with momentum_gradient_descent using learning rate=1.0E-01, t=10
Training set mean squared error: 5706.3651
Validation set mean squared error: 5541.1250
Testing set mean squared error: 7366.5416
Fitting with stochastic_gradient_descent using learning rate=1.0E-01, t=10
Training set mean squared error: 25841.8291
Validation set mean squared error: 24473.3987
Testing set mean squared error: 32345.2797
Fitting with momentum_stochastic_gradient_descent using learning rate=1.0E-01, t=10
Training set mean squared error: 17087.4906
Validation set mean squared error: 17860.2085
Testing set mean squared error: 15073.6718
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

        # Implement the __compute_gradients function

        # Add bias to x so that its shape is (d, N) -> (d + 1, N)
        x = np.concatenate([np.ones([1, x.shape[1]]), x], axis=0)

        # Gradients for all samples will be (d + 1, N)
        gradients = np.zeros(x.shape)

        # MSE loss for Ridge Regression:
        #   f(w) = 1/N || Xw - y ||^2_2 + lambda/N ||w||^2_2
        #        = 1/N sum_n^N (w^T x^n - y^n)^2 + lambda/N (w^T w)
        # Derivative of MSE loss for Ridge Regression:
        #   f'(w) = 1/N sum_n^N 2 * (w^T x^n - y^n) \nabla (w^T x^n - y^n) + (2 * lambda/N * w)
        #         = 1/N sum_n^N 2 * (w^T x^n - y^n) x^n + (2 * lambda/N * w)

        N = x.shape[1]
        # Compute summation part
        for n in range(N):
            # x_n : (d + 1 , 1)
            x_n = np.expand_dims(x[:, n], axis=1)

            # w.T (d + 1, 1)^T  *  x_n (d + 1, 1)
            prediction = np.matmul(w.T, x_n)
            gradient = 2 * (prediction - y[n]) * x_n
            gradients[:, n] = np.squeeze(gradient)

         # Retain the last dimension so that we have (d + 1, 1)
        gradient_with_mse_only = np.mean(gradients, axis=1, keepdims=True)

        gradient_with_lambda = gradient_with_mse_only + 2 * lambda_weight_decay / N * w

        return gradient_with_lambda

    def __cube_root_decay(self, time_step):
        '''
        Computes the cube root polynomial decay factor t^{-1/3}

        Args:
            time_step : int
                current step in optimization

        Returns:
            float : cube root decay factor to adjust learning rate
        '''

        # Implement cube root polynomial decay factor to adjust learning rate
        return t ** (-1.0 / 3.0)

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
            beta : float
                momentum discount rate
            batch_size : int
                batch size for stochastic and momentum stochastic gradient descent
            time_step : int
                current step in optimization

        Returns:
            numpy : 1 x d weights
        '''

        # Implement the optimizer update function

        if self.__momentum is None:
            self.__momentum = np.zeros_like(w)

        alpha = self.__learning_rate

        if optimizer_type == 'gradient_descent':
            
            # w^(t + 1) = w^(t) - alpha \nabla loss(w^(t))
            #           = w - alpha * self.__compute_gradients(w, x, y, lambda_weight_decay)

            # Compute gradients
            gradients = self.__compute_gradients(w, x, y, lambda_weight_decay)

            # Update weights
            w = w - alpha * gradients
            
            return w

        elif optimizer_type == 'momentum_gradient_descent':
            
            # v^(t) = beta * v^(t-1) + (1-beta) \nabla loss(w^(t))
            #       = beta * v^(t-1) + (1-beta) gradients
            # w^(t + 1) = w^(t) - alpha * v^(t)

            # Compute gradients
            gradients = self.__compute_gradients(w, x, y, lambda_weight_decay)

            # Compute momentum
            self.__momentum = beta * self.__momentum + (1-beta) * gradients

            # Update weights
            w = w - alpha * self.__momentum
            
            return w

        elif optimizer_type == 'stochastic_gradient_descent':

            # Implement stochastic gradient descent
            
            # Sample batch from dataset
            N = x.shape[1]
            batch_indexes = np.random.choice(range(0, N), batch_size)
            x_batch = x[:, batch_indexes]
            y_batch = y[batch_indexes]

            # Compute gradients
            gradients = self.__compute_gradients(w, x_batch, y_batch, lambda_weight_decay)

            # Compute cube root decay factor and multiply by learning rate
            decay_factor = self.__cube_root_decay(time_step)

            # Update weights
            w = w - decay_factor * gradients
            
            return w

        elif optimizer_type == 'momentum_stochastic_gradient_descent':

            # v^(t) = beta * v^(t-1) + (1-beta) \nabla loss(w^(t))
            #       = beta * v^(t-1) + (1-beta) gradients
            # w^(t + 1) = w^(t) - alpha * v^(t)
            # Same as before, but loss/gradients are computed for a batch rather than all N samples

            # Implement momentum stochastic gradient descent

            # Sample batch from dataset
            N = x.shape[1]
            batch_indexes = np.random.choice(range(0, N), batch_size)
            x_batch = x[:, batch_indexes]
            y_batch = y[batch_indexes]

            # Compute gradients
            gradients = self.__compute_gradients(w, x_batch, y_batch, lambda_weight_decay)

            # Compute momentum
            self.__momentum = beta * self.__momentum + (1-beta) * gradients

            # Compute cube root decay factor and multiply by learning rate
            decay_factor = self.__cube_root_decay(time_step)

            # Update weights
            w = w - decay_factor * gradients
            
            return w


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

        # Implement the fit function

        # Initialize weights (d + 1, 1)
        self.__weights = np.zeros([x.shape[0] + 1, 1])
        self.__weights[0] = 1.0

        # Initialize optimizer
        self.__optimizer = GradientDescentOptimizer(learning_rate)

        for time_step in range(1, t + 1):

            # Compute loss function
            loss, loss_data_fidelity, loss_regularization = self.__compute_loss(x, y, lambda_weight_decay)

            if (time_step % 500) == 0:
                print("time step: {}".format(time_step))
                print("loss: {}".format(loss))
                print("loss_data_fidelity: {}".format(loss_data_fidelity))
                print("loss_regularization: {}".format(loss_regularization))
                print('Step={:5}  Loss={:.4f}  Data Fidelity={:.4f}  Regularization={:.4f}'.format(
                    time_step, loss, loss_data_fidelity, loss_regularization))

            # Update weights
            self.__weights = self.__optimizer.update(
                self.__weights, 
                x, 
                y, 
                optimizer_type, 
                lambda_weight_decay, 
                beta, 
                batch_size, 
                time_step
            )

    def predict(self, x):
        '''
        Predicts the label for each feature vector x

        Args:
            x : numpy
                d x N feature vector

        Returns:
            numpy : N element vector
        '''

        # Implements the predict function

        N = x.shape[1]

        # Add bias to x: (d, N) -> (d + 1, N)
        x = np.concatenate([np.ones([1, N]), x], axis=0)

        # predictions should be of shape (N, )
        predictions = np.zeros(N)

        for n in range(N):
            # x_n : (d + 1, 1)
            x_n = np.expand_dims(x[:, n], axis=1)

            # y_hat or h_x = w^T x
            # w^T (d + 1, 1)^T \times x_n (d + 1, 1)
            prediction = np.matmul(self.__weights.T, x_n)
            predictions[n] = prediction

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

        # Implement the __compute_loss function
        
        N = x.shape[1]

        # Add bias to x so that its shape is (d, N) -> (d + 1, N)
        x = np.concatenate([np.ones([1, N]), x], axis=0)

        # Ridge Regression loss function:
        # f(w) = 1/N || Xw - y ||^2_2  +  lambda / N || w ||^2_2
        #      = 1/N sum_n^N (w^T x^n - y^n)^2  +  lambda / N (w^T w)

        data_fidelity_losses = []

        for n in range(N):
            # x_n : (d + 1 , 1)
            x_n = np.expand_dims(x[:, n], axis=1)

            prediction = np.matmul(self.__weights.T, x_n)
            current_data_fidelity_loss = (prediction - y[n]) ** 2
            data_fidelity_losses = np.append(data_fidelity_losses, current_data_fidelity_loss)

        loss_data_fidelity = np.mean(data_fidelity_losses)
        loss_regularization = lambda_weight_decay / N * np.matmul(self.__weights.T, self.__weights)[0][0] # todo this is messy
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
    learning_rates = [0.1, 0.1, 0.1, 0.1]

    # TODO: Select number of steps (t) to train
    T = [8000, 5000, 20000, 15000]

    # TODO: Select beta for momentum (do not replace None)
    betas = [None, 0.05, None, 0.05]

    # TODO: Select batch sizes for stochastic and momentum stochastic gradient descent (do not replace None)
    batch_sizes = [None, None, 1, 1]

    # TODO: Convert dataset (N x d) to correct shape (d x N)
    x_train = np.transpose(x_train, axes=(1, 0))
    x_val = np.transpose(x_val, axes=(1, 0))
    x_test = np.transpose(x_test, axes=(1, 0))

    print('Results on using Ridge Regression using gradient descent variants')

    hyper_parameters = \
        zip(optimizer_types, learning_rates, T, betas, batch_sizes)

    for optimizer_type, learning_rate, t, beta, batch_size in hyper_parameters:

        # Conditions on batch size and beta
        if batch_size is not None:
            assert batch_size <= 0.90 * x_train.shape[1]

        if beta is not None:
            assert beta >= 0.05

        # Initialize ridge regression trained with gradient descent variants
        ridge_grad_descent_model = RidgeRegressionGradientDescent()

        print('Fitting with {} using learning rate={:.1E}, t={}'.format(
            optimizer_type, learning_rate, t))

        # Train ridge regression using gradient descent variants
        ridge_grad_descent_model.fit(
            x=x_train, 
            y=y_train, 
            optimizer_type=optimizer_type, 
            learning_rate=learning_rate, 
            t=t, 
            lambda_weight_decay=lambda_weight_decay, 
            beta=beta, 
            batch_size=batch_size
        )

        # Test model on training set
        score_mse_grad_descent_train = score_mean_squared_error(ridge_grad_descent_model, x_train, y_train)
        print('Training set mean squared error: {:.4f}'.format(score_mse_grad_descent_train))

        # Test model on validation set
        score_mse_grad_descent_val = score_mean_squared_error(ridge_grad_descent_model, x_val, y_val)
        print('Validation set mean squared error: {:.4f}'.format(score_mse_grad_descent_val))

        # Test model on testing set
        score_mse_grad_descent_test = score_mean_squared_error(ridge_grad_descent_model, x_test, y_test)
        print('Testing set mean squared error: {:.4f}'.format(score_mse_grad_descent_test))
