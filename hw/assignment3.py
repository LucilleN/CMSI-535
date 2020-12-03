import numpy as np
import sklearn.datasets as skdata
import sklearn.metrics as skmetrics
import sklearn.preprocessing as skpreprocess
from sklearn.linear_model import Ridge as RidgeRegression
from matplotlib import pyplot as plt


'''
Name: Njoo, Lucille

Collaborators: N/A

Collaboration details: N/A

Summary:

    In this assignment, we implemented a GradientDescentOptimizer for Ridge Regression 
    that contains methods for updating weights using either regular gradient descent, 
    momentum gradient descent, stochastic gradient descent, or momentum stochastic 
    gradient descent. We then implemented a RidgeRegressionGradientDescent class, with 
    fit and predict methods, that can be trained with a variety of hyperparameters: 
    optimizer type, learning rate (alpha), timesteps (t), weight decay (lambda), discount 
    factor (beta) for optimizers with momentum, and batch size (B) for stochastic 
    optimizers. The main method loads, splits, and performs polynomial expansion on the 
    sklearn Diabetes data, then trains and tests the sklearn Ridge Regression model for 
    comparison, and finally trains and tests our own implementation of Ridge Regression 
    with gradient descent using all four optimizer types, each with different learning rates, 
    timesteps, betas, and batch sizes. The results of our implementation are very close 
    to sklearn's results, with about 3200 testing set MSE loss.

Please answer the following questions and report your scores:

1. What did you observe when using larger versus smaller momentum for
momentum gradient descent and momentum stochastic gradient descent?

    With smaller momentums, momentum gradient descent and stochastic momentum gradient 
    descent are slower to converge (and for stochastic GD, sometimes do not converge at 
    all, depending on other hyperparameters) than with larger momentums. A momentum of 0 
    makes momentum GD and momentum stochastic GD both become simply regular GD and 
    stochastic GD, and stochastic GD especially causes a lot of fluctuations in loss. 
    With small or no momentums, momentum stochastic GD is thus much slower to converge 
    than momentum GD because the random sampling causes the loss to fluctuate 
    significantly, and sometimes it fluctuates so much that the function diverges. As we 
    increase momentum, the momentum stochastic GD converges much faster since using the 
    moving average rather than the gradient at a single timestep helps to keep the steps 
    moving in the right general direction, reducing the amount that the loss bounces 
    around. Thus, with larger momentums, momentum stochastic GD performs just as well as 
    non-stochastic momentum GD, and with small or no momentum, the momentum stochastic GD 
    (like stochastic GD) is slower to converge. Since momentum keeps the loss going in 
    the right general direction, we are able to use larger learning rates without 
    diverging. 

2. What did you observe when using larger versus smaller batch size
for stochastic gradient descent?

    I observed that larger batch sizes took a longer amount of time to train at each time 
    step, but made the stochastic gradient descent converge in fewer steps so that its 
    scores were able to end up similar to the regular gradient descent optimizer. Smaller 
    batch sizes made the timesteps run much more quickly, but they also made the loss 
    decrease very slowly and sometimes not at all, causing the loss to jump around 
    without converging. This makes sense because increasing the batch size makes the 
    algorithm more similar to the regular gradient descent that uses all N samples 
    without batching; if we increase the size of B all the way to N, then stochastic 
    gradient descent is just gradient descent. With more samples, each batch can be more 
    representative of the full dataset, and so we are less likely to diverge, and loss 
    decreases more at each time step. 

3. Explain the difference between gradient descent, momentum gradient descent,
stochastic gradient descent, and momentum stochastic gradient descent?

    Gradient descent is an iterative optimization algorithm that minimizes loss by 
    updating the weights with the negative of the gradient of the loss function at each 
    timestep, scaled by some scalar learning rate alpha. Thus, the weights are updated as 
    follows:
        w^(t + 1) = w^(t) - alpha * \nabla loss(w^(t))

    Momentum gradient descent follows the same process and concept, but rather than updating 
    the weights with the just gradient of the loss at the current time step (scaled by some 
    some learning rate again), we instead update the weights with the momentum, or the moving 
    average, of the losses; this way, instead of updating with just the current gradient, we 
    also add a fraction of the previous gradient(s) and thereby keeping the correct general 
    direction, which helps with convergence. We first compute the momentum at this time step, 
    which relies on the hyperparameter beta:
        v^(t) = beta * v^(t-1) + (1-beta) \nabla loss(w^(t))
    Then we use this momentum to compute the weights for the next time step:
        w^(t+1) = w^(t) - alpha * v^(t)
    Where alpha is the learning rate.

    Stochastic gradient descent is almost the same as gradient descent, but it approximates 
    the gradient of the loss at each time step rather than computing the exact loss on all 
    data samples. At each time step, rather than summing over all N data samples in the 
    training set, we randomly choose some batch (of a specified batch size) from the N 
    samples. We then compute the gradient of the loss on just that subset of data, and use 
    this to update the weights in the same way as in gradient descent. This is more 
    computationally efficient and helps to make problems with giant datasets more tractable. 
    At each timestep, we sample B items from the dataset D, and then update the weights as 
    follows:
        w^(t+1) = w^(t) - alpha * \nabla (loss for |B| samples)
    The loss on the batch B is computed the same way as it is on the whole dataset. Thus, we 
    can also write this as a sum of the losses for each sample in B:
        w^(t+1) = w^(t) - alpha * 1/|B| * \sum_b \nabla loss^b(w^(t))
    Where alpha is the learning rate.

    Momentum stochastic gradient descent combines momentum gradient descent and stochastic 
    gradient descent. Like in stochastic gradient descent, we sample a batch at each time step 
    and approximate the gradient of the loss by computing it only on the current batch. Then, 
    like in momentum gradient descent, rather than simply updating the weights with that 
    computed gradient for the current timestep, we use the moving average or momentum in order 
    to keep the loss moving in the correct general direction. Thus, we sample B items from the 
    dataset D, then compute the momentum using the gradient of the loss for just this batch: 
        v^(t) = beta * v^(t-1) + (1-beta) 1/|B| * \sum_b \nabla loss^b(w^(t))
    Then we use this momentum to compute the weights for the next time step:
        w^(t+1) = w^(t) - alpha * v^(t)
    Where alpha is the learning rate.

Report your scores here.
    Note: I didn't paste the loss printouts at every 500 steps because it was getting very 
    long and clunky.

Results on using scikit-learn Ridge Regression model
Training set mean squared error: 2749.2155
Validation set mean squared error: 3722.5782
Testing set mean squared error: 3169.6860
Results on using Ridge Regression using gradient descent variants
Fitting with gradient_descent using learning rate=1.5E-01, t=9000
Training set mean squared error: 2762.6766
Validation set mean squared error: 3732.5916
Testing set mean squared error: 3171.7095
Fitting with momentum_gradient_descent using learning rate=2.0E-01, t=7000
Training set mean squared error: 2762.1484
Validation set mean squared error: 3732.6714
Testing set mean squared error: 3171.5082
Fitting with stochastic_gradient_descent using learning rate=7.0E-02, t=24000
Training set mean squared error: 2779.5115
Validation set mean squared error: 3730.5139
Testing set mean squared error: 3198.7875
Fitting with momentum_stochastic_gradient_descent using learning rate=2.5E-01, t=20000
Training set mean squared error: 2789.0917
Validation set mean squared error: 3739.5281
Testing set mean squared error: 3190.7055
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
            # Same as gradient descent, but sample a batch first
            
            # Sample batch from dataset
            N = x.shape[1]
            batch_indexes = np.random.randint(low=0, high=N, size=batch_size)
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

            # v^(t) = beta * v^(t-1) + (1-beta) \nabla loss^B(w^(t))
            #       = beta * v^(t-1) + (1-beta) gradients_for_batch
            # w^(t + 1) = w^(t) - alpha * v^(t)
            # Same as before, but loss/gradients are computed for a batch rather than all N samples

            # Implement momentum stochastic gradient descent

            # Sample batch from dataset
            N = x.shape[1]
            batch_indexes = np.random.randint(low=0, high=N, size=batch_size)
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

        losses = []
        time_steps = []

        for time_step in range(1, t + 1):

            # Compute loss function
            loss, loss_data_fidelity, loss_regularization = self.__compute_loss(x, y, lambda_weight_decay)

            if (time_step % 500) == 0:
                print('Step={:5}  Loss={:.4f}  Data Fidelity={:.4f}  Regularization={:.4f}'.format(
                    time_step, loss, loss_data_fidelity, loss_regularization))
                losses.append(loss)
                time_steps.append(time_step)

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
        
        fig = plt.figure()
        fig.suptitle("Loss over timesteps for {}".format(optimizer_type))
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(time_steps, losses)

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
        loss_regularization = lambda_weight_decay / N * np.matmul(self.__weights.T, self.__weights).squeeze() 
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

    # Select learning rates for each optimizer
    learning_rates = [0.15, 0.2, 0.07, 0.25]

    # Select number of steps (t) to train
    T = [9000, 7000, 24000, 20000]

    # Select beta for momentum (do not replace None)
    betas = [None, 0.05, None, 0.15]

    # Select batch sizes for stochastic and momentum stochastic gradient descent (do not replace None)
    batch_sizes = [None, None, 300, 270]

    # Convert dataset (N x d) to correct shape (d x N)
    x_train = np.transpose(x_train, axes=(1, 0))
    x_val = np.transpose(x_val, axes=(1, 0))
    x_test = np.transpose(x_test, axes=(1, 0))

    print('Results on using Ridge Regression using gradient descent variants')

    hyper_parameters = \
        zip(optimizer_types, learning_rates, T, betas, batch_sizes)

    for optimizer_type, learning_rate, t, beta, batch_size in hyper_parameters:

        # if optimizer_type != 'momentum_stochastic_gradient_descent': # and optimizer_type != 'stochastic_gradient_descent':
        #     continue

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
    
    """
    Uncomment the line below to view loss plots:
    """
    # plt.show()