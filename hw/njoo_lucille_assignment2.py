import numpy as np
import sklearn.datasets as skdata
import sklearn.metrics as skmetrics
import sklearn.preprocessing as skpreprocess
from sklearn.linear_model import Ridge as RidgeRegression
from matplotlib import pyplot as plt


'''
Name: Njoo, Lucille

Collaborators: Arteaga, Andrew 

Collaboration details: Discussed how to do the matrix multiplication in the closed-form solution equation for w*. 

Summary:

What you did for this assignment:

For this assignment, I implemented a RidgeRegressionClosedForm class by writing its `score`, `predict`, and `fit` 
functions. The `fit` function computed the optimal weights, w*, using the closed form solution of: 
    w* = ( Z^T * Z + lambda * I )^-1 * Z^T * y
where w* is the optimal weights that minimize loss as much as possible while staying within the regularization 
constraint. 
Then, the main method performs two experiments to complare our RidgeRegressionClosedForm class with the 
skikit-learn implementation of the RidgeRegression class. To do so, we first initialize an sklearn 
RidgeRegression model, and then for increasing values of alpha from 10^0 up to 10^5, we train the model on the 
polynomial-expanded training set, test it on the polynomial-expanded training, validation, and testing sets, and 
finally plot the MSE and R-squared scores for the different alphas. Then we repeat this entire
training-validation-testing loop for our own implementation of the RidgeRegressionClosedForm class, and plot its 
MSE and R-squared scores for different alphas. 

What loss you minimized:

The loss function I minimized was: 
    l(w) = 1/N * ( (Zw-y)^T * (Zw-y) + lambda * w^T * w )
The first term of the sum in the above equation, (Zw-y)^T * (Zw-y), represents the data fidelity loss. The second 
term, lambda * w^T * w, represents the regularization loss. (In the implementation below, we use the variable 
`alpha` in place of lambda.) Thus, the above equation can be captured by: loss = data fidelity loss + 
regularization loss, where the data fidelity loss is 1/N * ((Zw-y)^T * (Zw-y)), and the regularization loss is 1/
N * (lambda * w^T * w).

How you minimized the loss function:

I minimized the loss by computing the optimal weights, w*, using the closed-form solution for Ridge Regression 
with a soft constraint. These weights were computed with the equation w* = ( Z^T * Z + lambda * I )^-1 * Z^T * y.
This equation gives us the weights that minimize the loss while still staying within the regularization 
constraint, because it is derived by setting the gradient of the data fidelity loss, plus lambda * the gradient 
of the regularization loss, to 0. This means that the solution w* must be the point where the gradients point in 
the opposite directions, and the w* that we calculate is the closest we can get to the weights that minimize loss 
without regularization. Thus, the minimum loss is l(w*). 

Report your scores here:

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
Training Loss: 6.664
Data Fidelity Loss: 6.413  Regularization Loss: 0.252
Training set mean squared error: 6.4127
Training set r-squared scores: 0.9262
Validation set mean squared error: 8.9723
Validation set r-squared scores: 0.8720
Testing set mean squared error: 18.4835
Testing set r-squared scores: 0.7633
Results for our RidgeRegression model with alpha=10.0
Training Loss: 7.415
Data Fidelity Loss: 7.026  Regularization Loss: 0.389
Training set mean squared error: 7.0258
Training set r-squared scores: 0.9191
Validation set mean squared error: 9.5386
Validation set r-squared scores: 0.8639
Testing set mean squared error: 16.1997
Testing set r-squared scores: 0.7926
Results for our RidgeRegression model with alpha=100.0
Training Loss: 8.347
Data Fidelity Loss: 7.930  Regularization Loss: 0.417
Training set mean squared error: 7.9301
Training set r-squared scores: 0.9087
Validation set mean squared error: 10.6471
Validation set r-squared scores: 0.8481
Testing set mean squared error: 16.3874
Testing set r-squared scores: 0.7902
Results for our RidgeRegression model with alpha=1000.0
Training Loss: 9.429
Data Fidelity Loss: 8.911  Regularization Loss: 0.517
Training set mean squared error: 8.9114
Training set r-squared scores: 0.8974
Validation set mean squared error: 11.2366
Validation set r-squared scores: 0.8397
Testing set mean squared error: 14.5313
Testing set r-squared scores: 0.8139
Results for our RidgeRegression model with alpha=10000.0
Training Loss: 10.707
Data Fidelity Loss: 10.042  Regularization Loss: 0.665
Training set mean squared error: 10.0420
Training set r-squared scores: 0.8844
Validation set mean squared error: 11.8909
Validation set r-squared scores: 0.8304
Testing set mean squared error: 13.8512
Testing set r-squared scores: 0.8226
Results for our RidgeRegression model with alpha=100000.0
Training Loss: 12.984
Data Fidelity Loss: 11.598  Regularization Loss: 1.385
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
        # Turn z from (d x N) to Z (N x d) by taking its transpose
        Z = z.T
        N, d = Z.shape

        # w* = ( Z^T * Z + lambda * I )^-1 * Z^T * y
        Z_transpoze_Z = np.matmul(Z.T, Z)
        lambda_I = alpha * np.identity(d)
        inverse = np.linalg.inv(Z_transpoze_Z + lambda_I)
        w_star = np.matmul(np.matmul(inverse, Z.T), y)
        
        self.__weights = w_star
        
        # Compute loss:
        # l(w) = 1/N * ( (Zw-y)^T * (Zw-y) + lambda * w^T * w )
        # plug in w_star to get the minimum loss
        Z_w_star_minus_y = np.matmul(Z, w_star) - y

        loss_data_fidelity = 1/N * np.matmul(Z_w_star_minus_y.T, Z_w_star_minus_y)
        loss_regularization = 1/N * alpha * np.matmul(w_star.T, w_star)
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

        # Unexplained variation: u = sum (y_hat - y)^2
        sum_squared_errors = np.sum((y_hat - y) ** 2)

        # Total variation in the data: v = sum (y - y_mean)^2
        y_mean = np.mean(y)
        sum_variance = np.sum((y - y_mean) ** 2)

        # R-squared score: r^2 = 1 - (u / v)
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
        # MSE = mean(difference^2)
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
        predictions = self.predict(x)

        if scoring_func == 'r_squared':
            return self.__score_r_squared(predictions, y)
        elif scoring_func == 'mean_squared_error':
            return self.__score_mean_squared_error(predictions, y)
        else:
            raise ValueError('Encountered unsupported scoring_func: {}'.format(scoring_func))


'''
Utility functions to compute error and plot
'''
def score_mean_squared_error(model, x, y):
    '''
    Scores the model on mean squared error metric from skmetrics

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
    Part 1: Scikit-Learn Ridge Regression Model
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

    # Set x (alpha in log scale) and y values (MSE)
    x_values = [np.log(np.asarray(alphas) + 1.0)] * n_experiments
    y_values = [
        scores_mse_ridge_scikit_train,
        scores_mse_ridge_scikit_val,
        scores_mse_ridge_scikit_test,
    ]

    # Plot MSE scores for training, validation, testing sets
    #   - Set x limits to 0 to max of x_values + 1 and y limits between 0 and 40
    #   - Set x label to 'alpha (log scale)' and y label to 'MSE',
    plot_results(
        axis=ax,
        x_values=x_values,
        y_values=y_values,
        labels=labels,
        colors=colors,
        x_limits=[0.0, np.log(alphas[-1]) + 1],
        y_limits=[0.0, 40.0],
        x_label='alphas',
        y_label='MSE'
    )

    # Create the second subplot of a 1 by 2 figure to plot R-squared for training, validation, testing
    ax = fig.add_subplot(1, 2, 2)

    # Set x (alpha in log scale) and y values (R-squared)
    x_values = [np.log(np.asarray(alphas) + 1.0)] * n_experiments
    y_values = [
        scores_r2_ridge_scikit_train,
        scores_r2_ridge_scikit_val,
        scores_r2_ridge_scikit_test,
    ]

    # Plot R-squared scores for training, validation, testing sets
    #   - Set x limits to 0 to max of x_values + 1 and y limits between 0 and 1
    #   - Set x label to 'alpha (log scale)' and y label to 'R-squared',
    plot_results(
        axis=ax,
        x_values=x_values,
        y_values=y_values,
        labels=labels,
        colors=colors,
        x_limits=[0.0, np.log(alphas[-1]) + 1],
        y_limits=[0.0, 1.0],
        x_label='alphas',
        y_label='R-squared'
    )

    # Create super title 'Scikit-Learn Ridge Regression on Training, Validation and Testing Sets'
    plt.suptitle('Scikit-Learn Ridge Regression on Training, Validation and Testing Sets')

    '''
    Part 2: Our Ridge Regression Model
    Trains and tests our ridge regression model using different alphas
    '''

    # Initialize empty lists to store scores for MSE and R-squared
    scores_mse_ridge_ours_train = []
    scores_r2_ridge_ours_train = []
    scores_mse_ridge_ours_val = []
    scores_r2_ridge_ours_val = []
    scores_mse_ridge_ours_test = []
    scores_r2_ridge_ours_test = []

    # Convert dataset (N x d) to correct shape (d x N)
    x_poly_train = np.transpose(x_poly_train, axes=(1, 0))
    x_poly_val = np.transpose(x_poly_val, axes=(1, 0))
    x_poly_test = np.transpose(x_poly_test, axes=(1, 0))

    # For each alpha, train a ridge regression model on degree 2 polynomial features
    for alpha in alphas:

        # Initialize our own ridge regression model
        model = RidgeRegressionClosedForm()

        print('Results for our RidgeRegression model with alpha={}'.format(alpha))

        # Train model on training set
        model.fit(x_poly_train, y_train, alpha=alpha)

        # Test model on training set using mean squared error and r-squared
        score_mse_ridge_ours_train = model.score(x_poly_train, y_train, scoring_func="mean_squared_error")
        print('Training set mean squared error: {:.4f}'.format(score_mse_ridge_ours_train))
        score_r2_ridge_ours_train = model.score(x_poly_train, y_train, scoring_func="r_squared")
        print('Training set r-squared scores: {:.4f}'.format(score_r2_ridge_ours_train))

        # Save MSE and R-squared training scores
        scores_mse_ridge_ours_train.append(score_mse_ridge_ours_train)
        scores_r2_ridge_ours_train.append(score_r2_ridge_ours_train)

        # Test model on validation set using mean squared error and r-squared
        score_mse_ridge_ours_val = model.score(x_poly_val, y_val, scoring_func="mean_squared_error")
        print('Validation set mean squared error: {:.4f}'.format(score_mse_ridge_ours_val))
        score_r2_ridge_ours_val = model.score(x_poly_val, y_val, scoring_func="r_squared")
        print('Validation set r-squared scores: {:.4f}'.format(score_r2_ridge_ours_val))

        # Save MSE and R-squared validation scores
        scores_mse_ridge_ours_val.append(score_mse_ridge_ours_val)
        scores_r2_ridge_ours_val.append(score_r2_ridge_ours_val)

        # Test model on testing set using mean squared error and r-squared
        score_mse_ridge_ours_test = model.score(x_poly_test, y_test, scoring_func="mean_squared_error")
        print('Testing set mean squared error: {:.4f}'.format(score_mse_ridge_ours_test))
        score_r2_ridge_ours_test = model.score(x_poly_test, y_test, scoring_func="r_squared")
        print('Testing set r-squared scores: {:.4f}'.format(score_r2_ridge_ours_test))

        # Save MSE and R-squared testing scores
        scores_mse_ridge_ours_test.append(score_mse_ridge_ours_test)
        scores_r2_ridge_ours_test.append(score_r2_ridge_ours_test)

    # Convert each scores to NumPy arrays
    scores_mse_ridge_ours_train = np.array(scores_mse_ridge_ours_train)
    scores_mse_ridge_ours_val = np.array(scores_mse_ridge_ours_val)
    scores_mse_ridge_ours_test = np.array(scores_mse_ridge_ours_test)
    scores_r2_ridge_ours_train = np.array(scores_r2_ridge_ours_train)
    scores_r2_ridge_ours_val = np.array(scores_r2_ridge_ours_val)
    scores_r2_ridge_ours_test = np.array(scores_r2_ridge_ours_test)

    # Clip each set of MSE scores between 0 and 40
    scores_mse_ridge_ours_train = np.clip(scores_mse_ridge_ours_train, 0.0, 40.0)
    scores_mse_ridge_ours_val = np.clip(scores_mse_ridge_ours_val, 0.0, 40.0)
    scores_mse_ridge_ours_test = np.clip(scores_mse_ridge_ours_test, 0.0, 40.0)

    # Clip each set of R-squared scores between 0 and 1
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

    # Set x (alpha in log scale) and y values (MSE)
    x_values = [np.log(np.asarray(alphas) + 1.0)] * n_experiments
    y_values = [
        scores_mse_ridge_ours_train,
        scores_mse_ridge_ours_val,
        scores_mse_ridge_ours_test,
    ]

    # Plot MSE scores for training, validation, testing sets
    #   - Set x limits to 0 to max of x_values + 1 and y limits between 0 and 40
    #   - Set x label to 'alpha (log scale)' and y label to 'MSE'
    plot_results(
        axis=ax,
        x_values=x_values,
        y_values=y_values,
        labels=labels,
        colors=colors,
        x_limits=[0.0, np.log(alphas[-1]) + 1],
        y_limits=[0.0, 40.0],
        x_label='alphas',
        y_label='MSE'
    )

    # Create the second subplot of a 1 by 2 figure to plot R-squared for training, validation, testing
    ax = fig.add_subplot(1, 2, 2)

    # Set x (alpha in log scale) and y values (R-squared)
    x_values = [np.log(np.asarray(alphas) + 1.0)] * n_experiments
    y_values = [
        scores_r2_ridge_ours_train,
        scores_r2_ridge_ours_val,
        scores_r2_ridge_ours_test,
    ]

    # Plot R-squared scores for training, validation, testing sets
    #   - Set x limits to 0 to max of x_values + 1 and y limits between 0 and 1
    #   - Set x label to 'alpha (log scale)' and y label to 'R-squared',
    plot_results(
        axis=ax,
        x_values=x_values,
        y_values=y_values,
        labels=labels,
        colors=colors,
        x_limits=[0.0, np.log(alphas[-1]) + 1],
        y_limits=[0.0, 1.0],
        x_label='alphas',
        y_label='R-squared'
    )
    
    # Create super title 'Our Ridge Regression on Training, Validation and Testing Sets'
    plt.suptitle('Our Ridge Regression on Training, Validation and Testing Sets')

    plt.show()
