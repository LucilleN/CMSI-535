import numpy as np
import sklearn.datasets as skdata
import sklearn.metrics as skmetrics
from sklearn.linear_model import LinearRegression


'''
Name: Njoo, Lucille

Collaborators: Arteaga, Andrew

Collaboration details: Discussed `__fit_normal_equation` implementation details with Andrew Arteaga.

Summary:

Results using scikit-learn LinearRegression model
Training set mean squared error: 23.2560
Training set r-squared scores: 0.7323
Validation set mean squared error: 17.6111
Validation set r-squared scores: 0.7488
Testing set mean squared error: 17.1465
Testing set r-squared scores: 0.7805
Results using our linear regression model trained with normal_equation
Training set mean squared error: 25.9360
Training set r-squared scores: 0.7015
Validation set mean squared error: 18.4747
Validation set r-squared scores: 0.7365
Testing set mean squared error: 18.1262
Testing set r-squared scores: 0.7679
Results using our linear regression model trained with pseudoinverse
Training set mean squared error: 25.9360
Training set r-squared scores: 0.7015
Validation set mean squared error: 18.4747
Validation set r-squared scores: 0.7365
Testing set mean squared error: 18.1262
Testing set r-squared scores: 0.7679
'''

'''
Implementation of Linear Regression model with two methods for fitting self.__weights 
to the training data: by directly solving the normal equation to find w*, or by 
solving the pseudoinverse using SVD. The class also supports two different methods
for scoring the model: using R^2 or using MSE (mean squared error). 

The main script then initializes a Linear Regression model from scikit learn, fits the 
model on training data, and evaluates it on the training, validation, and testing sets 
using both R^2 and MSE. Then, we initialize two of our own LinearRegressionClosedForm 
objects and fit to the training data using both of the two aforementioned methods -- via 
the normal equation and via the pseudoninverse. Finally, we evaluate both on the training, 
validation, and testing sets, again using both R^2 and MSE.
'''


class LinearRegressionClosedForm(object):

    def __init__(self):
        # Define private variables
        self.__weights = None

    def __fit_normal_equation(self, X, y):
        '''
        Fits the model to x and y via normal equation

        Args:
            X : numpy
                N x d feature vector
            y : numpy
                1 x N ground-truth label
        '''
        # w* = (X^T * X)^-1 * X^T * y
        inverse = np.linalg.inv(np.matmul(X.T, X))
        self.__weights = np.matmul(np.matmul(inverse, X.T), y)

    def __fit_pseudoinverse(self, X, y):
        '''
        Fits the model to x and y via pseudoinverse

        Args:
            X : numpy
                N x d feature vector
            y : numpy
                1 x N ground-truth label
        '''
        # To get pseudoinverse of X:
        # 1. compute U, S, V_t from SVD
        # 2. get pseudoinverse S+ by taking reciprocal of S and transposing the result
        # 3. multiply V S+ U.T to get X+

        # Assume X is (N x d)
        # We need: U (N x N), S (N x d), V_t (d x d)
        U, S, V_t = np.linalg.svd(X)
        # This gives us: U (N x N), S (d), V_t (d, d)
        # So we need to convert S (d) to S (N x d), a diagonal matrix

        # To get diagonal (square) matrix from a vector, we can use np.diag()
        # Recall that we also have to compute 1/s for every sigma s
        S_diag = np.diag(1.0 / S)  # a (d x d) matrix

        # We need to turn this into a (N, d) matrix
        # S should be zeros everywhere else; just pad it with extra rows of zeros
        # Specifically, we need to pad N-d along the 0th dimension, and d along the 1st dimension
        # U.shape[0] gives us N, S.shape[0] gives us d
        # So we’re adding N-d rows (each with d columns) so that we have a total of N rows
        padding = np.zeros([U.shape[0] - S.shape[0], S.shape[0]])
        S_pseudo = np.concatenate([S_diag, padding], axis=0)

        # To get Sigma+, we need to transpose it
        S_pseudo = S_pseudo.T

        # X+ = V S+ U.T
        # Since we're given V_t, we need to transpose to get V
        # We're given U, need to transpose to get U_t
        X_pseudo = np.matmul(np.matmul(V_t.T, S_pseudo), U.T)

        self.__weights = np.matmul(X_pseudo, y)

    def fit(self, x, y, solver='normal_equation'):
        '''
        Fits the model to x and y by solving the ordinary least squares
        using normal equation or pseudoinverse (SVD)

        Args:
            x : numpy
                d x N feature vector
            y : numpy
                1 x N ground-truth label
            solver : str
                solver types: normal_equation, pseudoinverse
        '''
        # Turn x from (d x N) to X (N, d) by taking its transpose
        X = x.T

        if solver == 'normal_equation':
            self.__fit_normal_equation(X, y)
        elif solver == 'pseudoinverse':
            self.__fit_pseudoinverse(X, y)
        else:
            raise ValueError(
                'Encountered unsupported solver: {}'.format(solver))

    def predict(self, x):
        '''
        Predicts the label for each feature vector x

        Args:
            x : numpy
                d x N feature vector

        Returns:
            numpy : d x 1 label vector
        '''
        predictions = np.matmul(self.__weights.T, x)
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
        # Unexplained variation (u): sum (y_hat - y)^2
        sum_squared_errors = np.sum((y_hat - y) ** 2)

        # Total variation in the data (v): sum (y - y_mean)^2
        y_mean = np.mean(y)
        sum_variance = np.sum((y - y_mean) ** 2)

        # r^2 = 1 - (u / v)
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
            raise ValueError(
                'Encountered unsupported scoring_func: {}'.format(scoring_func))


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
    Trains and tests linear regression model from scikit-learn
    '''
    model = LinearRegression()
    model.fit(x_train, y_train)

    print('Results using scikit-learn LinearRegression model')

    predictions_train = model.predict(x_train)
    scores_mse_train = skmetrics.mean_squared_error(
        predictions_train, y_train)  # 23.2560, looks big because squared
    print('Training set mean squared error: {:.4f}'.format(scores_mse_train))
    # sklearn score function uses r2
    scores_r2_train = model.score(x_train, y_train)  # 0.7323
    print('Training set r-squared scores: {:.4f}'.format(scores_r2_train))

    predictions_val = model.predict(x_val)
    scores_mse_val = skmetrics.mean_squared_error(
        predictions_val, y_val)  # 17.6111
    print('Validation set mean squared error: {:.4f}'.format(scores_mse_val))
    scores_r2_val = model.score(x_val, y_val)  # 0.7488
    print('Validation set r-squared scores: {:.4f}'.format(scores_r2_val))

    predictions_test = model.predict(x_test)
    scores_mse_test = skmetrics.mean_squared_error(
        predictions_test, y_test)  # 17.1465
    print('Testing set mean squared error: {:.4f}'.format(scores_mse_test))
    scores_r2_test = model.score(x_test, y_test)  # 0.7805
    print('Testing set r-squared scores: {:.4f}'.format(scores_r2_test))

    '''
    Trains and tests our linear regression model using different solvers
    '''
    # Obtain dataset in correct shape (d x N) previously (N x d)
    x_train = np.transpose(x_train, axes=(1, 0))
    x_val = np.transpose(x_val, axes=(1, 0))
    x_test = np.transpose(x_test, axes=(1, 0))

    # Train two LinearRegressionClosedForm models using normal equation and pseudoinverse
    solvers = ['normal_equation', 'pseudoinverse']
    for solver in solvers:
        model = LinearRegressionClosedForm()

        print('Results using our linear regression model trained with {}'.format(solver))
        model.fit(x_train, y_train, solver=solver)

        # Test model on training set using mean squared error and r-squared
        scores_mse_train = model.score(
            x_train, y_train, scoring_func="mean_squared_error")
        print('Training set mean squared error: {:.4f}'.format(
            scores_mse_train))

        scores_r2_train = model.score(
            x_train, y_train, scoring_func="r_squared")
        print('Training set r-squared scores: {:.4f}'.format(scores_r2_train))

        # Test model on validation set using mean squared error and r-squared
        scores_mse_val = model.score(
            x_val, y_val, scoring_func="mean_squared_error")
        print('Validation set mean squared error: {:.4f}'.format(
            scores_mse_val))

        scores_r2_val = model.score(
            x_val, y_val, scoring_func="r_squared")
        print('Validation set r-squared scores: {:.4f}'.format(scores_r2_val))

        # Test model on testing set using mean squared error and r-squared
        scores_mse_test = model.score(
            x_test, y_test, scoring_func="mean_squared_error")
        print('Testing set mean squared error: {:.4f}'.format(scores_mse_test))

        scores_r2_test = model.score(
            x_test, y_test, scoring_func="r_squared")
        print('Testing set r-squared scores: {:.4f}'.format(scores_r2_test))
