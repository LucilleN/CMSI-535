import numpy as np
import sklearn.datasets as skdata
import sklearn.metrics as skmetrics
import sklearn.preprocessing as skpreprocess
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge as RidgeRegression
from matplotlib import pyplot as plt


'''
Name: Doe, John (Please write names in <Last Name, First Name> format)

Collaborators: Doe, Jane (Please write names in <Last Name, First Name> format)

Collaboration details: Discussed <function name> implementation details with Jane Doe.

Summary:
Report your scores here. For example,

Experiment 1: Overfitting Linear Regression with Polynomial Expansion
.
.
.
Experiment 2: Underfitting Ridge Regression with alpha/lambda
.
.
.
Experiment 3: Ridge Regression with alpha/lambda and Polynomial Expansion
.
.
.

'''

def score_mean_squared_error(model, x, y):
    '''
    Scores the model on mean squared error metric

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
    # Implement the score mean squared error function
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
        labels : list[str] # this used to be just str
            list of names for legend
        colors : str
            colors for each line
        x_limits : list[float]
            min and max values of x axis
        y_limits : list[float]
            min and max values of y axis
        x_label : str # these used to be just floats
            name of x axis
        y_label : str
            name of y axis
    '''

    # Iterate through x_values, y_values, labels, and colors and plot them
    # with associated legend
    for x, y, label, color in zip(x_values, y_values, labels, colors):
        ax.plot(x, y, marker='o', color=color, label=label)
        ax.legend(loc='best')

    # Set x and y limits
    ax.set_xlim(x_limits)
    ax.set_ylim(y_limits)

    # Set x and y labels
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)


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

    # Initialize a list containing 1, 2, 3 as the degrees for polynomial expansion
    degrees = [1, 2, 3]

    # Initialize empty lists to store scores for MSE and R-squared
    scores_mse_linear_overfit_train = []
    scores_r2_linear_overfit_train = []
    scores_mse_linear_overfit_val = []
    scores_r2_linear_overfit_val = []
    scores_mse_linear_overfit_test = []
    scores_r2_linear_overfit_test = []

    for degree in degrees:

        # Initialize polynomial expansion
        poly_transform = skpreprocess.PolynomialFeatures(degree=degree)

        # Compute the polynomial terms needed for the data
        poly_transform.fit(x_train)

        # Transform the data by nonlinear mapping
        x_poly_train = poly_transform.transform(x_train)
        x_poly_val = poly_transform.transform(x_val)
        x_poly_test = poly_transform.transform(x_test)

        # Initialize scikit-learn linear regression model
        model_linear_overfit = LinearRegression()

        # Trains scikit-learn linear regression model
        model_linear_overfit.fit(x_poly_train, y_train)

        print('Results for linear regression model with degree {} polynomial expansion'.format(degree))

        # Test model on training set
        score_mse_linear_overfit_train = score_mean_squared_error(model_linear_overfit, x_poly_train, y_train)
        print('Training set mean squared error: {:.4f}'.format(score_mse_linear_overfit_train))

        score_r2_linear_overfit_train = model_linear_overfit.score(x_poly_train, y_train)
        print('Training set r-squared scores: {:.4f}'.format(score_r2_linear_overfit_train))

        # Save MSE and R-squared training scores
        scores_mse_linear_overfit_train.append(score_mse_linear_overfit_train)
        scores_r2_linear_overfit_train.append(score_r2_linear_overfit_train)
    
        # Test model on validation set
        score_mse_linear_overfit_val = score_mean_squared_error(model_linear_overfit, x_poly_val, y_val)
        print('Validation set mean squared error: {:.4f}'.format(score_mse_linear_overfit_val))

        score_r2_linear_overfit_val = model_linear_overfit.score(x_poly_val, y_val)
        print('Validation set r-squared scores: {:.4f}'.format(score_r2_linear_overfit_val))

        # Save MSE and R-squared validation scores
        scores_mse_linear_overfit_val.append(score_mse_linear_overfit_val)
        scores_r2_linear_overfit_val.append(score_r2_linear_overfit_val)

        # Test model on testing set
        score_mse_linear_overfit_test = score_mean_squared_error(model_linear_overfit, x_poly_test, y_test)
        print('Testing set mean squared error: {:.4f}'.format(score_mse_linear_overfit_test))

        score_r2_linear_overfit_test = model_linear_overfit.score(x_poly_test, y_test)
        print('Testing set r-squared scores: {:.4f}'.format(score_r2_linear_overfit_test))

        # Save MSE and R-squared testing scores
        scores_mse_linear_overfit_test.append(score_mse_linear_overfit_test)
        scores_r2_linear_overfit_test.append(score_r2_linear_overfit_test)

    # Convert each scores to NumPy arrays
    scores_mse_linear_overfit_train = np.array(scores_mse_linear_overfit_train)
    scores_mse_linear_overfit_val = np.array(scores_mse_linear_overfit_val)
    scores_mse_linear_overfit_test = np.array(scores_mse_linear_overfit_test)
    scores_r2_linear_overfit_train = np.array(scores_r2_linear_overfit_train)
    scores_r2_linear_overfit_val = np.array(scores_r2_linear_overfit_val)
    scores_r2_linear_overfit_test = np.array(scores_r2_linear_overfit_test)

    # Clip each set of MSE scores between 0 and 40
    scores_mse_linear_overfit_train = np.clip(scores_mse_linear_overfit_train, 0.0, 40.0)
    scores_mse_linear_overfit_val = np.clip(scores_mse_linear_overfit_val, 0.0, 40.0)
    scores_mse_linear_overfit_test = np.clip(scores_mse_linear_overfit_test, 0.0, 40.0)

    # Clip each set of R-squared scores between 0 and 1
    scores_r2_linear_overfit_train = np.clip(scores_r2_linear_overfit_train, 0.0, 1.0)
    scores_r2_linear_overfit_val = np.clip(scores_r2_linear_overfit_val, 0.0, 1.0)
    scores_r2_linear_overfit_test = np.clip(scores_r2_linear_overfit_test, 0.0, 1.0)

    # Create figure for training, validation and testing scores for different features
    n_experiments = scores_mse_linear_overfit_train.shape[0]
    fig = plt.figure()

    labels = ['Training', 'Validation', 'Testing']
    colors = ['blue', 'red', 'green']

    # Create the first subplot of a 1 by 2 figure to plot MSE for training, validation, testing
    ax = fig.add_subplot(1, 2, 1) 

    # Set x and y values
    # x_values refer to the degree of the polynomial [1, 2, 3]
    x_values = [range(1, n_experiments + 1)] * n_experiments # This is a list of 3 lists
    y_values = [
        scores_mse_linear_overfit_train,
        scores_mse_linear_overfit_val,
        scores_mse_linear_overfit_test,
    ]

    # Plot MSE scores for training, validation, testing sets
    # Set x limits to 0 to number of experiments + 1 and y limits between 0 and 40
    # Set x label to 'p-degree' and y label to 'MSE'
    plot_results(
        axis=ax,
        x_values=x_values,
        y_values=y_values,
        labels=labels,
        colors=colors,
        x_limits=[0.0, n_experiments + 1],
        y_limits=[0.0, 40.0],
        x_label='p-degree',
        y_label='MSE'
    )

    # TODO: Create the second subplot of a 1 by 2 figure to plot R-squared for training, validation, testing

    # TODO: Set x and y values

    # TODO: Plot R-squared scores for training, validation, testing sets
    # Set x limits to 0 to number of experiments + 1 and y limits between 0 and 1
    # Set x label to 'p-degree' and y label to 'R-squared',

    # TODO: Create super title 'Overfitted Linear Regression on Training, Validation and Testing Sets'

    plt.show()

    '''
    Experiment 2:
    Demonstrate that ridge regression will underfit with high weight (alpha/lambda) values
    '''

    print('Experiment 2: Underfitting Ridge Regression with alpha/lambda')

    # Initialize a list containing:
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
        # Initialize scikit-learn ridge regression model
        # alpha is the lambda (degree of regularization)
        model_ridge_underfit = RidgeRegression(alpha=alpha)

        # Trains scikit-learn ridge regression model
        model_ridge_underfit.fit(x_train, y_train)

        print('Results for scikit-learn RidgeRegression model with alpha={}'.format(alpha))

        # Test model on training set
        score_mse_ridge_underfit_train = score_mean_squared_error(model_ridge_underfit, x_train, y_train)
        print('Training set mean squared error: {:.4f}'.format(score_mse_ridge_underfit_train))

        score_r2_ridge_underfit_train = model_ridge_underfit.score(x_train, y_train)
        print('Training set r-squared scores: {:.4f}'.format(score_r2_ridge_underfit_train))

        # Save MSE and R-squared training scores
        scores_mse_ridge_underfit_train.append(score_mse_ridge_underfit_train)
        scores_r2_ridge_underfit_train.append(score_r2_ridge_underfit_train)

        # Test model on validation set
        score_mse_ridge_underfit_val = score_mean_squared_error(model_ridge_underfit, x_val, y_val)
        print('Validation set mean squared error: {:.4f}'.format(score_mse_ridge_underfit_val))

        score_r2_ridge_underfit_val = model_ridge_underfit.score(x_val, y_val)
        print('Validation set r-squared scores: {:.4f}'.format(score_r2_ridge_underfit_val))

        # Save MSE and R-squared validation scores
        scores_mse_ridge_underfit_val.append(score_mse_ridge_underfit_val)
        scores_r2_ridge_underfit_val.append(score_r2_ridge_underfit_val)

        # Test model on testing set
        score_mse_ridge_underfit_test = score_mean_squared_error(model_ridge_underfit, x_test, y_test)
        print('Testing set mean squared error: {:.4f}'.format(score_mse_ridge_underfit_test))

        score_r2_ridge_underfit_test = model_ridge_underfit.score(x_test, y_test)
        print('Testing set r-squared scores: {:.4f}'.format(score_r2_ridge_underfit_test))

        # Save MSE and R-squared testing scores
        scores_mse_ridge_underfit_test.append(score_mse_ridge_underfit_test)
        scores_r2_ridge_underfit_test.append(score_r2_ridge_underfit_test)

    # Convert each scores to NumPy arrays
    scores_mse_ridge_underfit_train = np.array(scores_mse_ridge_underfit_train)
    scores_mse_ridge_underfit_val = np.array(scores_mse_ridge_underfit_val)
    scores_mse_ridge_underfit_test = np.array(scores_mse_ridge_underfit_test)
    scores_r2_ridge_underfit_train = np.array(scores_r2_ridge_underfit_train)
    scores_r2_ridge_underfit_val = np.array(scores_r2_ridge_underfit_val)
    scores_r2_ridge_underfit_test = np.array(scores_r2_ridge_underfit_test)

    # Clip each set of MSE scores between 0 and 40
    scores_mse_ridge_underfit_train = np.clip(scores_mse_ridge_underfit_train, 0.0, 40.0)
    scores_mse_ridge_underfit_val = np.clip(scores_mse_ridge_underfit_val, 0.0, 40.0)
    scores_mse_ridge_underfit_test = np.clip(scores_mse_ridge_underfit_test, 0.0, 40.0)

    # Clip each set of R-squared scores between 0 and 1
    scores_r2_ridge_underfit_train = np.clip(scores_r2_ridge_underfit_train, 0.0, 1.0)
    scores_r2_ridge_underfit_val = np.clip(scores_r2_ridge_underfit_val, 0.0, 1.0)
    scores_r2_ridge_underfit_test = np.clip(scores_r2_ridge_underfit_test, 0.0, 1.0)

    # Create figure for training, validation and testing scores for different features
    n_experiments = scores_mse_ridge_underfit_train.shape[0]
    fig = plt.figure()

    labels = ['Training', 'Validation', 'Testing']
    colors = ['blue', 'red', 'green']

    # TODO: Create the first subplot of a 1 by 2 figure to plot MSE for training, validation, testing

    # TODO: Set x values (alphas in log scale )and y values (R-squared)

    # TODO: Plot MSE scores for training, validation, testing sets
    # Set x limits to 0 to log of highest alphas + 1 and y limits between 0 and 40
    # Set x label to 'alphas' and y label to 'MSE',

    # TODO: Create the second subplot of a 1 by 2 figure to plot R-squared for training, validation, testing

    # TODO: Set x values (alphas in log scale) and y values (R-squared)

    # TODO: Plot R-squared scores for training, validation, testing sets
    # Set x limits to 0 to 1100 and y limits between 0 and 1
    # Set x label to 'alphas' and y label to 'R-squared',

    # TODO: Create super title 'Underfitted Ridge Regression on Training, Validation and Testing Sets'

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
    alphas = []

    # TODO: Initialize polynomial expansion

    # TODO: Compute the polynomial terms needed for the data

    # TODO: Transform the data by nonlinear mapping

    # Initialize empty lists to store scores for MSE and R-squared
    scores_mse_ridge_poly_train = []
    scores_r2_ridge_poly_train = []
    scores_mse_ridge_poly_val = []
    scores_r2_ridge_poly_val = []
    scores_mse_ridge_poly_test = []
    scores_r2_ridge_poly_test = []

    for alpha in alphas:

        # TODO: Initialize scikit-learn linear regression model

        # TODO: Trains scikit-learn linear regression model

        print('Results for ridge regression model with alpha={} using degree {} polynomial expansion'.format(alpha, degree))

        # TODO: Test model on training set
        score_mse_ridge_poly_train = 0.0
        print('Training set mean squared error: {:.4f}'.format(score_mse_ridge_poly_train))

        score_r2_ridge_poly_train = 0.0
        print('Training set r-squared scores: {:.4f}'.format(score_r2_ridge_poly_train))

        # TODO: Save MSE and R-squared training scores

        # TODO: Test model on validation set
        score_mse_ridge_poly_val = 0.0
        print('Validation set mean squared error: {:.4f}'.format(score_mse_ridge_poly_val))

        score_r2_ridge_poly_val = 0.0
        print('Validation set r-squared scores: {:.4f}'.format(score_r2_ridge_poly_val))

        # TODO: Save MSE and R-squared validation scores

        # TODO: Test model on testing set
        score_mse_ridge_poly_test = 0.0
        print('Testing set mean squared error: {:.4f}'.format(score_mse_ridge_poly_test))

        score_r2_ridge_poly_test = 0.0
        print('Testing set r-squared scores: {:.4f}'.format(score_r2_ridge_poly_test))

        # TODO: Save MSE and R-squared testing scores

    # Convert each scores to NumPy arrays
    scores_mse_ridge_poly_train = np.array(scores_mse_ridge_poly_train)
    scores_mse_ridge_poly_val = np.array(scores_mse_ridge_poly_val)
    scores_mse_ridge_poly_test = np.array(scores_mse_ridge_poly_test)
    scores_r2_ridge_poly_train = np.array(scores_r2_ridge_poly_train)
    scores_r2_ridge_poly_val = np.array(scores_r2_ridge_poly_val)
    scores_r2_ridge_poly_test = np.array(scores_r2_ridge_poly_test)

    # TODO: Clip each set of MSE scores between 0 and 40

    # TODO: Clip each set of R-squared scores between 0 and 1

    # Create figure for training, validation and testing scores for different features
    n_experiments = scores_mse_ridge_poly_train.shape[0]
    fig = plt.figure()

    labels = ['Training', 'Validation', 'Testing']
    colors = ['blue', 'red', 'green']

    # TODO: Create the first subplot of a 1 by 2 figure to plot MSE for training, validation, testing

    # TODO: Set x values (alphas in log scale) and y values (R-squared)

    # TODO: Plot MSE scores for training, validation, testing sets
    # Set x limits to 0 to 1100 and y limits between 0 and 40
    # Set x label to 'alphas' and y label to 'MSE',

    # TODO: Create the second subplot of a 1 by 2 figure to plot R-squared for training, validation, testing

    # TODO: Set x values (alphas in log scale )and y values (R-squared)

    # TODO: Plot R-squared scores for training, validation, testing sets
    # Set x limits to 0 to 1100 and y limits between 0 and 1
    # Set x label to 'alphas' and y label to 'R-squared',

    # TODO: Create super title 'Ridge Regression with various alphas on Training, Validation and Testing Sets'

    plt.show()
