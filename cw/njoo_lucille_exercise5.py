import numpy as np
import sklearn.datasets as skdata
import sklearn.metrics as skmetrics
import sklearn.preprocessing as skpreprocess
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt


'''
Name: Njoo, Lucille

Collaborators: None

Collaboration details: N/A

Summary:
Report your scores here. For example,

Results using scikit-learn LinearRegression model with linear features
Training set mean squared error: 23.2560
Training set r-squared scores: 0.7323
Validation set mean squared error: 17.6111
Validation set r-squared scores: 0.7488
Testing set mean squared error: 17.1465
Testing set r-squared scores: 0.7805
Results of LinearRegression model using scikit-learn order-2 polynomial expansion features
Training set mean squared error: 8.8948
Training set r-squared scores: 0.8976
Validation set mean squared error: 11.4985
Validation set r-squared scores: 0.8360
Testing set mean squared error: 34.8401
Testing set r-squared scores: 0.5539
Results of LinearRegression model using scikit-learn order-3 polynomial expansion features
Training set mean squared error: 0.0000
Training set r-squared scores: 1.0000
Validation set mean squared error: 131227.0451
Validation set r-squared scores: -1870.9939
Testing set mean squared error: 119705.2269
Testing set r-squared scores: -1531.7130
Results for LinearRegression model using our implementation of order-2 polynomial expansion features
Training set mean squared error: 8.8948
Training set r-squared scores: 0.8976
Validation set mean squared error: 11.4985
Validation set r-squared scores: 0.8360
Testing set mean squared error: 34.8401
Testing set r-squared scores: 0.5539
Results for LinearRegression model using our implementation of order-3 polynomial expansion features
Training set mean squared error: 6.5635
Training set r-squared scores: 0.9245
Validation set mean squared error: 9.4650
Validation set r-squared scores: 0.8650
Testing set mean squared error: 27.7505
Testing set r-squared scores: 0.6447
'''

'''
Implementation of our polynomial expansion for nonlinear mapping
'''
class PolynomialFeatureExpansion(object):

    def __init__(self, degree):
        '''
        Args:
            degree : int
                order or degree of polynomial for expansion
        '''
        # Degree or order of polynomial we will expand to
        self.__degree = degree

        # Polynomial terms that we will use based on the specified degree
        self.__polynomial_terms = []

    def transform(self, X):
        '''
        Computes up to p-order (degree) polynomial features and augments them to the data by
        1. We need to add a bias (x_0)
        2. We need to augment up to p-order polynomial

        Args:
            X : numpy
                N x d feature vector

        Returns:
            polynomial expanded features in Z space
        '''

        # Initialize the bias
        bias = np.ones([X.shape[0], 1])

        # Initialize polynomial expansion features Z
        # Z contains [x_0, x_1, x_2, ..., x_d]
        Z = [bias, X]

        # If degree is less than 2, then return the original features
        if self.__degree < 2:
            Z = np.concatenate(Z, axis=1)
            return Z

        # Split X into its d dimensions separately
        linear_features = np.split(X, indices_or_sections=X.shape[1], axis=1)

        # if self.__degree == 2:
        for p in range(2, self.__degree + 1):
            # Let's consider the example x = (x_1, x_2)
            # phi_2(x) = (x_0, x_1, x_2, x_1^2, x_1 x_2, x_2^2)

            # Maintain a list of new polynomial features that we've accumulated
            new_polynomial_features = []

            # Keep track of the polynomial terms that we will keep
            polynomial_terms = []

            # For every linear feature
            for l1 in range(len(linear_features)):

                # Multiply it by every linear feature
                for l2 in range(len(linear_features)):
                    # First iteration of outer loop
                    # 1. x_1 * x_1 = x_1^2
                    # 2. x_1 * x_2 = x_1 x_2
                    # Second iteration of outer loop
                    # 1. x_2 * x_1 = x_1 x_2 (discard based on condition, already exists)
                    # 2. x_2 * x_2 = x_2^2

                    polynomial_feature = linear_features[l1] * linear_features[l2]

                    # Check if we have already found the polynomial terms to keep
                    if len(self.__polynomial_terms) < self.__degree - 1:

                        # If we have not, then iterate through the expansion
                        keep_polynomial_term  = True

                        # Check if we already have the feature created
                        for feature in new_polynomial_features:
                            if np.sum(polynomial_feature - feature) == 0.0:
                                keep_polynomial_term = False
                                break

                        # Keep track of whether we keep or discard (True/False) the term
                        polynomial_terms.append(keep_polynomial_term)

                        if keep_polynomial_term:
                            # And append the result to the new set of polynomial features
                            new_polynomial_features.append(polynomial_feature)
                    else:
                        # Check if the current polynomial term was kept
                        keep_polynomial_term = self.__polynomial_terms[0][l1 * len(linear_features) + l2]

                        if keep_polynomial_term:
                            # And append the result to the new set of polynomial features
                            new_polynomial_features.append(polynomial_feature)

            # If we've never processed the polynomial terms before, save the list of terms to keep
            if len(self.__polynomial_terms) < self.__degree - 1:
                self.__polynomial_terms.append(polynomial_terms)

            # Add the new polynomial features to Z
            # Concatenates x_1^2, x_1 x_2, x_2^2 together into a matrix
            # [x_1^2, x_1 x_2, x_2^2] of shape (N, 3)
            Z.append(np.concatenate(new_polynomial_features, axis=1))

        # Concatenate every term into the feature vector (augmenting X with polynomial features)
        # Z becomes [x_0, x_1, x_2, x_1^2, x_1 x_2, x_2^2] of shape (N x 6)
        Z = np.concatenate(Z, axis=1)

        # TODO: Implement the polynomial_expansion function for degree larger than 2
        # Hint: (x_1 + x_2)^3 = (x_1 + x_2) * (x_1 + x_2) * (x_1 + x_2)

        return Z

def split_data(x, y):
    '''
    Splits raw data x and y into training, validation, and testing sets, using an 
    80% - 10% - 10% split.

    Args:
        x: numpy array
        y: numpy array

    Returns:
        (x_train, x_val, x_test, y_train, y_val, y_test): tuple of 6 numpy arrays
    '''
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
    return (x_train, x_val, x_test, y_train, y_val, y_test)


def score_model_on_data(model, x, y):
    '''
    Returns MSE and r-squared scores of model evaluated on any arbitrary 
    testing sets of x and y.

    Args:
        x: numpy array
        y: numpy array

    Returns:
        (score_mse, score_r2): tuple of 2 floats
    '''
    predictions = model.predict(x)
    score_mse = skmetrics.mean_squared_error(predictions, y)
    score_r2 = model.score(x, y)
    return (score_mse, score_r2)


if __name__ == '__main__':

    '''
    Load Boston Housing data and split into train, val, test
    '''
    boston_housing_data = skdata.load_boston()
    x = boston_housing_data.data
    y = boston_housing_data.target

    x_train, x_val, x_test, y_train, y_val, y_test = split_data(x, y)

    '''
    Trains and tests linear regression from scikit-learn
    '''
    # Variables to hold training, validation and testing scores for linear up to p-order polynomial expansion
    scores_mse_train = []
    scores_mse_val = []
    scores_mse_test = []
    scores_r2_train = []
    scores_r2_val = []
    scores_r2_test = []

    # Initialize scikit-learn linear regression model
    model = LinearRegression()
    # Trains scikit-learn linear regression model
    model.fit(x_train, y_train)
    print('Results using scikit-learn LinearRegression model with linear features')

    # Test model on training set
    # predictions_train = model.predict(x_train)
    # score_mse_train = skmetrics.mean_squared_error(predictions_train, y_train)
    # score_r2_train = model.score(x_train, y_train)
    score_mse_train, score_r2_train = score_model_on_data(model, x_train, y_train)
    # Save MSE and R-square scores on training set
    scores_mse_train.append(score_mse_train)
    scores_r2_train.append(score_r2_train)
    print('Training set mean squared error: {:.4f}'.format(score_mse_train))
    print('Training set r-squared scores: {:.4f}'.format(score_r2_train))

    # Test model on validation set
    predictions_val = model.predict(x_val)
    score_mse_val = skmetrics.mean_squared_error(predictions_val, y_val)
    print('Validation set mean squared error: {:.4f}'.format(score_mse_val))
    score_r2_val = model.score(x_val, y_val)
    print('Validation set r-squared scores: {:.4f}'.format(score_r2_val))
    # Save MSE and R-square scores on validation set
    scores_mse_val.append(score_mse_val)
    scores_r2_val.append(score_r2_val)

    # Test model on testing set
    predictions_test = model.predict(x_test)
    score_mse_test = skmetrics.mean_squared_error(predictions_test, y_test)
    print('Testing set mean squared error: {:.4f}'.format(score_mse_test))
    score_r2_test = model.score(x_test, y_test)
    print('Testing set r-squared scores: {:.4f}'.format(score_r2_test))
    # Save MSE and R-square scores on testing set
    scores_mse_test.append(score_mse_test)
    scores_r2_test.append(score_r2_test)

    # Set the degrees/orders of polynomials to be 2 and 3 for nonlinear mapping
    degrees_polynomial = [2, 3]

    '''
    Trains and tests linear regression from scikit-learn with scikit-learn polynomial features
    '''
    for degree in degrees_polynomial:

        print('Results of LinearRegression model using scikit-learn order-{} polynomial expansion features'.format(degree))

        # Initialize polynomial expansion
        poly_transform = skpreprocess.PolynomialFeatures(degree=degree)

        # Compute the polynomial terms needed for the data
        # Generates x_1^2, x_1 x_2, x_1 x_3, ... , x_d^2
        poly_transform.fit(x_train)

        # Transform the data by nonlinear mapping
        # Applies all the polynomial terms to the data and augments it to x
        # Computes the values for x_0, x_1, x_2, ..., x_1^2, x_1 x_2, ... x_d^2
        # x_1 = 2, x_2 = 4 : x -> (1, 2, 4, ..., 4, 8, ..., x_d^2)
        # This is the part that plugs x into phi(x) to get z
        x_poly_train = poly_transform.transform(x_train)
        x_poly_val = poly_transform.transform(x_val)
        x_poly_test = poly_transform.transform(x_test)

        # Initialize scikit-learn linear regression model
        model_poly = LinearRegression()
        # Trains scikit-learn linear regression model using p-order polynomial expansion
        model_poly.fit(x_poly_train, y_train)

        # Test model on training set
        predictions_poly_train = model_poly.predict(x_poly_train)
        score_mse_poly_train = skmetrics.mean_squared_error(predictions_poly_train, y_train)
        print('Training set mean squared error: {:.4f}'.format(score_mse_poly_train))
        score_r2_poly_train = model_poly.score(x_poly_train, y_train)
        print('Training set r-squared scores: {:.4f}'.format(score_r2_poly_train))
        # Save MSE and R-square scores on training set
        scores_mse_train.append(score_mse_poly_train)
        scores_r2_train.append(score_r2_poly_train)

        # Test model on validation set
        predictions_poly_val = model_poly.predict(x_poly_val)
        score_mse_poly_val = skmetrics.mean_squared_error(predictions_poly_val, y_val)
        print('Validation set mean squared error: {:.4f}'.format(score_mse_poly_val))
        score_r2_poly_val = model_poly.score(x_poly_val, y_val)
        print('Validation set r-squared scores: {:.4f}'.format(score_r2_poly_val))
        # Save MSE and R-square scores on validation set
        scores_mse_val.append(score_mse_poly_val)
        scores_r2_val.append(score_r2_poly_val)

        # Test model on testing set
        predictions_poly_test = model_poly.predict(x_poly_test)
        score_mse_poly_test = skmetrics.mean_squared_error(predictions_poly_test, y_test)
        print('Testing set mean squared error: {:.4f}'.format(score_mse_poly_test))
        score_r2_poly_test = model_poly.score(x_poly_test, y_test)
        print('Testing set r-squared scores: {:.4f}'.format(score_r2_poly_test))
        # Save MSE and R-square scores on testing set
        scores_mse_test.append(score_mse_poly_test)
        scores_r2_test.append(score_r2_poly_test)

    # Convert each scores list to numpy arrays so that we can plot them
    scores_mse_train = np.array(scores_mse_train)
    scores_mse_val = np.array(scores_mse_val)
    scores_mse_test = np.array(scores_mse_test)
    scores_r2_train = np.array(scores_r2_train)
    scores_r2_val = np.array(scores_r2_val)
    scores_r2_test = np.array(scores_r2_test)

    # Clip each set of MSE scores between 0 and 50 because some are huge
    scores_mse_train = np.clip(scores_mse_train, 0.0, 50.0)
    scores_mse_val = np.clip(scores_mse_val, 0.0, 50.0)
    scores_mse_test = np.clip(scores_mse_test, 0.0, 50.0)
    # Clip each set of R-squared scores between 0 and 1
    scores_r2_train = np.clip(scores_r2_train, 0.0, 1.0)
    scores_r2_val = np.clip(scores_r2_val, 0.0, 1.0)
    scores_r2_test = np.clip(scores_r2_test, 0.0, 1.0)

    n_experiments = len(scores_mse_train)

    # Create figure for training, validation and testing scores for different features
    fig = plt.figure()

    # Create subplot for MSE for training, validation, testing
    # 1 row, 2 columns, and get 1st subplot in the figure
    ax = fig.add_subplot(1, 2, 1)
    x_values = [range(1, n_experiments + 1)] * n_experiments
    y_values = [scores_mse_train, scores_mse_val, scores_mse_test]
    labels = ['Training', 'Validation', 'Testing']
    colors = ['blue', 'red', 'green']
    # Plot MSE scores for training, validation, testing sets
    for x, y, label, color in zip(x_values, y_values, labels, colors):
        ax.plot(x, y, marker='o', color=color, label=label)
        ax.legend(loc='best')

    # Set y limits between 0 and 50, set x limits to 0 to number experiments + 1
    ax.set_ylim([0.0, 50.0])
    ax.set_xlim([0.0, n_experiments + 1])
    # Set y label to 'MSE', set x label to 'p-degree'
    ax.set_ylabel('MSE')
    ax.set_xlabel('p-degree')

    # Create subplot for R-square for training, validation, testing
    # 1 row, 2 columns, and get 2nd subplot in the figure
    ax = fig.add_subplot(1, 2, 2)
    x_values = [range(1, n_experiments + 1)] * n_experiments
    y_values = [scores_r2_train, scores_r2_val, scores_r2_test]
    labels = ['Training', 'Validation', 'Testing']
    colors = ['blue', 'red', 'green']
    # Plot R-squared scores for training, validation, testing sets
    for x, y, label, color in zip(x_values, y_values, labels, colors):
        ax.plot(x, y, marker='o', color=color, label=label)
        ax.legend(loc='best')

    # Set y limits between 0 and 1, set x limits to 0 to number experiments + 1
    ax.set_ylim(0.0, 1.0)
    ax.set_xlim(0.0, n_experiments + 1)
    # Set y label to 'R-squared', set x label to 'p-degree'
    ax.set_ylabel('R-squared')
    ax.set_xlabel('p-degree')

    # Create super title 'Scikit-learn Polynomial Expansion on Training, Validation and Testing Sets'
    plt.suptitle('Scikit-learn Polynomial Expansion on Training, Validation and Testing Sets')

    '''
    Trains and tests linear regression from scikit-learn with our implementation of polynomial features
    '''
    # Instantiate lists containing the training, validation and testing
    # MSE and R-squared scores obtained from linear regression without nonlinear mapping
    scores_mse_train = [score_mse_train]
    scores_mse_val = [score_mse_val]
    scores_mse_test = [score_mse_test]
    scores_r2_train = [score_r2_train]
    scores_r2_val = [score_r2_val]
    scores_r2_test = [score_r2_test]

    for degree in degrees_polynomial:

        print('Results for LinearRegression model using our implementation of order-{} polynomial expansion features'.format(degree))

        # Transform the data by nonlinear mapping using our implementation of polynomial expansion
        poly_transform = PolynomialFeatureExpansion(degree=degree)

        x_poly_train = poly_transform.transform(x_train)
        x_poly_val = poly_transform.transform(x_val)
        x_poly_test = poly_transform.transform(x_test)

        # Initialize scikit-learn linear regression model
        model_poly = LinearRegression()
        # Trains scikit-learn linear regression model using p-order polynomial expansion
        model_poly.fit(x_poly_train, y_train)

        # Test model on training set
        predictions_poly_train = model_poly.predict(x_poly_train)
        score_mse_poly_train = skmetrics.mean_squared_error(predictions_poly_train, y_train)
        print('Training set mean squared error: {:.4f}'.format(score_mse_poly_train))
        score_r2_poly_train = model_poly.score(x_poly_train, y_train)
        print('Training set r-squared scores: {:.4f}'.format(score_r2_poly_train))
        # Save MSE and R-square scores on training set
        scores_mse_train.append(score_mse_poly_train)
        scores_r2_train.append(score_r2_poly_train)

        # Test model on validation set
        predictions_poly_val = model_poly.predict(x_poly_val)
        score_mse_poly_val = skmetrics.mean_squared_error(predictions_poly_val, y_val)
        print('Validation set mean squared error: {:.4f}'.format(score_mse_poly_val))
        score_r2_poly_val = model_poly.score(x_poly_val, y_val)
        print('Validation set r-squared scores: {:.4f}'.format(score_r2_poly_val))
        # Save MSE and R-square scores on validation set
        scores_mse_val.append(score_mse_poly_val)
        scores_r2_val.append(score_r2_poly_val)

        # Test model on testing set
        predictions_poly_test = model_poly.predict(x_poly_test)
        score_mse_poly_test = skmetrics.mean_squared_error(predictions_poly_test, y_test)
        print('Testing set mean squared error: {:.4f}'.format(score_mse_poly_test))
        score_r2_poly_test = model_poly.score(x_poly_test, y_test)
        print('Testing set r-squared scores: {:.4f}'.format(score_r2_poly_test))
        # Save MSE and R-square scores on testing set
        scores_mse_test.append(score_mse_poly_test)
        scores_r2_test.append(score_r2_poly_test)

    # Convert each scores to NumPy arrays
    scores_mse_train = np.array(scores_mse_train)
    scores_mse_val = np.array(scores_mse_val)
    scores_mse_test = np.array(scores_mse_test)
    scores_r2_train = np.array(scores_r2_train)
    scores_r2_val = np.array(scores_r2_val)
    scores_r2_test = np.array(scores_r2_test)

    # Clip each set of MSE scores between 0 and 50
    scores_mse_train = np.clip(scores_mse_train, 0.0, 50.0)
    scores_mse_val = np.clip(scores_mse_val, 0.0, 50.0)
    scores_mse_test = np.clip(scores_mse_test, 0.0, 50.0)
    # Clip each set of R-squared scores between 0 and 1
    scores_r2_train = np.clip(scores_r2_train, 0.0, 1.0)
    scores_r2_val = np.clip(scores_r2_val, 0.0, 1.0)
    scores_r2_test = np.clip(scores_r2_test, 0.0, 1.0)

    n_experiments = len(scores_mse_train)

    # Create figure for training, validation and testing scores for different features
    fig = plt.figure()

    # Create subplot for MSE for training, validation, testing
    ax = fig.add_subplot(1, 2, 1)
    x_values = [range(1, n_experiments + 1)] * n_experiments
    y_values = [scores_mse_train, scores_mse_val, scores_mse_test]
    labels = ['Training', 'Validation', 'Testing']
    colors = ['blue', 'red', 'green']
    # Plot MSE scores for training, validation, testing sets
    for x, y, label, color in zip(x_values, y_values, labels, colors):
        ax.plot(x, y, marker='o', color=color, label=label)
        ax.legend(loc='best')

    # Set y limits between 0 and 50, set x limits to 0 to number experiments + 1
    ax.set_ylim([0.0, 50.0])
    ax.set_xlim([0.0, n_experiments + 1])
    # Set y label to 'MSE', set x label to 'p-degree'
    ax.set_ylabel('MSE')
    ax.set_xlabel('p-degree')
    
    # Create subplot for R-square for training, validation, testing
    ax = fig.add_subplot(1, 2, 2)
    x_values = [range(1, n_experiments + 1)] * n_experiments
    y_values = [scores_r2_train, scores_r2_val, scores_r2_test]
    labels = ['Training', 'Validation', 'Testing']
    colors = ['blue', 'red', 'green']
    # Plot R-squared scores for training, validation, testing sets
    for x, y, label, color in zip(x_values, y_values, labels, colors):
        ax.plot(x, y, marker='o', color=color, label=label)
        ax.legend(loc='best')

    # Set y limits between 0 and 1, set x limits to 0 to number experiments + 1
    ax.set_ylim(0.0, 1.0)
    ax.set_xlim(0.0, n_experiments + 1)
    # Set y label to 'R-squared', set x label to 'p-degree'
    ax.set_ylabel('R-squared')
    ax.set_xlabel('p-degree')

    # Create super title 'Our Polynomial Expansion on Training, Validation and Testing Sets'
    plt.suptitle('Our Polynomial Expansion on Training, Validation and Testing Sets')

    plt.show()
