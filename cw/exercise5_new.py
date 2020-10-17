import numpy as np
import sklearn.datasets as skdata
import sklearn.metrics as skmetrics
import sklearn.preprocessing as skpreprocess
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt


'''
Name: Doe, John (Please write names in <Last Name, First Name> format)

Collaborators: Doe, Jane (Please write names in <Last Name, First Name> format)

Collaboration details: Discussed <function name> implementation details with Jane Doe.

Summary:

TODO:
(1) Summarize the polynomial feature expansion algorithm for non-linear mapping.

(2) Why do we use this algorithm?
We use this algorithm in order to create a nonlinear mapping for linear features

(3) What negative learning phenonmenon is this algorithm prone to, and why does it happen?
This algorithm is prone to overfitting, which is 
This happens because producing so many extra features gives too many degrees of freedom
more features than number of data points --> more variables than number of constraints
this means that we can create an overly complex function that fits to not just the
data but also the noise 

TODO: Report your scores here. For example,

Results using scikit-learn LinearRegression model with linear features
Training set mean squared error: 23.2560
Training set r-squared scores: 0.7323
Validation set mean squared error: 17.6111
Validation set r-squared scores: 0.7488
Testing set mean squared error: 17.1465
Testing set r-squared scores: 0.7805
Results of LinearRegression model using scikit-learn order-0 polynomial expansion features
Training set mean squared error: 0.0000
Training set r-squared scores: 0.0000
Validation set mean squared error: 0.0000
Validation set r-squared scores: 0.0000
Testing set mean squared error: 0.0000
Testing set r-squared scores: 0.0000
Results of LinearRegression model using scikit-learn order-0 polynomial expansion features
Training set mean squared error: 0.0000
Training set r-squared scores: 0.0000
Validation set mean squared error: 0.0000
Validation set r-squared scores: 0.0000
Testing set mean squared error: 0.0000
Testing set r-squared scores: 0.0000
Results for LinearRegression model using our implementation of order-0 polynomial expansion features
Training set mean squared error: 0.0000
Training set r-squared scores: 0.0000
Validation set mean squared error: 0.0000
Validation set r-squared scores: 0.0000
Testing set mean squared error: 0.0000
Testing set r-squared scores: 0.0000
Results for LinearRegression model using our implementation of order-0 polynomial expansion features
Training set mean squared error: 0.0000
Training set r-squared scores: 0.0000
Validation set mean squared error: 0.0000
Validation set r-squared scores: 0.0000
Testing set mean squared error: 0.0000
Testing set r-squared scores: 0.0000
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

        # List of boolean lists (True, False) to represent which polynomials we will create
        # Examples of polynomials:
        # [
        #   [x1^2, x1x2, x2x1, x2^2]  2nd order
        #   [...]
        # ]
        # Corresponding polynomials terms to create:
        # [
        #   [True, True, False, True]  2nd order
        #   [...]
        # ]
        self.__polynomial_terms = []

    def transform(self, X):
        '''
        Computes up to p-order (degree) polynomial features and augments them to the data

        Args:
            X : numpy
                N x d feature vector

        Returns:
            polynomial expanded features in Z space
        '''

        # Initialize the bias
        # TODO: What is the shape of bias and why do we select this shape?
        bias = np.ones([X.shape[0], 1])

        # Initialize polynomial expansion features Z
        # TODO: Suppose x = [x1, x2], what terms are in Z?
        Z = [bias, X]

        # If degree is less than 2, then return the original features
        if self.__degree < 2:
            Z = np.concatenate(Z, axis=1)
            return Z

        # Split X into it's d dimensions separately
        linear_features = np.split(X, indices_or_sections=X.shape[1], axis=1)

        if self.__degree == 2:
            # Keep a list of new polynomial features that we've accumulated
            new_polynomial_features = []

            # Keep track of the polynomial terms that we will keep
            polynomial_terms = []

            # For every linear feature
            for l1 in range(len(linear_features)):

                # Multiply it by every linear feature
                for l2 in range(len(linear_features)):

                    # TODO: Suppose x = [x_1, x_2]
                    # write the polynomial terms after each iteration
                    # for 2nd order polynomial

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
                        # l1 * len(linear_features) + l2 indexes into the term we are creating
                        keep_polynomial_term = self.__polynomial_terms[0][l1 * len(linear_features) + l2]

                        if keep_polynomial_term:
                            # And append the result to the new set of polynomial features
                            new_polynomial_features.append(polynomial_feature)

            # If we've never processed the polynomial terms before, save the list of terms to keep
            if len(self.__polynomial_terms) < self.__degree - 1:
                self.__polynomial_terms.append(polynomial_terms)

            # Add the new polynomial features to Z
            Z.append(np.concatenate(new_polynomial_features, axis=1))

        if self.__degree > 2:
            # Start off with X as both the set of linear and current polynomial features
            linear_features = np.split(X, indices_or_sections=X.shape[1], axis=1)
            current_polynomial_features = linear_features

            # Since we will be taking the difference of the sum of features at every
            # iteration of the inner loop, let's compute their sums first to save compute
            sum_Z_features = [
                np.sum(f) for f in linear_features
            ]

            # For every degree expansion
            for d in range(0, self.__degree - 1):
                # Initialize a list to hold the new polynomial features
                new_polynomial_features = []

                # Keep track of the polynomial terms that we will keep
                polynomial_terms = []

                # Since expanding a polynomial (x1 + x2)^2 to a higher order (x1 + x2)^3 is just
                # multiplying by linear terms e.g. (x1 + x2)^3 = (x1 + x2)^2 (x1 + x2)
                # we treat the outer loop as the current polynomial term (x1 + x2)^2
                # that we are processing

                # For every polynomial feature
                for p in range(len(current_polynomial_features)):

                    # Multiply it by every linear feature
                    for l in range(len(linear_features)):

                        # TODO: Suppose x = [x_1, x_2]
                        # write the polynomial terms after each iteration
                        # for 3rd order polynomial

                        polynomial_feature = current_polynomial_features[p] * linear_features[l]

                        # Check if we have already found the polynomial terms to keep
                        if len(self.__polynomial_terms) < self.__degree - 1:

                            # If we have not, then iterate through the expansion
                            keep_polynomial_term  = True

                            # Check if we already have the feature created
                            # To save some compute sum this once before going into loop
                            sum_polynomial_feature = np.sum(polynomial_feature)

                            for sum_Z_feature in sum_Z_features:
                                # We check if the absolute difference of the sums is less than a small epsilon
                                if np.abs(sum_polynomial_feature - sum_Z_feature) < 1e-9:
                                    keep_polynomial_term = False
                                    break

                            # Keep track of whether we keep or discard (True/False) the term
                            polynomial_terms.append(keep_polynomial_term)

                            if keep_polynomial_term:
                                # And append the result to the new set of polynomial features
                                new_polynomial_features.append(polynomial_feature)
                                sum_Z_features.append(sum_polynomial_feature)

                        else:
                            # Check if the current polynomial term was kept
                            # p * len(linear_features) + l indexes into the term we are creating
                            # TODO: What is d referring to?

                            # TODO: For third degree expansion of x = [x1, x2],
                            # What terms are we indexing to if we just use p * len(linear_features) instead?

                            keep_polynomial_term = self.__polynomial_terms[d][p * len(linear_features) + l]

                            if keep_polynomial_term:
                                # And append the result to the new set of polynomial features
                                new_polynomial_features.append(polynomial_feature)

                # If we've never processed the polynomial terms before, save the list of terms to keep
                if len(self.__polynomial_terms) < self.__degree - 1:
                    self.__polynomial_terms.append(polynomial_terms)

                # Add the new polynomial features to Z
                # TODO: Why do we concatenate along the 1st axis?

                Z.append(np.concatenate(new_polynomial_features, axis=1))

                # TODO: For 3rd order polynomial expansion, what does Z contain after
                # each iteration of the outer for loop (d)

                # Set the new polynomial features to curren polynomial features and repeat
                current_polynomial_features = new_polynomial_features

        # Concatenate every term into the feature vector
        Z = np.concatenate(Z, axis=1)

        return Z


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


def clip_scores(scores):
    '''
    Helper function that prepares the score numpy arrays for plotting by
    clipping the MSE values between 0 and 50 and clipping the R2 values 
    between 0 and 1. 
    
    Args:
        scores: a dictionary containing two dictionaries, one for the 
                numpy arrays of MSE scores and another for the numpy
                arrays of R2 scores.
    '''
    # Clip each set of MSE scores between 0 and 50
    for test_set in scores["mse"]:
        scores["mse"][test_set] = np.clip(scores["mse"][test_set], 0.0, 50.0)
    # Clip each set of R-squared scores between 0 and 1
    for test_set in scores["r2"]:
        scores["mse"][test_set] = np.clip(scores["mse"][test_set], 0.0, 50.0)


def set_up_subplot(ax, scores, score_type):
    n_experiments = len(scores[score_type]["train"])
    x_values = [range(1, n_experiments + 1)] * n_experiments
    y_values = scores[score_type].values()
    labels = ['Training', 'Validation', 'Testing']
    colors = ['blue', 'red', 'green']
    # Plot MSE or R2 scores for training, validation, testing sets
    for x, y, label, color in zip(x_values, y_values, labels, colors):
        ax.plot(x, y, marker='o', color=color, label=label)
        ax.legend(loc='best')
    ax.set_xlim([0.0, n_experiments + 1])
    if score_type == "mse":
        ax.set_ylim([0.0, 50.0])
    elif score_type == "r2":
        ax.set_ylim([0.0, 1.0])
    else: 
        raise ValueError("Unexpected score_type")
    # Set y label to 'MSE' or 'R2' depending on score_type
    ax.set_ylabel(score_type.upper())
    # Set x label to 'p-degree'
    ax.set_xlabel('p-degree')


if __name__ == '__main__':

    '''
    Load Boston Housing data and split into train, val, test
    '''
    boston_housing_data = skdata.load_boston()
    x = boston_housing_data.data
    y = boston_housing_data.target

    # 80 percent train, 10 percent validation, 10 percent test split
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

    # Test model on training set and save MSE and R-square scores
    score_mse_train, score_r2_train = score_model_on_data(model, x_train, y_train)
    scores_mse_train.append(score_mse_train)
    scores_r2_train.append(score_r2_train)
    print('Training set mean squared error: {:.4f}'.format(score_mse_train))
    print('Training set r-squared scores: {:.4f}'.format(score_r2_train))

    # Test model on validation set and save MSE and R-square scores
    score_mse_val, score_r2_val = score_model_on_data(model, x_val, y_val)
    scores_mse_val.append(score_mse_val)
    scores_r2_val.append(score_r2_val)
    print('Validation set mean squared error: {:.4f}'.format(score_mse_val))
    print('Validation set r-squared scores: {:.4f}'.format(score_r2_val))

    # Test model on testing set and save MSE and R-square scores
    score_mse_test, score_r2_test = score_model_on_data(model, x_test, y_test)
    scores_mse_test.append(score_mse_test)
    scores_r2_test.append(score_r2_test)
    print('Testing set mean squared error: {:.4f}'.format(score_mse_test))
    print('Testing set r-squared scores: {:.4f}'.format(score_r2_test))

    # Set the degrees/orders of polynomials to be 2 and 3 for nonlinear mapping
    degrees_polynomial = [2, 3]

    '''
    Trains and tests linear regression from scikit-learn with scikit-learn polynomial features
    '''
    for degree in degrees_polynomial:

        print('Results of LinearRegression model using scikit-learn order-{} polynomial expansion features'.format(degree))

        # Initialize polynomial expansion
        poly_transform = skpreprocess.PolynomialFeatures(degree=degree)

        # TODO: Compute the polynomial terms needed for the data
        # Generates x_1^2, x_1 x_2, x_1 x_3, ..., x_d^2
        poly_transform.fit(x_train)

        # TODO: Transform the data by nonlinear mapping
        # Applies all the polynomial terms to the data and augments it to x
        # Computes the values for x_0, x_1, x_2, ..., x_1^2, x_1 x_2, ... x_d^2
        # x_1 = 2, x_2 = 4 : x -> (1, 2, 4, ..., 4, 8, ..., x_d^2)
        x_poly_train = poly_transform.transform(x_train)
        x_poly_val = poly_transform.transform(x_val)
        x_poly_test = poly_transform.transform(x_test)

        # Initialize scikit-learn linear regression model
        model_poly = LinearRegression()
        # Trains scikit-learn linear regression model using p-order polynomial expansion
        print('Features after polynomial transform order-{}: {}'.format(degree, x_poly_train.shape[1]))
        model_poly.fit(x_poly_train, y_train)

        # Test model on training set and save MSE and R-square scores
        score_mse_poly_train, score_r2_poly_train = score_model_on_data(model_poly, x_poly_train, y_train)
        scores_mse_train.append(score_mse_poly_train)
        scores_r2_train.append(score_r2_poly_train)
        print('Training set mean squared error: {:.4f}'.format(score_mse_poly_train))
        print('Training set r-squared scores: {:.4f}'.format(score_r2_poly_train))

        # Test model on validation set and save MSE and R-square scores
        score_mse_poly_val, score_r2_poly_val = score_model_on_data(model_poly, x_poly_val, y_val)
        scores_mse_val.append(score_mse_poly_val)
        scores_r2_val.append(score_r2_poly_val)
        print('Validation set mean squared error: {:.4f}'.format(score_mse_poly_val))
        print('Validation set r-squared scores: {:.4f}'.format(score_r2_poly_val))

        # Test model on testing set and save MSE and R-square scores
        score_mse_poly_test, score_r2_poly_test = score_model_on_data(model_poly, x_poly_test, y_test)
        scores_mse_test.append(score_mse_poly_test)
        scores_r2_test.append(score_r2_poly_test)
        print('Testing set mean squared error: {:.4f}'.format(score_mse_poly_test))
        print('Testing set r-squared scores: {:.4f}'.format(score_r2_poly_test))

    # TODO: Convert each scores to NumPy arrays
    scores = {
        "mse": {
            "train": np.array(scores_mse_train),
            "val": np.array(scores_mse_val),
            "test": np.array(scores_mse_test)
        },
        "r2": {
            "train": np.array(scores_r2_train),
            "val": np.array(scores_r2_val),
            "test": np.array(scores_r2_test)
        }
    }
    clip_scores(scores)


    # scores_mse_train = np.array(scores_mse_train)
    # scores_mse_val = np.array(scores_mse_val)
    # scores_mse_test = np.array(scores_mse_test)
    # scores_r2_train = np.array(scores_r2_train)
    # scores_r2_val = np.array(scores_r2_val)
    # scores_r2_test = np.array(scores_r2_test)

    # scores_mse_train = np.clip(scores_mse_train, 0.0, 50.0)
    # scores_mse_val = np.clip(scores_mse_val, 0.0, 50.0)
    # scores_mse_test = np.clip(scores_mse_test, 0.0, 50.0)

    # scores_r2_train = np.clip(scores_r2_train, 0.0, 1.0)
    # scores_r2_val = np.clip(scores_r2_val, 0.0, 1.0)
    # scores_r2_test = np.clip(scores_r2_test, 0.0, 1.0)

    # n_experiments = len(scores_mse_train)
    # n_experiments = len(scores["mse"]["train"])

    # Create figure for training, validation and testing scores for different features
    fig = plt.figure()



    # Create subplot for MSE for training, validation, testing
    # 1 row, 2 columns, and get 1st subplot in the figure
    ax = fig.add_subplot(1, 2, 1)
    # x_values = [range(1, n_experiments + 1)] * n_experiments
    # # y_values = [scores_mse_train, scores_mse_val, scores_mse_test]
    # y_values = [scores["mse"].values()]
    # labels = ['Training', 'Validation', 'Testing']
    # colors = ['blue', 'red', 'green']

    # # Plot MSE scores for training, validation, testing sets
    # for x, y, label, color in zip(x_values, y_values, labels, colors):
    #     ax.plot(x, y, marker='o', color=color, label=label)
    #     ax.legend(loc='best')

    # # TODO: Set y limits between 0 and 50, set x limits to 0 to number experiments + 1
    # ax.set_ylim([0.0, 50.0])
    # ax.set_xlim([0.0, n_experiments + 1])

    # # TODO: Set y label to 'MSE', set x label to 'p-degree'
    # ax.set_ylabel('MSE')
    # ax.set_xlabel('p-degree')
    set_up_subplot(ax, scores, 'mse')

    # TODO: Create subplot for R-square for training, validation, testing


    x_values = [[0], [0], [0]]
    y_values = [[0], [0], [0]]
    labels = ['Training', 'Validation', 'Testing']
    colors = ['blue', 'red', 'green']

    # TODO: Plot R-squared scores for training, validation, testing sets

    # TODO: Set y limits between 0 and 1, set x limits to 0 to number experiments + 1

    # TODO: Set y label to 'R-squared', set x label to 'p-degree'

    # TODO: Create super title 'Scikit-learn Polynomial Expansion on Training, Validation and Testing Sets'
    plt.suptitle('Scikit-learn Polynomial Expansion on Training, Validation and Testing Sets')

    '''
    Trains and tests linear regression from scikit-learn with our implementation of polynomial features
    '''
    # TODO: Instantiate lists containing the training, validation and testing
    # MSE and R-squared scores obtained from linear regression without nonlinear mapping
    scores_mse_train = [0]
    scores_mse_val = [0]
    scores_mse_test = [0]
    scores_r2_train = [0]
    scores_r2_val = [0]
    scores_r2_test = [0]

    for degree in degrees_polynomial:

        print('Results for LinearRegression model using our implementation of order-{} polynomial expansion features'.format(degree))

        # TODO: Transform the data by nonlinear mapping using our implementation of polynomial expansion
        poly_transform = PolynomialFeatureExpansion(degree=degree)

        # Transform x_train to x_poly_train with p-degree expansion
        x_poly_train = poly_transform.transform(x_train)

        # Initialize scikit-learn linear regression model
        model_poly = LinearRegression()

        # TODO: Trains scikit-learn linear regression model using p-order polynomial expansion
        print('Features after polynomial transform order-{}: {}'.format(degree, x_poly_train.shape[1]))
        model_poly.fit(x_poly_train, y_train)

        # TODO: Test model on training set

        score_mse_poly_train = 0.0
        print('Training set mean squared error: {:.4f}'.format(score_mse_poly_train))

        score_r2_poly_train = model_poly.score(x_poly_train, y_train)
        print('Training set r-squared scores: {:.4f}'.format(score_r2_poly_train))

        # TODO: Save MSE and R-square scores on training set

        # TODO: Test model on validation set

        score_mse_poly_val = 0.0
        print('Validation set mean squared error: {:.4f}'.format(score_mse_poly_val))

        score_r2_poly_val = 0.0
        print('Validation set r-squared scores: {:.4f}'.format(score_r2_poly_val))

        # TODO: Save MSE and R-square scores on validation set

        # TODO: Test model on testing set

        score_mse_poly_test = 0.0
        print('Testing set mean squared error: {:.4f}'.format(score_mse_poly_test))

        score_r2_poly_test = 0.0
        print('Testing set r-squared scores: {:.4f}'.format(score_r2_poly_test))

        # TODO: Save MSE and R-square scores on testing set

    # TODO: Convert each scores to NumPy arrays

    # TODO: Clip each set of MSE scores between 0 and 50

    # TODO: Clip each set of R-squared scores between 0 and 1

    n_experiments = len(scores_mse_train)

    # TODO: Create figure for training, validation and testing scores for different features

    # TODO: Create subplot for MSE for training, validation, testing

    x_values = [[0], [0], [0]]
    y_values = [[0], [0], [0]]
    labels = ['Training', 'Validation', 'Testing']
    colors = ['blue', 'red', 'green']

    # TODO: Plot MSE scores for training, validation, testing sets

    # TODO: Set y limits between 0 and 50, set x limits to 0 to number experiments + 1

    # TODO: Set y label to 'MSE', set x label to 'p-degree'

    # TODO: Create subplot for R-square for training, validation, testing

    x_values = [[0], [0], [0]]
    y_values = [[0], [0], [0]]
    labels = ['Training', 'Validation', 'Testing']
    colors = ['blue', 'red', 'green']

    # TODO: Plot R-squared scores for training, validation, testing sets

    # TODO: Set y limits between 0 and 1, set x limits to 0 to number experiments + 1

    # TODO: Set y label to 'R-squared', set x label to 'p-degree'

    # TODO: Create super title 'Our Polynomial Expansion on Training, Validation and Testing Sets'

    plt.show()
