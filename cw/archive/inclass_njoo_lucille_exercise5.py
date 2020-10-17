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
In this exercise, we implemented a PolynomialFeatureExpansion class that we use to add polynomial
features, up to some degree, to the given data, in order to do nonlinear mapping. This class performs 
the feature expansion by multiplying the linear features by themselves, then multiplying the resulting 
polynomial expansion by the linear features iteratively to generate polynomial features of increasing
orders. These features get appended to the original data until we have created all the polynomial 
features.

In the main method, we then commpared our PolynomialFeatureExpansion class to sklearn's PolynomialFeatures 
class by using both to create expanded data that we use to train, validate, and test an sklearn 
LinearRegression model. We plotted the MSE and r-squared scores of the models on both sklearn's 
PolynomialFeatures data and on our own PolynomialFeatureExpansion data, as a function of the degree 
polynomial we expanded to. 

Results:

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
Training set mean squared error: 0.0000
Training set r-squared scores: 1.0000
Validation set mean squared error: 168340.1784
Validation set r-squared scores: -2400.4241
Testing set mean squared error: 128662.1554
Testing set r-squared scores: -1646.3980
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
        # If degree = 2, x = [x1, x2] (linear)
        # (x1 + x2)^2 = (x1 + x2) (x1 + x2) = (x1^2 + x1x2 + x2x1 + x2^2)
        # We only care about the *unique* terms: x1^2, x1x2, and x2^2
        # Our polynomial expansion of degree 2 will be:
        # Z = [
        #   [x0, x1, x2],
        #   [x1^2, x1x2, x2^2]
        # ]
        # Given a degree, we want to create this Z
        self.__degree = degree

        # Polynomial terms that we will use based on the specified degree
        # List of Boolean lists (True or False); each internal list contains 
        # booleans representing whether the corresponding term of the polynomial 
        # expansion should be kept (if it is unique) or discarded (if it is 
        # a repeat of a term that was already computed)
        # Example: for degree 2, we generate 4 terms:
        #   (x1 + x2)^2 = (x1^2 + x1x2 + x2x1 + x2^2)
        # There are only 3 unique values, and x2x1 is a repeat, so:
        #   For new polynomials:
        #   [
        #     [x1^2, x1x2, x2x1, x2^2]  2nd order
        #     [ ... ]                   3rd order
        #   ]
        #   Polynomial terms to create:
        #   [
        #     [True, True, False, True]  2nd order
        #     [ ... ]                    3nd order
        #   ]
        # Goal is to populate self.__polynomial_terms
        self.__polynomial_terms = []


    def __check_feature_exists(self, feature, existing_features):
        '''
        Helper function for checking if feature is in the list of existing features

        Args:
            feature : numpy
                N x 1 feature vector
            existing_features: list[numpy]
                list containing N x 1 features

        Returns:
            boolean: True if feature exists, else False
        '''
        # Iterate through existing features and check if feature already exists
        for existing_feature in existing_features:
            # don't use feature == existing_features because they might be in 
            # different memory locations
            if np.abs(np.sum(feature - existing_features)) < 1e-9:
                return True
        # If the feature never matched any existing ones
        return False        

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
        # Initialize some value for the bias (x0)
        # Will be (N x 1) number of ones
        # TODO: Why should bias be (N, 1)?
        N = X.shape[0]
        bias = np.ones([N, 1])
        # Augment the bias term to x
        # Let's use the notation that we used in the slides:
        # Let X be x1, x2 for simplicity
        # Z = [
        #   [x0, X]
        # ]
        # which means that 
        # Z = [
        #   [x0, x1, x2]
        # ]
        # Let's keep this as a list for now, and then concatenate them all together when we are done
        Z = [bias, X]

        # In the case where nothing is changed or degree is less than 2
        if self.__degree < 2:
            # Return the original features with the bias attached
            # TODO: Why do we concatenate along axis=1?
            Z = np.concatenate(Z, axis=1)
            return Z

        if self.__degree == 2:
            # Consider the example of x = [x1, x2] as our original feature vector
            # Algorithm to expand the polynomial:
            # Multiply every term in the linear features by every term in the linear features
            # For x = [x1, x2]
            # (x1 + x2)^2 = x1x1 + x1x2 + x2x1 + x2x2
            #             = x1^2 + x1x2 + x2^2
            # phi_2(x) = x0, x1, x2, x1^2, x1x2, x2^2

            # We want to process each feature individualy
            # so we will first split X up into is individual 
            # x1, x2, ..., xd
            # in the x = [x1, x2], we should get 2 terms [x1, x2]
            # X.shape[1] is d, so we split X into d parts along axis=1
            # This will get us the d features: x1, x2, ..., xd
            d = X.shape[1]
            linear_features = np.split(X, indices_or_sections=d, axis=1)
                
            # Initialize a list to hold the new polynomial features that we have created
            new_polynomial_features = []            

            # TODO: Why do we write l1 from 0 up to the length of linear features?
            for l1 in range((len(linear_features))):

                # TODO: Why do we write l1 from 0 up to the length of linear features?
                for l2 in range(len(linear_features)):
                    # TODO: Write out step by step each multiplication for x = [x1, x2]
                    # for 2nd order polynomial expansion
                    # l1=0, l2=0: x1 x1 = x1^2
                    # l1=0, l2=1: x1 x2 = x1x2
                    # l1=1, l2=0: x2 x1 = x2x1
                    # l1=1, l2=1: x2 x2 = x2^2

                    # Multiply every term in the linear features by every term in the linear features
                    # This creats an N x 1 feature
                    polynomial_feature = linear_features[l1] * linear_features[l2]
                    
                    # We will get x1x1, x1x2, x2x1, x2x2
                    # We need a list to hold x1x1, x1x2, x2x1, x2x2
                    # We initialized that list outside the for loops

                    # Check if the polynomial feature has already been created, e.g. x1x2 and x2x1
                    feature_exists = self.__check_feature_exists(feature=polynomial_feature, existing_features=new_polynomial_features)
                    
                    # If the feature already exists, then don't create it
                    if feature_exists: 
                        # Set polynomial terms to be False so later on we know that we won't have to create it
                        self.__polynomial_terms.append(False)
                    else:
                        # Set to be True so we know to create the feature next time
                        self.__polynomial_terms.append(True)
                        # Keep track of the polynomial feature we just created by adding to the list
                        new_polynomial_features.append(polynomial_feature)

            # End up with all the polynomial features for 2nd order
            # Concatenate the list of polynomial features together into a (N, D-d-1) vector
            # D is all the features together; d is the number of original features we had
            # D - d - 1 gives us the number of new polynomial features (because we had a bias in there too)
            # TODO: Why do we need to concatenate?
            # It's because new_polynomial_features contanis a list of polynomial features, and each 
            # polynomial feature is its own separate vector of shape N x 1. There are d number of them. 
            # We want to put them all back into one matrix so that we can append it to our running list Z 
            # and concatenate the matrices at the very end.
            second_order_polynomial_features = np.concatenate(new_polynomial_features, axis=1)

            # Append the second order polynomial features to the list of features
            # TODO: What does Z contain after you append the features? 
            Z.append(second_order_polynomial_features)
            # Z = [ 
            #   bias,
            #   X,
            #   new_plynomial_features
            # ]

            Z = np.concatenate(Z, axis=1)

        if self.__degree > 2: 
            # (x1 + x2)^4 = (x1 + x2) (x1 + x2) (x1 + x2) (x1 + x2) 
            #             = (x1 + x2)^3 (x1 + x2) 
            # (x1 + x2)^3 = (x1 + x2) (x1 + x2) (x1 + x2) 
            #             = (x1 + x2)^2 (x1 + x2) 
            # (x1 + x2)^2 = (x1 + x2) (x1 + x2)
            
            # We want individual features to loop through each for multiplication
            d = X.shape[1]
            linear_features = np.split(X, indices_or_sections=d, axis=1)
            
            # Initialize current polynomial features
            current_polynomial_features = linear_features

            # TODO: Why do we choose degree-1?
            for d in range(0, self.__degree - 1):

                polynomial_terms = []

                # We need to keep track of the polynomial features we have created so far
                new_polynomial_features = []

                for p in range(len(current_polynomial_features)):

                    for l in range(len(linear_features)):

                        # Multiply polynomial feature by linear feature
                        polynomial_feature = current_polynomial_features[p] * linear_features[l]

                        # Polynomial terms is going to be a list of 0-elements unless we add in the True/False
                        if len(self.__polynomial_terms) < self.__degree - 1:
                            # We know we haven't processed at all yet, so we want to append/generate self.__polynomial_terms
                            feature_exists = self.__check_feature_exists(feature=polynomial_feature, existing_features=new_polynomial_features)
                            if feature_exists:
                                polynomial_terms.append(False)
                            else:
                                polynomial_terms.append(True)
                                new_polynomial_features.append(polynomial_feature)

                        else:
                            # If we have already processed which terms need to be created and which don't, 
                            # check self.__polynomial terms at element d
                            # The index p * len(linear_features) + l is the ndex of the current term computed 
                            # for the corresponding list of polynomials that we want
                            if self.__polynomial_terms[d][p * len(linear_features) + l]:
                                new_polynomial_features.append(polynomial_feature)

                if len(self.__polynomial_terms) < self.__degree - 1:
                    self.__polynomial_terms.append(polynomial_terms)

                # Concatenate polynomial features together
                current_polynomial_features = new_polynomial_features
                d_order_polynomial_features = np.concatenate(new_polynomial_features, axis=1)
                
                Z.append(d_order_polynomial_features)
            
            Z = np.concatenae(Z, axis=1)

        return Z

def split_data(x, y):
    '''
    Splits raw data x and y into training, validation, and testing sets, using an 
    80 - 10 - 10 split.

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


def convert_and_clip_scores(mse_arrays, r2_arrays):
    '''
    Helper function that prepares arrays for plotting. This function converts 
    all given scores arrays to numpy arrays, clips MSE values between 0 and 50,
    and clips R2 values between 0 and 1. 
    
    Args:
        mse_arrays: array of 3 arrays, containing scores_mse_train, scores_mse_val, and scores_mse_test
        r2_arrays: array of 3 arrays, containing scores_r2_train, scores_r2_val, scores_r2_test

    Returns:
        tuple of 6 numpy arrays, in the same order as above
    '''
    for i in range(len(mse_arrays)):
        mse_arrays[i] = np.array(mse_arrays[i])
        mse_arrays[i] = np.clip(mse_arrays[i], 0.0, 50.0)
    for i in range(len(r2_arrays)):
        r2_arrays[i] = np.array(r2_arrays[i])
        r2_arrays[i] = np.clip(r2_arrays[i], 0.0, 1.0)
    return (*mse_arrays, *r2_arrays)
    

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
    # degrees_polynomial = [2]

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
        print("Features after polynomial transform order={}: {}".format(degree, x_poly_train.shape[1]))
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

    # Prepare all the scores arrays for plotting in the following graphs
    mse_scores = [scores_mse_train, scores_mse_val, scores_mse_test]
    r2_scores = [scores_r2_train, scores_r2_val, scores_r2_test]
    scores_mse_train, scores_mse_val, scores_mse_test, \
        scores_r2_train, scores_r2_val, scores_r2_test = convert_and_clip_scores(mse_scores, r2_scores)

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
        print("Features after polynomial transform order={}: {}".format(degree, x_poly_train.shape[1]))
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

    # Prepare all the scores arrays for plotting in the following graphs
    mse_scores = [scores_mse_train, scores_mse_val, scores_mse_test]
    r2_scores = [scores_r2_train, scores_r2_val, scores_r2_test]
    scores_mse_train, scores_mse_val, scores_mse_test, \
        scores_r2_train, scores_r2_val, scores_r2_test = convert_and_clip_scores(mse_scores, r2_scores)

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
