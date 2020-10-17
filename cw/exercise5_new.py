import numpy as np
import sklearn.datasets as skdata
import sklearn.metrics as skmetrics
import sklearn.preprocessing as skpreprocess
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt


'''
Name: Njoo, Lucille

Collaborators: N/A

Collaboration details: N/A

Summary:

(1) Summarize the polynomial feature expansion algorithm for non-linear mapping.

We start with some d features in X, which we can denote with (x_1, ... , x_d). Our goal is 
to generate our expanded features Z, which will contain a total of D features. 
- First, we add a bias term x_0, which we set to 1. When we augment this bias to X, this 
  makes Z now contain (x_0, x_1, ... , x_d).
- Next, we augment new polynomial terms up to order p by iteratively generating increasing 
  polynomial degrees, augmenting them to Z, and multiplying these polynomials by the linear 
  features until we have reached degree p. To generate these terms, we do the following:
    - We initialize a variable to hold onto the current polynomial; it starts as simply the 
      linear features. 
    - For every degree up to p, the degree we want to expand to:
        - We multiply every term in the current polynomial by every term in the linear 
          features. Each one of these terms is one new polynomial feature. If the newly 
          generated feature is not unique, i.e. it is a repeat of a feature that we've 
          already computed, we ignore it; otherwise, we keep it.
        - We augment all the new polynomial features we generated to our growing matrix of 
          features.
        - We update the current polynomial to be the polynomial we just generated.
- Once we have added all the new polynomial features, we return Z, which now looks like: 
  (x_0, x_1, ... , x_d, x_1^p, ... x_d^p).
        

(2) Why do we use this algorithm?

We use the polynomial feature expansion algorithm in order to create a more complex 
hypothesis so that we can better fit to data that is not linearly separable. When we have 
data that is not linearly separable, we cannot use a linear hyperplane to separate the data 
well using only the original d features, so we use nonlinear features to create a nonlinear 
hypothesis function. The polynomial feature expansion algorithm is used to generate these 
additional nonlinear features by computing every unique product of features up to some 
polynomial degree p. In other words, this algorithm computes all the terms in the nonlinear 
mapping function Z = phi_p(x). We can then train our model on the D new, higher-order, 
higher-dimensional features. 

(3) What negative learning phenonmenon is this algorithm prone to, and why does it happen?

This algorithm is prone to overfitting, which causes our training loss to be very small but 
our testing loss to be very large. Overfitting happens easily with nonlinear mapping 
because producing so many extra features results in too many degrees of freedom; if we end 
up with more features than number of data points, this means there are more variables than 
the number of constraints. Since our model tries to minimize loss, it uses its full 
capacity, and as a result, we can create an overly complex function that fits to not just 
the data, but also to the noise, and thus is not a good representation of the true 
function. Such a model will be very accurate on the training data, but it will fail to 
generalize to unseen data during testing. 


Report your scores here:

Results using scikit-learn LinearRegression model with linear features
Training set mean squared error: 23.2560
Training set r-squared scores: 0.7323
Validation set mean squared error: 17.6111
Validation set r-squared scores: 0.7488
Testing set mean squared error: 17.1465
Testing set r-squared scores: 0.7805
Results of LinearRegression model using scikit-learn order-2 polynomial expansion features
Features after polynomial transform order-2: 105
Training set mean squared error: 8.8948
Training set r-squared scores: 0.8976
Validation set mean squared error: 11.4985
Validation set r-squared scores: 0.8360
Testing set mean squared error: 34.8401
Testing set r-squared scores: 0.5539
Results of LinearRegression model using scikit-learn order-3 polynomial expansion features
Features after polynomial transform order-3: 560
Training set mean squared error: 0.0000
Training set r-squared scores: 1.0000
Validation set mean squared error: 131227.1239
Validation set r-squared scores: -1870.9951
Testing set mean squared error: 119705.3692
Testing set r-squared scores: -1531.7148
Results for LinearRegression model using our implementation of order-2 polynomial expansion features
Features after polynomial transform order-2: 105
Training set mean squared error: 8.8948
Training set r-squared scores: 0.8976
Validation set mean squared error: 11.4985
Validation set r-squared scores: 0.8360
Testing set mean squared error: 34.8401
Testing set r-squared scores: 0.5539
Results for LinearRegression model using our implementation of order-3 polynomial expansion features
Features after polynomial transform order-3: 560
Training set mean squared error: 0.0000
Training set r-squared scores: 1.0000
Validation set mean squared error: 131094.5347
Validation set r-squared scores: -1869.1036
Testing set mean squared error: 119633.3746
Testing set r-squared scores: -1530.7930
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
        # Q: What is the shape of bias and why do we select this shape?
        # A: The bias is of shape (N x 1) because we want to add one new feature (our x_0 
        #    term) to the features for each of the N data samples, so we want N rows each with 
        #    1 column. 
        bias = np.ones([X.shape[0], 1])

        # Initialize polynomial expansion features Z
        # Q: Suppose x = [x1, x2], what terms are in Z?
        # A: Z now contains [x0, X].
        #    Technically, Z is still a list of lists because we haven't concatenated everything 
        #    together yet, so at the moment Z looks like: [[x0], [x1, x2]]. 
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

                    # Q: Suppose x = [x_1, x_2]
                    #    write the polynomial terms after each iteration
                    #    for 2nd order polynomial
                    # A: Iteration:      New polynomial term:
                    #     l1=0, l2=0:     x1 x1 = x1^2
                    #     l1=0, l2=1:     x1 x2 = x1x2
                    #     l1=1, l2=0:     x2 x1 = x2x1
                    #     l1=1, l2=1:     x2 x2 = x2^2

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

                        # Q: Suppose x = [x_1, x_2]
                        #    write the polynomial terms after each iteration
                        #    for 3rd order polynomial
                        # A: Iteration:      New polynomial term:
                        #     d=0, p=0, l=0:     x1 x1 = x1^2
                        #     d=0, p=0, l=1:     x1 x2 = x1x2
                        #     d=0, p=1, l=0:     x2 x1 = x2x1
                        #     d=0, p=1, l=1:     x2 x2 = x2^2
                        #     d=1, p=0, l=0:     x1^2 x1 = x1^3
                        #     d=1, p=0, l=1:     x1^2 x2 = x1^2x2
                        #     d=1, p=1, l=0:     x1x2 x1 = x1^2x2
                        #     d=1, p=1, l=1:     x1x2 x2 = x1x2^2
                        #     d=1, p=2, l=0:     x2^2 x1 = x1x2^2
                        #     d=1, p=2, l=1:     x2^2 x2 = x2^3
                        
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
                            # Q: What is d referring to?
                            # A: d refers to (the current polynomial expansion degree - 2); we 
                            #    use it to index into self.__polynomial_terms because self.
                            #    __polynomial_terms is a list of lists, where the inner lists 
                            #    represent which terms of the polynomial expansion are unique 
                            #    and thus should be kept, and which ones are repeats and thus 
                            #    shouldn't be kept. Since d starts at 0 and goes up to (self.
                            #    __degree - 1), and d=0 generates the degree-2 polynomial, that 
                            #    means that self.__polynomial_terms at index 0 will give us the 
                            #    list of booleans for degree 2; index 1 will give us the list 
                            #    of booleans at index 3; etc.

                            # Q: For third degree expansion of x = [x1, x2],
                            #    What terms are we indexing to if we just use p * len
                            #    (linear_features) instead?
                            # A: If we index to (p * len(linear_features)), we end up at the 
                            #    boolean that corresponds to the term produced by (the 
                            #    current pth-term of the polynomial * the first linear feature, 
                            #    x1). So if we're expanding to degree 3, these are the 
                            #    corresponding terms we'd end up at if we indexed to (p* len
                            #    (linear_features)):
                            #       d=0, p=0: x1 * x1 = x1^2
                            #       d=0, p=1: x2 * x1 = x1x2
                            #       d=1, p=0: x1^2 * x1 = x1^3
                            #       d=1, p=1: x1x2 * x1 = x1^2x2
                            #       d=1, p=2: x2^2 * x1 = x1x2^2

                            keep_polynomial_term = self.__polynomial_terms[d][p * len(linear_features) + l]

                            if keep_polynomial_term:
                                # And append the result to the new set of polynomial features
                                new_polynomial_features.append(polynomial_feature)

                # If we've never processed the polynomial terms before, save the list of terms to keep
                if len(self.__polynomial_terms) < self.__degree - 1:
                    self.__polynomial_terms.append(polynomial_terms)

                # Add the new polynomial features to Z
                # Q: Why do we concatenate along the 1st axis?
                # A: We concatenate along the 1st axis because we want to create a matrix with 
                #    N rows and len(new_polynomial_features) columns. Each new 
                #    polynomial_feature is a vector of N elements containing the values of that 
                #    feature for all N data points. We collect all our new polynomial_features 
                #    into the list of new_polynomial features, and we want to concatenate all 
                #    of them together so that there are N rows for each of the N data points, 
                #    and each row contains the new features for that data point, which means we 
                #    need to concatenate along the features axis, or axis=1. We want this shape 
                #    because at the very end we will be concatenating all the elements of Z 
                #    together, and Z needs to be in the shape (N x D), so we want the matrices 
                #    for each polynomial expansion to have N rows.  
                
                Z.append(np.concatenate(new_polynomial_features, axis=1))

                # Q: For 3rd order polynomial expansion, what does Z contain after
                #    each iteration of the outer for loop (d)
                # A: For a 3rd order polynomial expansion, we will have two loops where d=0 and 
                #    then d=1:
                #      d=0: Z=[[x0], [x1, x2], [x1^2, x1x2, x2^2]]
                #      d=1: Z=[[x0], [x1, x2], [x1^2, x1x2, x2^2], [x1^3, x1^2x2, x1x2^2, x2^3]]

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
        scores: dict
            a dictionary containing two dictionaries, one for the numpy arrays of MSE scores 
            and another for the numpy arrays of R2 scores.
    '''
    # Clip each set of MSE scores between 0 and 50
    for test_set in scores["mse"]:
        scores["mse"][test_set] = np.clip(scores["mse"][test_set], 0.0, 50.0)
    # Clip each set of R-squared scores between 0 and 1
    for test_set in scores["r2"]:
        scores["r2"][test_set] = np.clip(scores["r2"][test_set], 0.0, 50.0)


def set_up_subplot(ax, scores, score_type):
    '''
    Helper function that plots the given array of training, validation, and testing scores on the given subplot. This function takes care of setting limits on and labeling the axes.

    Args:
        ax: matplotlib.axes.SubplotBase object
            the matplotlib subplot on which we will plot our points
        scores: dict
            a dictionary containing two dictionaries, one for the numpy arrays of MSE scores 
            and another for the numpy arrays of R2 scores.
        score_type: string
            either 'mse' or 'r2'; this determines how we limit the y-axis and how we 
            label the graph 
    '''
    n_experiments = len(scores[score_type]["train"])
    x_values = [range(1, n_experiments + 1)] * n_experiments
    y_values = scores[score_type].values()
    labels = ['Training', 'Validation', 'Testing']
    colors = ['blue', 'red', 'green']
    # Plot MSE or R2 scores for training, validation, testing sets
    for x, y, label, color in zip(x_values, y_values, labels, colors):
        ax.plot(x, y, marker='o', color=color, label=label)
        ax.legend(loc='best')
    # Set x limits to 0 to number experiments + 1    
    ax.set_xlim([0.0, n_experiments + 1])
    # Set y limits between 0 and 50 for MSE, or between 0 and 1 for R2
    if score_type == "mse":
        ax.set_ylim([0.0, 50.0])
    elif score_type == "r2":
        ax.set_ylim([0.0, 1.0])
    else: 
        raise ValueError("Unexpected score_type")
    # Set x label to 'p-degree' for both types of graphs
    ax.set_xlabel('p-degree')
    # Set y label to 'MSE' or 'R-squared' depending on score_type
    y_label = 'MSE' if score_type == 'mse' else 'R-squared'
    ax.set_ylabel(y_label)


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

        # Compute the polynomial terms needed for the data
        # Generates x_1^2, x_1 x_2, x_1 x_3, ..., x_d^2
        poly_transform.fit(x_train)

        # Transform the data by nonlinear mapping
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

    # Convert each scores to NumPy arrays
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

    # Create figure for training, validation and testing scores for different features
    fig = plt.figure()
    # Create subplot for MSE for training, validation, testing
    ax = fig.add_subplot(1, 2, 1) # 1 row, 2 columns; get 1st subplot in the figure
    set_up_subplot(ax, scores, 'mse')
    # Create subplot for R-square for training, validation, testing
    ax = fig.add_subplot(1, 2, 2)
    set_up_subplot(ax, scores, 'r2')
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

        # Transform x_train to x_poly_train with p-degree expansion
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

    # Convert each scores to NumPy arrays
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

    # Create figure for training, validation and testing scores for different features
    fig = plt.figure()
    # Create subplot for MSE for training, validation, testing
    ax = fig.add_subplot(1, 2, 1) # 1 row, 2 columns; get 1st subplot in the figure
    set_up_subplot(ax, scores, 'mse')
    # Create subplot for R-square for training, validation, testing
    ax = fig.add_subplot(1, 2, 2)
    set_up_subplot(ax, scores, 'r2')
    # Create super title 'Our Polynomial Expansion on Training, Validation and Testing Sets'
    plt.suptitle('Our Polynomial Expansion on Training, Validation and Testing Sets')

    plt.show()
