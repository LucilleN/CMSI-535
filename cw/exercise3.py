import numpy as np
import sklearn.datasets as skdata
from sklearn.linear_model import Perceptron


'''
Name: Doe, John (Please write names in <Last Name, First Name> format)

Collaborators: Doe, Jane (Please write names in <Last Name, First Name> format)

Collaboration details: Discussed <function name> implementation details with Jane Doe.

Summary:
Report your scores here. For example,

Results using scikit-learn Perceptron model
Training set mean accuracy: 0.8289
Validation set mean accuracy: 0.7778
Testing set mean accuracy: 0.8200
Results using our Perceptron model trained with 10 steps
Training set mean accuracy: 0.0000
Validation set mean accuracy: 0.0000
Results using our Perceptron model trained with 20 steps
Training set mean accuracy: 0.0000
Validation set mean accuracy: 0.0000
Results using our Perceptron model trained with 60 steps
Training set mean accuracy: 0.0000
Validation set mean accuracy: 0.0000
Using best model trained with 10 steps
Testing set mean accuracy: 0.0000
'''

'''
Implementation of Perceptron for binary classification
'''
class PerceptronBinary(object):

    def __init__(self):
        # Define private variables
        self.__weights = None

    def __update(self, x, y):
        '''
        Update the weight vector during each training iteration

        Args:
            x : numpy
                d x N feature vector
            y : numpy
                1 x N ground-truth label
        '''
        # TODO: Implement the member update function

    def fit(self, x, y, T=100, tol=1e-3):
        '''
        Fits the model to x and y by updating the weight vector
        based on mis-classified examples for t iterations until convergence

        Args:
            x : numpy
                d x N feature vector
            y : numpy
                1 x N ground-truth label
            t : int
                number of iterations to optimize perceptron
            tol : float
                change of loss tolerance, if greater than loss + tolerance, then stop
        '''
        # TODO: Implement the fit function

    def predict(self, x):
        '''
        Predicts the label for each feature vector x

        Args:
            x : numpy
                N x d feature vector

        Returns:
            numpy : 1 x N label vector
        '''
        # TODO: Implement the predict function

    def score(self, x, y):
        '''
        Predicts labels based on feature vector x and computes the mean accuracy
        of the predictions

        Args:
            x : numpy
                d x N feature vector
            y : numpy
                1 x N ground-truth label

        Returns:
            float : mean accuracy
        '''
        # TODO: Implement the score function


if __name__ == '__main__':

    breast_cancer_data = skdata.load_breast_cancer()
    x = breast_cancer_data.data
    y = breast_cancer_data.target

    # 80 percent train, 10 percent validation, 10 percent test split
    train_idx = []
    val_idx = []
    test_idx = []
    for idx in range(x.shape[0]):
        if idx and idx % 9 == 0:
            val_idx.append(idx)
        elif idx and idx % 10 == 0:
            test_idx.append(idx)
        else:
            train_idx.append(idx)

    x_train, x_val, x_test = x[train_idx, :], x[val_idx, :], x[test_idx, :]
    y_train, y_val, y_test = y[train_idx], y[val_idx], y[test_idx]

    '''
    Trains and tests Perceptron model from scikit-learn
    '''
    model = Perceptron(penalty=None, alpha=0.0, tol=1e-3)
    # Trains scikit-learn Perceptron model
    model.fit(x_train, y_train)

    print('Results using scikit-learn Perceptron model')

    # Test model on training set
    scores_train = model.score(x_train, y_train)
    print('Training set mean accuracy: {:.4f}'.format(scores_train))

    # Test model on validation set
    scores_val = model.score(x_val, y_val)
    print('Validation set mean accuracy: {:.4f}'.format(scores_val))

    # Test model on testing set
    scores_test = model.score(x_test, y_test)
    print('Testing set mean accuracy: {:.4f}'.format(scores_test))

    '''
    Trains and tests our Perceptron model for binary classification
    '''
    # TODO: obtain dataset in correct shape (d x N)

    # TODO: obtain labels in {+1, -1} format

    # TODO: Initialize model, train model, score model on train, val and test sets

    # Train 3 PerceptronBinary models using 10, 50, and 60 steps with tolerance of 1
    models = []
    scores = []
    steps = [10, 20, 60]
    for T in steps:
        # Initialize PerceptronBinary model

        print('Results using our Perceptron model trained with {} steps'.format(T))
        # Train model on training set

        # Test model on training set
        scores_train = 0.0
        print('Training set mean accuracy: {:.4f}'.format(scores_train))

        # Test model on validation set
        scores_val = 0.0
        print('Validation set mean accuracy: {:.4f}'.format(scores_val))

        # Save the model and its score

    # Select the best performing model on the validation set
    best_idx = 0

    print('Using best model trained with {} steps'.format(steps[best_idx]))

    # Test model on testing set
    scores_test = 0.0
    print('Testing set mean accuracy: {:.4f}'.format(scores_test))
