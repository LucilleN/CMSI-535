import numpy as np
import sklearn.datasets as skdata
from sklearn.linear_model import Perceptron


'''
Name: Njoo, Lucille

Collaborators: N/A

Collaboration details: N/A

Summary:
Program output: 

    Results using scikit-learn Perceptron model
    Training set mean accuracy: 0.7943
    Validation set mean accuracy: 0.7321
    Testing set mean accuracy: 0.7857
    Results using our Perceptron model trained with 10 steps
    Training set mean accuracy: 0.7527
    Validation set mean accuracy: 0.6786
    Results using our Perceptron model trained with 20 steps
    Training set mean accuracy: 0.8206
    Validation set mean accuracy: 0.7679
    Results using our Perceptron model trained with 60 steps
    Training set mean accuracy: 0.7352
    Validation set mean accuracy: 0.6786
    Using best model trained with 20 steps
    Testing set mean accuracy: 0.8036

Final result:
With the way we separated data in class, using idx % 10 == 9 to get
the validation indexes, the best model is the one trained with
T=20 steps, which got us an accuracy of about 80% on the testing set.
'''

'''
Implementation of Perceptron for binary classification. 

This program implements a PerceptronBinary class that uses the
Perceptron Learning Algorithm to fit to training data, make predictions,
and score its own accuracy. The main script then loads sklearn data and 
tests the accuracy of the sklearn Perceptron on train, val, and test sets. 
Then, for comparison, it trains three of our own PerceptronBinary models, 
each with different numbers of training iteration steps, and selects the 
model that did the best on the validation set to score with the testing set. 
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
        # Get x in the right shape; x is currently d x N; we want it to be d+1 x N
        # We want x0 to be threshold
        threshold = 0.5 * np.ones([1, x.shape[1]])  # (1 x N)
        x = np.concatenate([threshold, x], axis=0)  # (d+1 x N)

        # loop from 0 to N-1: iterate through every data point and checks if incorrect
        for n in range(x.shape[1]):
            # x is in shape (d+1 x N), so shape of one column is (d+1), a 1-dimensional array
            x_n = np.expand_dims(x[:, n], axis=-1)  # has shape (d+1 x 1)
            # expand_dims restores the extra dimension
            # expand_dims(..., axis=-1) will add to end (a, b, c, d, e) --> (a, b, c, d, e, 1)
            # if axis=0, would add to beginning: (a, b, c, d, e) --> (1, a, b, c, d, e)

            # Note: this does the same thing:
            # x_n = np.reshape(x[:, n], (-1, 1)) # reshape -1 means all elements

            # Predict the label for x_n (will be a scalar, + or - 1)
            prediction = np.sign(np.matmul(self.__weights.T, x_n))

            # Check if prediction is equal to ground truth y; if not, update weights
            if prediction != y[n]:
                # Update weights: w^(t+1) = w^(t) + (y^n * x^n)
                # shape: (d+1 x 1) = (d+1 x 1)  +  [ 1 * (d+1 x 1) ]
                self.__weights = self.__weights + (y[n] * x_n)

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
        # Initialize the weights; should have shape (d+1 x 1)
        # [w0, w1, w2, w3, ... , wd] = [0, 0, 0, 0, ... , 0]
        self.__weights = np.zeros([x.shape[0]+1, 1])  # (d+1 x 1)
        # set w0 = -1 so that the w0x0 term can be -threshold
        self.__weights[0][0] = -1

        # Initialize previous loss and previous weights to keep track of them through loop
        prev_loss = 2.0  # because the loss we compute can be at most 1.0, since it's normalized
        prev_weights = np.copy(self.__weights)

        # Train our model with T iterations/timesteps
        for t in range(T):
            # Compute our loss
            predictions = self.predict(x)
            # l = 1/N \sum_n^N I(h(x^n) != y^n)
            # mean is same thing as summing them all and dividing by N, the length
            loss = np.mean(np.where(predictions != y, 1, 0))

            # print("t={} loss={}".format(t+1, loss))

            # Stopping conditions
            if loss == 0.0:
                break
            elif loss > prev_loss + tol and t > 2:
                # If loss from time t = 0.1, then time t+1 = 0.5, this means we did worse,
                # so we should take the previous timestep's weights.
                # Also, need to train at least 3 steps before we decide we can't fit to the data
                self.__weights = prev_weights
                break

            # Update previous loss and previous weights
            prev_loss = loss
            prev_weights = self.__weights

            self.__update(x, y)

    def predict(self, x):
        '''
        Predicts the label for each feature vector x

        Args:
            x : numpy
                d x N feature vector

        Returns:
            numpy : 1 x N label vector
        '''
        # Get our vectors in the right dimensions
        # [w0, w1, w2, w3, w4, ... , wd] is in the shape (d+1 x N)
        # [__, x1, x2, x3, x4, ... , xd] is in the shape (d x N) and we want the first term to be the threshold
        # shape of threshold should be (1 x N)
        threshold = 0.5 * np.ones([1, x.shape[1]])  # (1 x N) vector
        # Augment the features matrix x with the threshold
        # now the shape is (d+1 x N)
        x = np.concatenate([threshold, x], axis=0)  # (d+1 x N)

        # Predict using w^Tx
        # (d+1 x 1)^T times (d+1 x N) --> (1 x N) vector
        predictions = np.matmul(self.__weights.T, x)

        # What we care about is the sign (+ or -) of our prediction
        # Thus, we have h(x) = sign(w^Tx)
        return np.sign(predictions)

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
        # Predict labels based on our hypothesis
        predictions = self.predict(x)  # (1 x N) vector of values {-1, +1}

        # Comparing if our predictions and true y are the same
        scores = np.where(predictions == y, 1.0, 0.0)

        # Return the mean accuracy of the scores
        return np.mean(scores)


if __name__ == '__main__':

    breast_cancer_data = skdata.load_breast_cancer()
    x = breast_cancer_data.data
    y = breast_cancer_data.target

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
    # Obtain dataset in correct shape (d x N)
    x_train = np.transpose(x_train, axes=(1, 0))
    x_val = np.transpose(x_val, axes=(1, 0))
    x_test = np.transpose(x_test, axes=(1, 0))

    # Obtain labels in {+1, -1} format
    y_train = np.where(y_train == 0, -1, 1)
    y_val = np.where(y_val == 0, -1, 1)
    y_test = np.where(y_test == 0, -1, 1)

    # Train 3 PerceptronBinary models using 10, 50, and 60 steps with tolerance of 1
    models = []
    scores = []
    steps = [10, 20, 60]
    for T in steps:
        # Initialize PerceptronBinary model
        model = PerceptronBinary()

        print('Results using our Perceptron model trained with {} steps'.format(T))
        # Train model on training set
        model.fit(x_train, y_train, T=T, tol=1)

        # Test model on training set
        scores_train = model.score(x_train, y_train)
        print('Training set mean accuracy: {:.4f}'.format(scores_train))

        # Test model on validation set
        scores_val = model.score(x_val, y_val)
        print('Validation set mean accuracy: {:.4f}'.format(scores_val))

        # Save the model and its score
        models.append(model)
        scores.append(scores_val)

    # Select the best performing model on the validation set
    max_score = max(scores)
    best_idx = scores.index(max_score)
    best_model = models[best_idx]

    print('Using best model trained with {} steps'.format(steps[best_idx]))

    # Test model on testing set
    scores_test = best_model.score(x_test, y_test)
    print('Testing set mean accuracy: {:.4f}'.format(scores_test))

    # Output:
    #   Results using scikit-learn Perceptron model
    #   Training set mean accuracy: 0.7943
    #   Validation set mean accuracy: 0.7321
    #   Testing set mean accuracy: 0.7857
    #   Results using our Perceptron model trained with 10 steps
    #   Training set mean accuracy: 0.7527
    #   Validation set mean accuracy: 0.6786
    #   Results using our Perceptron model trained with 20 steps
    #   Training set mean accuracy: 0.8206
    #   Validation set mean accuracy: 0.7679
    #   Results using our Perceptron model trained with 60 steps
    #   Training set mean accuracy: 0.7352
    #   Validation set mean accuracy: 0.6786
    #   Using best model trained with 20 steps
    #   Testing set mean accuracy: 0.8036

    # Final result:
    # With the way we separated data in class, using idx % 10 == 9 to get
    # the validation indexes, the best model is the one trained with
    # T=20 steps, which got us an accuracy of about 80% on the testing set.
