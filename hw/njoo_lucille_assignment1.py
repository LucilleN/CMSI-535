import numpy as np
import sklearn.datasets as skdata
from sklearn.linear_model import Perceptron


'''
Name: Njoo, Lucille

Collaborators: Arteaga, Andrew

Collaboration details: Discussed high-level implementation details for `fit`, `__update`, and 
`predict` with Andrew Arteaga. I thought my code was broken, so we talked over what each 
method was doing, and after realizing that our code was doing the same thing in different ways, 
I did a lot more tuning with the hyperparameters and was able to get good results.

Summary:

1) What did you do in this assignment?

For this assignment, I implemented a multi-class perceptron class by writing its `fit`, `__update`, 
`predict`, and `score` methods. In the main function, I created three PerceptronMultiClass models 
with different numbers of timesteps for both the iris dataset and the wine dataset, then trained 
and validated them to find the best model for each. I then scored the best models for the iris and 
wine datasets on the testing data, and the end result is two models that score as well as or better
than the sklearn Perceptron, with testing set mean accuracies of 0.9286 and 0.5882 respectively. 

2) How did you do it?

To implement the PerceptronMultiClass methods, I used the multi-class perceptron learning algorithm:
for each of our N data samples, we use the current weights to predict the class of x_n, and if our
prediction is incorrect, we update the weights of both the classes c_hat and c_star such that our 
weights for c_hat = weights for c_hat - x_n, and our weights for c_star = weights for c_star + x_n.

Much of this was similar to the PerceptronBinary class we implemented as an in-class exercise, but 
this time with these main differences to allow for an arbitrary number of classes:
    - I changed `self.__weights` so that it contained the weight vectors for all of c classes. 
    - When updating the weights, whereas before we had only had to do a single adjustment, in 
      PerceptronMultiClass, I updated the weights for both the incorrectly predicted class and the 
      groundtruth class. 
    - Rather than the prediction simply being sign(w \dot x), this time, each w \dot x was treated 
      as a confidence score, and the prediction was the class that produced the highest confidence.

For the main method, I implemented the training loop by initializing the PerceptronMultiClass 
models, fitting them to the training data, then scoring their accuracies on training, validation, 
and testing data. In the process, I spent a lot of time tuning the hyperparameters: 
    - To decide on step and tolerance values to use, I tried tolerances ranging from 0.0001 to 100
      and steps ranging from 10 to 5000 for each one. 
    - I found that setting tolerances too low, such as 0.01 or 0.001, caused models to perform 
      poorly because their training iterations would be cut short, but setting tolerances too high, 
      such as 10, resulted in lower accuracy as well. 
    - Increasing the number of steps generally improved the models' scores. For the iris data, the 
      score seems to hit a maximum at 0.9286. For the wine data, I stopped at 5000 because of the
      increasing computation time. 

3) What are the constants and hyper-parameters you used?

    - I used tolerances of 1.0 for both the iris and the wine datasets. 
    - For the iris dataset, I used steps of 50, 500, and 1500 (best was 1500 steps).
    - For the wine dataset, I used steps of 500, 1000, and 5000 (best was 5000 steps).


Scores:

Results on the iris dataset using scikit-learn Perceptron model
Training set mean accuracy: 0.8512
Validation set mean accuracy: 0.7333
Testing set mean accuracy: 0.9286
Results on the iris dataset using our Perceptron model trained with 50 steps and tolerance of 1.0
Training set mean accuracy: 0.8843
Validation set mean accuracy: 0.8667
Results on the iris dataset using our Perceptron model trained with 500 steps and tolerance of 1.0
Training set mean accuracy: 0.8760
Validation set mean accuracy: 0.8667
Results on the iris dataset using our Perceptron model trained with 1500 steps and tolerance of 1.0
Training set mean accuracy: 0.9669
Validation set mean accuracy: 1.0000
Using best model trained with 1500 steps and tolerance of 1.0
Testing set mean accuracy: 0.9286
Results on the wine dataset using scikit-learn Perceptron model
Training set mean accuracy: 0.5625
Validation set mean accuracy: 0.4118
Testing set mean accuracy: 0.4706
Results on the wine dataset using our Perceptron model trained with 500 steps and tolerance of 1.0
Training set mean accuracy: 0.4375
Validation set mean accuracy: 0.3529
Results on the wine dataset using our Perceptron model trained with 1000 steps and tolerance of 1.0
Training set mean accuracy: 0.5278
Validation set mean accuracy: 0.4118
Results on the wine dataset using our Perceptron model trained with 5000 steps and tolerance of 1.0
Training set mean accuracy: 0.5833
Validation set mean accuracy: 0.5882
Using best model trained with 5000 steps and tolerance of 1.0
Testing set mean accuracy: 0.5882
'''

'''
Implementation of Perceptron for multi-class classification
'''
class PerceptronMultiClass(object):

    def __init__(self):
        # Define private variables, weights and number of classes
        self.__weights = None
        self.__n_class = 3

    def __concat_threshold_to_x(self, x):
        '''
        Concatenates the threshold to feature data x

        Args:
            x : numpy
                d x N feature vector
        Returns:
            numpy
                d+1 x N feature vector with threshold
        '''
        N = x.shape[1]
        # All the x_0's should be the threshold.
        thresholds = 1.0/self.__n_class * np.ones([1, N])  # (1 x N)
        return np.concatenate([thresholds, x], axis=0)  # (d+1 x N)

    def __update(self, x, y):
        '''
        Update the weight vector during each training iteration

        Args:
            x : numpy
                d x N feature vector
            y : numpy
                1 x N ground-truth label
        '''
        N = x.shape[1]

        x = self.__concat_threshold_to_x(x)
            
        # For each of N data samples, check if our prediction is correct
        for n in range(N):
            # Extract the input feature values for the current sample, in the shape (d+1 x 1)
            x_n = np.expand_dims(x[:, n], axis=-1)

            # The weights for class c's hyperplane is in column c of self.__weights
            # Expand dims so that each weights_c is a 2D array in the shape (d+1 x 1) instead of a 1D array in the shape (d+1)
            weights_for_each_class = [np.expand_dims(self.__weights[:, c], axis=-1) for c in range(self.__n_class)]
            prediction_confidence_scores = [np.matmul(weights_c.T, x_n) for weights_c in weights_for_each_class]

            # Our predicted class is the one that gives us the highest confidence
            highest_confidence = max(prediction_confidence_scores)
            predicted_label = prediction_confidence_scores.index(highest_confidence)

            # If our prediction does not match the ground truth label, update weights for c_hat and c_star
            groundtruth_label = y[n]
            if predicted_label != groundtruth_label: 
                self.__weights[:, predicted_label] = self.__weights[:, predicted_label] - np.squeeze(x_n, axis=-1) 
                self.__weights[:, groundtruth_label] = self.__weights[:, groundtruth_label] + np.squeeze(x_n, axis=-1) 


    def fit(self, x, y, T=100, tol=1e-3):
        '''
        Fits the model to x and y by updating the weight vector
        based on mis-classified examples for t iterations until convergence

        Args:
            x : numpy
                d x N feature vector
            y : numpy
                1 x N ground-truth label
            T : int
                number of iterations to optimize perceptron
            tol : float
                change of loss tolerance, if greater than loss + tolerance, then stop
        '''
        # The number of classes is the number of unique values in y
        self.__n_class = len(np.unique(y))

        # Initialize the weights to a (d+1 x c) matrix, where each column is the weights for class c 
        # Then set w_0 = -1 in all classes' columns to account for the threshold
        d = x.shape[0]
        self.__weights = np.zeros([d+1, self.__n_class])  # (d+1 x c)
        self.__weights[0, :] = -1.0
        
        # Keep track of loss and weights so that we know to stop when we have minimized loss
        # Initialize at 2.0 because the computed loss can be at most 1.0, since it's normalized
        prev_loss = 2.0  
        prev_weights = np.copy(self.__weights)

        for t in range(T):
            predictions = self.predict(x)
            loss = np.mean(np.where(predictions != y, 1.0, 0.0))

            # If we've minimized loss already, stop
            if loss == 0.0:
                break
            elif loss > prev_loss + tol and t > 2:
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
        N = x.shape[1]
        x = self.__concat_threshold_to_x(x)

        predictions = np.zeros([1, N])
        for n in range(N):
            # Extract the input feature values for the current sample in the shape (d+1 x 1)
            x_n = np.expand_dims(x[:, n], axis=-1)

            # The weights for class c's hyperplane is in column c of self.__weights
            # Expand dims so that each weights_c is a 2D array in the shape (d+1 x 1) instead of a 1D array in the shape (d+1)
            weights_for_each_class = [np.expand_dims(self.__weights[:, c], axis=-1) for c in range(self.__n_class)]
            label_confidence_scores = [np.matmul(weights_c.T, x_n) for weights_c in weights_for_each_class]

            # Prediction is the class that had the highest confidence
            highest_confidence = max(label_confidence_scores)
            predicted_label = label_confidence_scores.index(highest_confidence)

            predictions[0, n] = predicted_label
        
        return predictions

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
        # Predict labels for given feature data using our hypothesis weights
        predictions = self.predict(x) # (1 x N) 
        scores = np.where(predictions == y, 1.0, 0.0)
        return np.mean(scores)


def split_dataset(x, y, n_sample_train_to_val_test=8):
    '''
    Helper function to splits dataset into training, validation and testing sets

    Args:
        x : numpy
            d x N feature vector
        y : numpy
            1 x N ground-truth label
        n_sample_train_to_val_test : int
            number of training samples for every validation, testing sample

    Returns:
        x_train : numpy
            d x n feature vector
        y_train : numpy
            1 x n ground-truth label
        x_val : numpy
            d x m feature vector
        y_val : numpy
            1 x m ground-truth label
        x_test : numpy
            d x m feature vector
        y_test : numpy
            1 x m ground-truth label
    '''
    n_sample_interval = n_sample_train_to_val_test + 2

    train_idx = []
    val_idx = []
    test_idx = []
    for idx in range(x.shape[0]):
        if idx and idx % n_sample_interval == (n_sample_interval - 1):
            val_idx.append(idx)
        elif idx and idx % n_sample_interval == 0:
            test_idx.append(idx)
        else:
            train_idx.append(idx)

    x_train, x_val, x_test = x[train_idx, :], x[val_idx, :], x[test_idx, :]
    y_train, y_val, y_test = y[train_idx], y[val_idx], y[test_idx]

    return x_train, y_train, x_val, y_val, x_test, y_test


if __name__ == '__main__':

    iris_data = skdata.load_iris()
    wine_data = skdata.load_wine()

    datasets = [iris_data, wine_data]
    tags = ['iris', 'wine']

    # Experiment with 3 different max training steps (T) for each dataset
    train_steps_iris = [50, 500, 1500]
    train_steps_wine = [500, 1000, 5000]

    train_steps = [train_steps_iris, train_steps_wine]

    # Set a tolerance for each dataset
    tol_iris = 1.0 
    tol_wine = 1.0 

    tols = [tol_iris, tol_wine]

    for dataset, steps, tol, tag in zip(datasets, train_steps, tols, tags):
        # Split dataset into 80 training, 10 validation, 10 testing
        x = dataset.data
        y = dataset.target
        x_train, y_train, x_val, y_val, x_test, y_test = split_dataset(
            x=x,
            y=y,
            n_sample_train_to_val_test=8)

        '''
        Trains and tests Perceptron model from scikit-learn
        '''
        model = Perceptron(penalty=None, alpha=0.0, tol=1e-3)
        # Trains scikit-learn Perceptron model
        model.fit(x_train, y_train)

        print('Results on the {} dataset using scikit-learn Perceptron model'.format(tag))

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
        Trains, validates, and tests our Perceptron model for multi-class classification
        '''
        # Obtain dataset in correct shape (d x N)
        x_train = np.transpose(x_train, axes=(1, 0))
        x_val = np.transpose(x_val, axes=(1, 0))
        x_test = np.transpose(x_test, axes=(1, 0))

        # Initialize empty lists to hold models and scores
        models = []
        scores = []
        for T in steps:
            # Initialize PerceptronMultiClass model
            model = PerceptronMultiClass()

            print('Results on the {} dataset using our Perceptron model trained with {} steps and tolerance of {}'.format(tag, T, tol))
            # Train model on training set
            model.fit(x_train, y_train, T=T, tol=tol)

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

        print('Using best model trained with {} steps and tolerance of {}'.format(steps[best_idx], tol))

        # Test model on testing set
        scores_test = best_model.score(x_test, y_test)
        print('Testing set mean accuracy: {:.4f}'.format(scores_test))
