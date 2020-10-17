import numpy as np
import sklearn.datasets as skdata
from sklearn.linear_model import Perceptron

print("""
------------------------------------------------------------
BINARY CLASSIFICATION WITH PERCEPTRONS
Breast Cancer Dataset
------------------------------------------------------------
""")

"""
Load the breast cancer dataset
"""
breast_cancer_data = skdata.load_breast_cancer()
x = breast_cancer_data.data
y = breast_cancer_data.target

"""
Split the breast cancer dataset into 80% train, 10% validation, 10% test
"""
# Create permutation of all indices
number_of_samples = x.shape[0]
idx = np.random.permutation(number_of_samples)

train_split_idx = int(0.80 * number_of_samples)
val_split_idx = int(0.90 * number_of_samples)

train_idx = idx[:train_split_idx]
val_idx = idx[train_split_idx:val_split_idx]
test_idx = idx[val_split_idx:]

# Select samples from x and y to construct our training, validation, testing sets
x_train, y_train = x[train_idx, :], y[train_idx]
x_val, y_val = x[val_idx, :], y[val_idx]
x_test, y_test = x[test_idx, :], y[test_idx]

"""
Set up our Perceptron model:
"""
# tol is the stopping threshold if the training error at time t
#     is greater than the training error at time t-1 by tol
#     if error starts to increase, we've gone past the minimum of the loss function
# penalty and alpha relates to regularization (haven't covered yet)
model = Perceptron(penalty=None, alpha=0.0, tol=1e-1)

# Train our perceptron model
# returns the Perceptron object: Perceptron(alpha=0.0, tol=0.1)
model.fit(x_train, y_train)

"""
Evaluate our model on the validation set (whether tumor is benign or malignant)
"""
# Predict the class/labels
predictions_val = model.predict(x_val)
print("predictions_val: {}".format(predictions_val))
# output:
#   [0 0 1 0 1 1 1 1 0 0 0 1 1 1 0 1 0 0 0 1 1 1 1 1 0 0 0 1 0 1 1 0 1 0 0 1 1
#    1 0 1 1 1 1 0 1 1 1 1 1 1 1 0 1 0 1 0 0]
print("predictions_val shape: {}".format(predictions_val.shape))  # (57,)
print("predictions_val unique values: {}".format(
    np.unique(predictions_val)))  # [0 1]

# Check our accuracy: Add up the number of classifications we got wrong
# If classification was correct, then give score of 1; else 0
scores_val = np.where(predictions_val == y_val, 1, 0)
mean_accuracy_val = np.mean(scores_val)
# This will be different every time depending on how data is split in random permutation
print("mean accuracy on validation set: {}".format(mean_accuracy_val))

# We can also use scikit-learn's built-in function; it does the same thing!
mean_accuracy_val = model.score(x_val, y_val)

# Typically, we'd then use the results of validaton to tweak hyperparameters and repeat

"""
Evaluate on testing set
"""
# Predict the class/labels
predictions_test = model.predict(x_test)

# Check accuracy
scores_test = np.where(predictions_test == y_test, 1, 0)
mean_accuracy_test = np.mean(scores_test)
# or simply: mean_accuracy_test = model.score(x_test, y_test)

print("mean accuracy on test set: {}".format(mean_accuracy_test))


print("""
------------------------------------------------------------
MULTI-CLASS CLASSIFICATION WITH PERCEPTRONS
Wine Dataset
------------------------------------------------------------
""")


"""
Load the wine dataset
"""
wine_data = skdata.load_wine()
x = wine_data.data
y = wine_data.target

"""
Split the wine dataset into 80% train, 10% validation, 10% test
"""
# If we didn't want to use random permutation, we could just give the first 80%
# to training, the next 10% to validation, and the last 10% to testing
# But need to be careful -- this didn't work at all because the data actually came
# in ordered, so we have to randomize it!

# Create permutation of all indices
number_of_samples = x.shape[0]
idx = np.random.permutation(number_of_samples)
train_split_idx = int(0.80 * number_of_samples)
val_split_idx = int(0.90 * number_of_samples)

train_idx = idx[:train_split_idx]
val_idx = idx[train_split_idx:val_split_idx]
test_idx = idx[val_split_idx:]

# Select samples from x and y to construct our training, validation, testing sets
x_train, y_train = x[train_idx, :], y[train_idx]
x_val, y_val = x[val_idx, :], y[val_idx]
x_test, y_test = x[test_idx, :], y[test_idx]

# Train-Validate Loop
models = []
scores = []
for tol in [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]:

    # Create the perceptron model
    model = Perceptron(penalty=None, alpha=0.0, tol=tol)

    # Train the perceptron model
    model.fit(x_train, y_train)

    # Predict the class/labels
    predictions_val = model.predict(x_val)

    # Check accuracy
    score = model.score(x_val, y_val)
    print("tol: {}; score on validation set: {}".format(tol, score))

    # Save our model and score so we can use the best one later
    models.append(model)
    scores.append(score)

# tol: 0.001; score on validation set: 0.6666666666666666
# tol: 0.005; score on validation set: 0.6666666666666666
# tol: 0.01; score on validation set: 0.6666666666666666
# tol: 0.05; score on validation set: 0.6666666666666666
# tol: 0.1; score on validation set: 0.6666666666666666
# tol: 0.5; score on validation set: 0.6666666666666666
# tol: 1.0; score on validation set: 0.6666666666666666
# We end up converging to the same minimum for all the different tolerances
# So maybe perceptrons aren't the best way to do this

"""
Evaluate on testing set
"""
# Pick the best model
max_score = max(scores)
max_index = scores.index(max_score)
model = models[max_index]

# Predict the class/labels using the best model
predictions_test = model.predict(x_test)

# Check accuracy
mean_accuracy_test = model.score(x_test, y_test)

print("mean accuracy on test set: {}".format(mean_accuracy_test))

# Try on Iris, Wine, and Digits datasets

print("""
------------------------------------------------------------
MULTI-CLASS CLASSIFICATION WITH PERCEPTRONS
Iris Dataset
------------------------------------------------------------
""")

"""
Load the iris dataset
"""
iris_data = skdata.load_iris()
x = iris_data.data
y = iris_data.target

"""
Split the iris dataset into 80% train, 10% validation, 10% test
"""
# Create permutation of all indices
number_of_samples = x.shape[0]
idx = np.random.permutation(number_of_samples)
train_split_idx = int(0.80 * number_of_samples)
val_split_idx = int(0.90 * number_of_samples)

train_idx = idx[:train_split_idx]
val_idx = idx[train_split_idx:val_split_idx]
test_idx = idx[val_split_idx:]

# Select samples from x and y to construct our training, validation, testing sets
x_train, y_train = x[train_idx, :], y[train_idx]
x_val, y_val = x[val_idx, :], y[val_idx]
x_test, y_test = x[test_idx, :], y[test_idx]

# Train-Validate Loop
models = []
scores = []
for tol in [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]:

    # Create the perceptron model
    model = Perceptron(penalty=None, alpha=0.0, tol=tol)

    # Train the perceptron model
    model.fit(x_train, y_train)

    # Predict the class/labels
    predictions_val = model.predict(x_val)

    # Check accuracy
    score = model.score(x_val, y_val)
    print("tol: {}; score on validation set: {}".format(tol, score))

    # Save our model and score so we can use the best one later
    models.append(model)
    scores.append(score)

# tol: 0.001; score on validation set: 1.0
# tol: 0.005; score on validation set: 0.6666666666666666
# tol: 0.01; score on validation set: 0.6666666666666666
# tol: 0.05; score on validation set: 1.0
# tol: 0.1; score on validation set: 1.0
# tol: 0.5; score on validation set: 0.9333333333333333
# tol: 1.0; score on validation set: 0.9333333333333333

"""
Evaluate on testing set
"""
# Pick the best model
max_score = max(scores)
max_index = scores.index(max_score)
model = models[max_index]

# Predict the class/labels using the best model
predictions_test = model.predict(x_test)

# Check accuracy
mean_accuracy_test = model.score(x_test, y_test)

print("mean accuracy on test set: {}".format(mean_accuracy_test))
# outputs: mean accuracy on test set: 0.9333333333333333
