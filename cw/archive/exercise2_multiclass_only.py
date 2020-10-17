import numpy as np
import sklearn.datasets as skdata
from sklearn.linear_model import Perceptron


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

# Output (varies every time):
#   tol: 0.001; score on validation set: 0.6666666666666666
#   tol: 0.005; score on validation set: 0.6666666666666666
#   tol: 0.01; score on validation set: 0.6666666666666666
#   tol: 0.05; score on validation set: 0.6666666666666666
#   tol: 0.1; score on validation set: 0.6666666666666666
#   tol: 0.5; score on validation set: 0.6666666666666666
#   tol: 1.0; score on validation set: 0.6666666666666666
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

print("""
Mean accuracy on test set: {}
""".format(mean_accuracy_test))

# Try on Iris, Wine, and Digits datasets
