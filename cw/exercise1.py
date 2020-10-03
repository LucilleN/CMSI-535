# Lucille Njoo
# Classwork 1, September 1, 2020

import numpy as np
import sklearn.datasets as skdata

"""
Boston Housing Dataset
"""
print("BOSTON HOUSNG DATASET")

"""
Loading data
"""
boston_housing_data = skdata.load_boston()

# Load data as NumPy arrays
feature_names = boston_housing_data.feature_names
x = boston_housing_data.data
y = boston_housing_data.target

# Look at the shape of the data
print("shape of Boston Housing Data")
print(x.shape) # outputs (506, 13): 506 samples in the dataset, each with 13 dimensions (features)
print(y.shape) # outputs (506,): each sample has a corresponding groundtruth label

# Select the 1st sample and print its name and value in each line as "<name>: <value>"
print("feature names and values for first sample")
for name, value in zip(feature_names, x[0, ...]): # makes a new array with all 13 elements found in x[0]
    print("{}: {}".format(name, value))

# This outputs:
#   CRIM: 0.00632
#   ZN: 18.0
#   INDUS: 2.31
#   CHAS: 0.0
#   NOX: 0.538
#   RM: 6.575
#   AGE: 65.2
#   DIS: 4.09
#   RAD: 1.0
#   TAX: 296.0
#   PTRATIO: 15.3
#   B: 396.9
#   LSTAT: 4.98

# Print its groundtruth label
print("label: {0}".format(y[0]))
# outputs:
#   label: 24.0

"""
Split the data into train (80%), validation (10%), and testing (10%) sets
"""
number_of_samples = x.shape[0] # 516 samples, each with 13 values
idx = np.random.permutation(number_of_samples) # creates permutation of all indices
# this essentially randomly shuffles the indices from 0 to 506
# print(idx) # this shows an array of all the numbers from 0 to 506 in random order

train_split_idx = int(0.80 * number_of_samples)
val_split_indx = int(0.90 * number_of_samples)

train_idx = idx[:train_split_idx] # 0th element to the 80% element
val_idx = idx[train_split_idx:val_split_indx] # 80% element to 90% element
test_idx = idx[val_split_indx:] # 90% element to last element

# Select the examples from x and y to construct our training, validation, and testing sets
x_train, y_train = x[train_idx, :], y[train_idx]
x_val, y_val = x[val_idx, :], y[val_idx]
x_test, y_test = x[test_idx, :], y[test_idx]

print("########################")

print("Boston Housing Dataset")
print("Training Data:") # there should be 404 entries (80% of 516)
print(x_train.shape) # (404, 13)
print(y_train.shape) # (404,)
print("Validation Data:") # there should be 51 entires (10% of 506)
print(x_val.shape) # (51, 13)
print(y_val.shape) # (51,)
print("Testing Data:") # there should be 51 entires (10% of 506)
print(x_test.shape) # (51, 13)
print(y_test.shape) # (51,)

print("########################")

"""
Repeat with Breast Cancer Dataset
"""
print("BREAST CANCER DATASET")

"""
Loading Data
"""
breast_cancer_data = skdata.load_breast_cancer()

# Load data as NumPy arrays
breast_cancer_feature_names = breast_cancer_data.feature_names
bc_x = breast_cancer_data.data
bc_y = breast_cancer_data.target

# Look at the shape of the data
print("shape of Breast Cancer Data")
print(bc_x.shape) # (569, 30)
print(bc_y.shape) # (569,)

# Select the first sample and print the feature name and value for every feature in it
print("feature names and values for first sample")
for name, value in zip(breast_cancer_feature_names, bc_x[0, ...]):
    print("{}: {}".format(name, value))
# outputs:
#   mean radius: 17.99
#   mean texture: 10.38
#   mean perimeter: 122.8
#   mean area: 1001.0
#   mean smoothness: 0.1184
#   mean compactness: 0.2776
#   mean concavity: 0.3001
#   mean concave points: 0.1471
#   mean symmetry: 0.2419
#   mean fractal dimension: 0.07871
#   radius error: 1.095
#   texture error: 0.9053
#   perimeter error: 8.589
#   area error: 153.4
#   smoothness error: 0.006399
#   compactness error: 0.04904
#   concavity error: 0.05373
#   concave points error: 0.01587
#   symmetry error: 0.03003
#   fractal dimension error: 0.006193
#   worst radius: 25.38
#   worst texture: 17.33
#   worst perimeter: 184.6
#   worst area: 2019.0
#   worst smoothness: 0.1622
#   worst compactness: 0.6656
#   worst concavity: 0.7119
#   worst concave points: 0.2654
#   worst symmetry: 0.4601
#   worst fractal dimension: 0.1189

# Print the groundtruth label
print("label: {}".format(bc_y[0])) 
# outputs:
#   label: 0

"""
Split the data into train (80%), validation (10%), and testing (10%) sets
"""

breast_cancer_indexes = np.random.permutation(bc_x.shape[0])

bc_train_split_index = int(0.8 * bc_x.shape[0])
bc_val_split_index = int(0.9 * bc_x.shape[0])

bc_train_indexes = breast_cancer_indexes[:bc_train_split_index]
bc_val_indexes = breast_cancer_indexes[bc_train_split_index:bc_val_split_index]
bc_test_indexes = breast_cancer_indexes[bc_val_split_index:]

bc_x_train, bc_y_train = bc_x[bc_train_indexes, :], bc_y[bc_train_indexes]
bc_x_val, bc_y_val = bc_x[bc_val_indexes, :], bc_y[bc_val_indexes]
bc_x_test, bc_y_test = bc_x[bc_test_indexes, :], bc_y[bc_test_indexes]

print("########################")

print("Breast Cancer Dataset")
print("Training Data:") # there should be 455 entries (80% of 516)
print(bc_x_train.shape) # (455, 30)
print(bc_y_train.shape) # (455,)
print("Validation Data:") # there should be 57 entires (10% of 506)
print(bc_x_val.shape) # (57, 30)
print(bc_y_val.shape) # (57,)
print("Testing Data:") # there should be 57 entires (10% of 506)
print(bc_x_test.shape) # (57, 30)
print(bc_y_test.shape) # (57,)

print("########################")