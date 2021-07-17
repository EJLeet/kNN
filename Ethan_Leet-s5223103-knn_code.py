""" 
Ethan Leet
s5223103
2802 ICT
Assignment 2
Task 1 - kNN
"""
"""
There are two distance measures in this program, Euclidean and Manhattan.
To change the distance measure comment and uncomment lines 72 and 73 depending on what
measure you want to use. Also comment and uncomment lines 145 and 146 so the plot drawn
has the right description.

To change the size of the testing and training data set change the percentage on line 50.

To change the initial k value that prints the flower class, predicted class, boolean value
of if prediction was correct and the testing/training set accuracy change line 101.

To change the how many times the k value loops change line 126.
"""

from math import sqrt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None

# read in data
data = pd.read_csv("iris.csv")
# create lsit of integer/float data
all_data = data.iloc[:]
# shuffle the data based on row index
shuffle_index = np.random.permutation(all_data.shape[0])
all_data = all_data.iloc[shuffle_index]


# split data function takes a percentage and sets up dataframes, values and true class values
def split_data(percentage):
    training_size = int(all_data.shape[0] * percentage)
    training_df = all_data.iloc[:training_size, :]
    testing_df = all_data.iloc[training_size:, :]
    training = training_df.values
    testing = testing_df.values
    x_true = training[:, -1]
    y_true = testing[:, -1]
    return training_df, testing_df, training, testing, x_true, y_true


# initialise dataframes, value and true class values based on 70% training set size
training_df, testing_df, training, testing, x_true, y_true = split_data(0.7)

# euclidean distance function to calculate kNN
def euclidean_distance(x_testing, x_training):
    distance = 0
    for i in range(len(x_testing)-1):
        distance += (x_testing[i]-x_training[i])**2
    return sqrt(distance)

# manhattan distance function to calculate kNN
def manhattan_distance(x_testing, x_training):
    distance = 0
    for i in range(len(x_testing)-1):
        distance += (abs(x_testing[i] - x_training[i]))
    return distance

# calculate kNN
def kNN(x_testing, x_training, num_neighbors):
    distances = []
    data = []
    for i in x_training:
        # comment/uncomment one of these depending on what measure you want to use
        distances.append(euclidean_distance(x_testing, i))
        #distances.append(manhattan_distance(x_testing, i))
        data.append(i)
    distances = np.array(distances)
    data = np.array(data)
    sort_indexes = distances.argsort()
    data = data[sort_indexes]
    return data[:num_neighbors]

# classify new data
def classification(x_testing, x_training, num_neighbors):
    classes = []
    neighbors = kNN(x_testing, x_training, num_neighbors)
    for i in neighbors:
        classes.append(i[-1])
    predicted = max(classes, key=classes.count)
    return predicted

# work out accuracy based on true vs predicted
def accuracy(true, pred):
    num_correct = 0
    for i in range(len(true)):
        if true[i] == pred[i]:
            num_correct += 1
    accuracy = num_correct/len(true)
    return accuracy


# printing test/train data and accuracy for a k value for each piece of data
k = 100
# for test set
y_pred = []
for i in testing:
    y_pred.append(classification(i, training, k))
testing_accuracy = accuracy(y_true, y_pred)
testing_df.insert(5, 'predicted species', y_pred)
# boolean value to see if class = predicted class
testing_df['correct prediction'] = testing_df['species'] == testing_df['predicted species']
print(testing_df[['species', 'predicted species',
      'correct prediction']].to_string(index=False))
print("Testing set accuracy: ", testing_accuracy * 100, "%", '\n')
# for train set
x_pred = []
for i in training:
    x_pred.append(classification(i, training, k))
training_accuracy = accuracy(x_true, x_pred)
training_df.insert(5, 'predicted species', x_pred)
# boolean value to see if class = predicted class
training_df['correct prediction'] = training_df['species'] == training_df['predicted species']
print(training_df[['species', 'predicted species',
      'correct prediction']].to_string(index=False))
print("Training set accuracy: ", training_accuracy * 100, "%", '\n')

# testing different k values
changing_k = 100
testing_accuracy = np.empty(changing_k)
training_accuracy = np.empty(changing_k)
# loop from k = 1 to 15 inclusive
for k_vals in range(1, changing_k):
    y_pred = []
    for j in testing:
        y_pred.append(classification(j, training, k_vals))
    x_pred = []
    for i in training:
        x_pred.append(classification(i, training, k_vals))
    testing_accuracy[k_vals] = accuracy(y_true, y_pred)
    training_accuracy[k_vals] = accuracy(x_true, x_pred)
    print("Accuracy of testing set k-value", k_vals,
          "  ", testing_accuracy[k_vals] * 100, '%')
    print("Accuracy of training set k-value", k_vals,
          "  ", training_accuracy[k_vals] * 100, '%', '\n')

# plotting different k values
plt.title('k-NN: Varying Number of Neighbors with Euclidean Distance')
#plt.title('k-NN: Varying Number of Neighbors with Manhattan Distance')
plt.plot(testing_accuracy[1:], label='Testing Accuracy')
plt.plot(training_accuracy[1:], label='Training Accuracy')
plt.legend()
plt.xlabel('K-Value')
plt.ylabel('Accuracy')
plt.show()