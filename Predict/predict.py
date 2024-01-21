import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

# Add the parent directory to sys.path to be able to import train
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from Train.train import *
import numpy as np

# Read the data from the csv file
data = pd.read_csv('data/train.csv')
data = np.array(data)

# Get the shape of the data for later computation
m, n = data.shape

# Shuffle the data to avoid any bias
np.random.shuffle(data)

#create training sets

# Dev set is the first 2000 examples
# This will serve as a validation set
dev_set = data[0:2000].T
Y_dev = dev_set[0]
X_dev = dev_set[1:n]
X_dev = X_dev / 255

# Training set is the rest of the examples
test_set = data[2000:m].T
Y_train_set = test_set[0]
X_train_set = test_set[1:n]
X_train_set = X_train_set / 255
n_train, m_train = X_train_set.shape

def get_predictions(A2):
    """
    Convert probabilities A2 to 0/1 predictions.
    
    Args:
    A2: numpy.ndarray
        The output of the second activation, of shape (1, number of examples).
    
    Returns:
    predictions: numpy.ndarray
        Predictions for the given dataset X.
    """
    return np.argmax(A2, 0)


def get_accuracy(predictions, Y):
    """
    Computes the accuracy of the predictions by comparing them with the true labels Y.
    
    Args:
    predictions: numpy.ndarray
        Predictions of the model, of shape (1, number of examples).
    Y: numpy.ndarray
        True labels vector, of shape (1, number of examples).
    
    Returns:
    accuracy: float
        Accuracy of the predictions (% of correctly predicted examples).
    """
    return np.sum(predictions == Y) / Y.size


def gradient_decent(X,Y,learning_rate,iterations):
    W1, b1, W2, b2 = initialize_parameters()
    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_propagation(X, W1, b1, W2, b2)
        dW1, db1, dW2, db2 = backward_propagation(X, Y, Z1, A1, Z2, A2, W1, W2)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, learning_rate)

        if i % 5 == 0:
            print("Iteration: ", i)
            predictions = get_predictions(A2)
            print(get_accuracy(predictions, Y))
    
    return W1, b1, W2, b2

def main():
    W1, b1, W2, b2 = gradient_decent(X_train_set, Y_train_set, 0.4, 1000)
    print(W1, b1, W2, b2)

    # Save the parameters
    np.save('parameters/W1.npy', W1)
    np.save('parameters/b1.npy', b1)
    np.save('parameters/W2.npy', W2)
    np.save('parameters/b2.npy', b2)
    


if __name__ == "__main__":
    main()