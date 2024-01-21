import numpy as np
import pandas as pd
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


def initialize_parameters():
    """
    Initialize the parameters (Weights and bias).

    Returns:
    W1: numpy.ndarray, shape (n_h, n_x)
        Weight matrix for the first layer.
    b1: numpy.ndarray, shape (n_h, 1)
        Bias vector for the first layer.
    W2: numpy.ndarray, shape (n_y, n_h)
        Weight matrix for the second layer.
    b2: numpy.ndarray, shape (n_h, 1)
        Bias vector for the second layer.
    """
    n_x = 784
    n_h = 10
    np.random.seed(1)
    W1 = np.random.randn(n_h, n_x) - 0.5
    b1 = np.zeros((n_h, 1)) - 0.5
    W2 = np.random.randn(n_y, n_h) - 0.5
    b2 = np.zeros((n_h, 1)) - 0.5

    return W1, b1, W2, b2

def ReLU(z):
    """
    Apply the Rectified Linear Unit (ReLU) activation function.

    Args:
    z: numpy.ndarray
        Input array.

    Returns:
    numpy.ndarray
        Output array after applying ReLU activation.
    """
    return np.maximum(0, z)

def ReLU_derivative(z):
    """
    Compute the derivative of the ReLU activation function.

    Args:
    z: numpy.ndarray
        Input array.

    Returns:
    numpy.ndarray
        Derivative of the ReLU activation function.
    """
    return z > 0

def softmax(z):
    """
    Apply the softmax activation function.

    Args:
    z: numpy.ndarray
        Input array.

    Returns:
    numpy.ndarray
        Output array after applying softmax activation.
    """
    Z = np.max(z, axis=0)
    A = np.exp(Z) / np.sum(np.exp(Z), axis=0)
    return A

def forward_propagation(X, W1, b1, W2, b2):
    """
    Perform forward propagation to compute the output of the network.

    Args:
    X: numpy.ndarray
        Input data.
    W1: numpy.ndarray, shape (n_h, n_x)
        Weight matrix for the first layer.
    b1: numpy.ndarray, shape (n_h, 1)
        Bias vector for the first layer.
    W2: numpy.ndarray, shape (n_y, n_h)
        Weight matrix for the second layer.
    b2: numpy.ndarray, shape (n_h, 1)
        Bias vector for the second layer.

    Returns:
    Z1: numpy.ndarray
        Output of the first linear unit.
    A1: numpy.ndarray
        Output of the first activation function.
    Z2: numpy.ndarray
        Output of the second linear unit.
    A2: numpy.ndarray
        Output of the second activation function.
    """
    Z1 = np.dot(W1, X) + b1
    A1 = ReLU(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = softmax(Z2)

    return Z1, A1, Z2, A2

def compute_cost(A2, Y):
    """
    Compute the cross-entropy cost.

    Args:
    A2: numpy.ndarray
        Output of the second activation function.
    Y: numpy.ndarray
        True "label" vector.

    Returns:
    float
        Cross-entropy cost.
    """
    m = Y.shape[0]
    cost = -1 / m * np.sum(Y * np.log(A2) + (1 - Y) * np.log(1 - A2))
    return cost

def one_hot(Y):
    """
    Encode labels into a one-hot representation.

    Args:
    Y: numpy.ndarray
        True "label" vector.


    Returns:
    numpy.ndarray
        One-hot encoded labels.
    """
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y

def backward_propagation(X, Y, Z1, A1, Z2, A2, W1, W2):
    """
    Perform backward propagation.

    Args:
    X: numpy.ndarray
        Input data.
    Y: numpy.ndarray
        True "label" vector.
    Z1: numpy.ndarray
        Output of the first linear unit.
    A1: numpy.ndarray
        Output of the first activation function.
    Z2: numpy.ndarray
        Output of the second linear unit.
    A2: numpy.ndarray
        Output of the second activation function.
    W1: numpy.ndarray, shape (n_h, n_x)
        Weight matrix for the first layer.
    W2: numpy.ndarray, shape (n_y, n_h)
        Weight matrix for the second layer.

    Returns:
    dW1: numpy.ndarray
        Gradient of the cost with respect to W1.
    db1: numpy.ndarray
        Gradient of the cost with respect to b1.
    dW2: numpy.ndarray
        Gradient of the cost with respect to W2.
    db2: numpy.ndarray
        Gradient of the cost with respect to b2.
    """
    one_hot_Y = one_hot(Y)
    m = Y.shape[0]
    dZ2 = A2 - one_hot_Y
    dW2 = 1 / m * np.dot(dZ2, A1.T)
    db2 = 1 / m * np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = np.multiply(np.dot(W2.T, dZ2), ReLU_derivative(Z1))
    dW1 = 1 / m * np.dot(dZ1, X.T)
    db1 = 1 / m * np.sum(dZ1, axis=1, keepdims=True)

    return dW1, db1, dW2, db2

def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, learning_rate):
    """
    Update the parameters using gradient descent.

    Args:
    W1: numpy.ndarray, shape (n_h, n_x)
        Weight matrix for the first layer.
    b1: numpy.ndarray, shape (n_h, 1)
        Bias vector for the first layer.
    W2: numpy.ndarray, shape (n_y, n_h)
        Weight matrix for the second layer.
    b2: numpy.ndarray, shape (n_h, 1)
        Bias vector for the second layer.
    dW1: numpy.ndarray
        Gradient of the cost with respect to W1.
    db1: numpy.ndarray
        Gradient of the cost with respect to b1.
    dW2: numpy.ndarray
        Gradient of the cost with respect to W2.
    db2: numpy.ndarray
        Gradient of the cost with respect to b2.
    learning_rate: float
        Learning rate of the gradient descent update rule.

    Returns:
    W1: numpy.ndarray, shape (n_h, n_x)
        Updated weight matrix for the first layer.
    b1: numpy.ndarray, shape (n_h, 1)
        Updated bias vector for the first layer.
    W2: numpy.ndarray, shape (n_y, n_h)
        Updated weight matrix for the second layer.
    b2: numpy.ndarray, shape (n_h, 1)
        Updated bias vector for the second layer.
    """
    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1
    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2

    return W1, b1, W2, b2