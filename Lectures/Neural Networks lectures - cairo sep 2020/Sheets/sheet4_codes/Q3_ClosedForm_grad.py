import numpy as np


def loss(y, X, w):
    L = 0.5 * np.dot(np.transpose((y - np.dot(X, w))), (y - np.dot(X, w))) * (1/y.shape[0])
    return L


def grad_loss(x, w, y):
    return np.dot(np.dot(np.transpose(x), x), w) - np.dot(np.transpose(x), y)



X = np.array([[0, 1], [1, 1], [2, 1], [3, 1], [4, 1], [5, 1], [6, 1], [7, 1], [8, 1]])
y = np.array([0, 0.81, 0.95, 0.31, -0.59, -1, -0.59, 0.31, 0.95])
w = np.array([0, 0])
for i in range(5):
    lossk = loss(y, X, w)
    delta = grad_loss(X, w, y) / 9
    w = w - 0.01 * delta
    print("Weights   :   ", w)
    print("loss   :   ", lossk)
    print("delta   :   ", delta)
    print("----------------------------------------------------------")
