import numpy as np

# Add your data and labels
X = np.array([[-1, -1, 1], [1, -1, 1], [-1, 1, 1], [1, 1, 1]])
y = np.array([1, 1, 1, -1])

# Initiliaze Weights
weights = np.zeros((1,3))

# learning rate & epsilon
alpha = 1
epsilon = 0.2

step = 0
while True:
    step += 1
    delta = np.zeros_like(weights)
    for i in range(len(X)):
        delta = delta - (y[i]*X[i])/(1+np.exp(y[i]*np.dot(weights, X[i])))
    delta = delta/4
    weights = weights - alpha*delta
    print("Step{}: delta = {}, weights = {}".format(step, delta, weights))
    if (np.linalg.norm(delta) < epsilon):
        break