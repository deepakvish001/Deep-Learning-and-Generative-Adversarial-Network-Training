import numpy as np

# Add your data and labels
X = np.array(
    [
        [0, 0, 0],
        [255, 0, 0],
        [0, 255, 0],
        [0, 0, 255],
        [255, 255, 0],
        [255, 0, 255],
        [0, 255, 255],
        [255, 255, 255],
    ]
)
y = np.array([1, 1, 1, -1, 1, -1, -1, 1])

# Initiliaze Weights
weights = np.zeros((1, 4))

# learning rate & epsilon
alpha = 1
epsilon = 0.2

# Normalization for fast training
X = X/255

step = 0
while True:
    step += 1
    # Making sure that delta is zero every while loop iteration
    delta = np.zeros_like(weights)
    for i in range(len(X)):
        # Linear Regression Output
        z = np.dot(weights, np.array([*X[i], 1]))
        # Updating delta with gradient loss
        delta = delta - (y[i] * np.array([*X[i], 1])) / (1 + np.exp(y[i] * z))
    delta = delta / 4
    weights = weights - alpha * delta
    print("Step{}: delta = {}, weights = {}".format(step, delta, weights))
    if np.linalg.norm(delta) < epsilon:
        break
