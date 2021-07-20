import numpy as np

# Preceptron Loss
def loss(pred, label):
    d = np.zeros(len(pred))
    for i in range(len(pred)):
        if i == label:
            d[i] = 0
        else:
            d[i] = np.max(pred[i] - pred[label], 0)
    return np.max(d)


# Loss Gradient with a trick
def grad_loss(inp, pred, label):
    r = np.argmax(pred)  # WX
    d = np.zeros((len(pred), len(inp)))
    if r != label:
        d[r] = inp
        d[label] = -inp
    return d


X = np.array(
    [
        [0, 0, 1],
        [0, 1, 1],
        [1, 0, 1],
        [2, 0, 1],
        [2, 1, 1],
        [3, 0, 1],
        [0, 3, 1],
        [0, 4, 1],
        [1, 3, 1],
    ]
)

y = np.array([1, 1, 1, 2, 2, 2, 3, 3, 3]) - 1

# Weights Dim => (Number of Output Nodes, input dimension + 1)
weights = np.random.randn(3, 3)

# Initilizations
epsilon = 0.1
step = 0
delta = np.zeros_like(weights)
while np.linalg.norm(delta) > epsilon or step == 0:
    delta = np.zeros_like(weights)
    step += 1
    lo = 0
    for i in range(len(X)):
        pred = np.dot(weights, X[i])
        lo += loss(pred, y[i])
        delta = delta + grad_loss(X[i], pred, y[i])
    print("Loss #{}: {}".format(step, lo))
    delta = delta / len(X)
    weights = weights - delta


print("Final weights:\n", weights)
print("Num of Steps:", step)