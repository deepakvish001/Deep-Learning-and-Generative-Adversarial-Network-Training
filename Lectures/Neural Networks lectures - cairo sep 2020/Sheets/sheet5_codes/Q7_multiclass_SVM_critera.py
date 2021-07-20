import numpy as np


def loss(pred, label):
    # d = np.zeros_like(pred)
    # for i in range(len(pred)):
    #     if i == label:
    #         d[i]=0
    #     else:
    #         d[i]=np.maximum(1 + pred[i] - pred[label], 0)

    ##### Faster way #####
    d = np.maximum(1 + pred - pred[label], 0)
    d[label] = 0
    return np.sum(d)


def grad_loss(inp, pred, label):
    delta = np.zeros((len(pred), len(inp)))
    d = np.maximum(1 + pred - pred[label], 0)
    d[label] = 0
    d = d > 0
    delta[d] = inp
    delta[label] = -inp * np.sum(d)
    return delta


# Weights for question 6
X = np.array(
    [
        [3, 0, 1],
        [4, 0, 1],
        [2, 0, 1],
        [3, 1, 1],
        [3, -1, 1],
        [-3, 0, 1],
        [-2, 0, 1],
        [-4, 0, 1],
        [-3, 1, 1],
        [-3, -1, 1],
        [0, 3, 1],
        [1, 3, 1],
        [-1, 3, 1],
        [0, 4, 1],
        [0, 2, 1],
        [0, -3, 1],
        [1, -3, 1],
        [-1, -3, 1],
        [0, -2, 1],
        [0, -4, 1],
    ]
)

y = (
    np.array([1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4]) - 1
)  # substracted one just for the indexing

# Weights from question 5
# X = np.array(
#     [
#         [0, 0, 1],
#         [0, 1, 1],
#         [1, 0, 1],
#         [2, 0, 1],
#         [2, 1, 1],
#         [3, 0, 1],
#         [0, 3, 1],
#         [0, 4, 1],
#         [1, 3, 1],
#     ]
# )

# y = np.array([1, 1, 1, 2, 2, 2, 3, 3, 3]) - 1

# Weights Dim => (Number of Output Nodes, input dimension + 1)
# Don't forget to adjust the weights dimensions based on the data
weights = np.random.randn(4, 3)

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
    print("Loss #{}: {}, delta: {}".format(step, lo, np.linalg.norm(delta)))
    delta = delta / len(X)
    weights = weights - delta


print("Final weights:\n", weights)
print("Num of Steps:", step)