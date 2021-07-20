import numpy as np
import matplotlib.pyplot as plt
from time import sleep

# initiliaze a figure
plt.ion()
fig = plt.figure()
ax = fig.add_subplot(111)

def loss_fn(prediction, label):
    return max(0, -label*prediction)


def grad_loss_fn(prediction, label, inp):
    if label*prediction > 0:
        return np.array([0])
    else:
        return np.array([-label*inp])

# Add your data
data = [[-1, -1, 1], [1, -1, 1], [-1, 1, 1], [1, 1, 1]]
data = np.array(data).reshape(len(data), -1) # shape (4,3)

# Add your labels
labels = [1, 1, 1, -1]
labels = np.array(labels).reshape(len(labels), -1) # shape (4,1)

# Add your weights
weights = [-1, -1, 3]
weights = np.array(weights).reshape(len(weights), -1) # shape (3,1)
epsilon = 0.0001
alpha = 0.1

# plot line
x1 = np.linspace(-5, 5, 100)
x2 = (-weights[0] * x1 - weights[2]) / weights[1]  # w1*x1 + w2*x2 + bias = 0
line, = ax.plot(x1, x2, "r-")

# limit graph
plt.hlines(0, -2, 2)
plt.vlines(0, -2, 2)
plt.xlim(-4, 4)
plt.ylim(-4, 4)

# Plot a line at each location specified in data
ax.scatter(data[:3, 0], data[:3, 1], color="blue")
ax.scatter(data[3, 0], data[3, 1], color="red")

pred = np.dot(data[0], weights) # shape (1,1)
loss = 10
step = 0
while loss > epsilon:
    # Stochastic Gradient Descent
    loss = 0
    for i in range(len(data)):
        pred = np.dot(data[i], weights) # shape (1,1) 
        weights = weights - alpha * grad_loss_fn(pred, labels[i], data[i]).T # shape(3,1)
        loss += float(loss_fn(pred, labels[i]))
    print("weights: {}, loss: {}".format(weights.T, loss))
    sleep(1)
    line.set_ydata((-weights[0] * x1 - weights[2]) / weights[1])
    fig.canvas.draw()
    fig.canvas.flush_events()
    step += 1

print("Num of steps:", step)