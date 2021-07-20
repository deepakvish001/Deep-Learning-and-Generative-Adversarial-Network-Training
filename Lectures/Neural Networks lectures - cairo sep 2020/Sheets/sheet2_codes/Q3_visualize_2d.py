from matplotlib import pyplot as plt
import numpy as np

# initiliaze a figure
plt.figure()

# add weights
weights = [-1, -1]
bias = 0.8

# plot line
x1 = np.linspace(-5, 5, 100)
x2 = (-weights[0] * x1 - bias) / weights[1]  # w1*x1 + w2*x2 + bias = 0
plt.plot(x1, x2, "r-")

# data from sheet
data = np.array([[-1, -1], [1, -1], [-1, 1], [1, 1]])
labels = [1, 1, 1, -1]

# limit graph
plt.hlines(0, -2, 2)
plt.vlines(0, -2, 2)
plt.xlim(-4, 4)
plt.ylim(-4, 4)

# Plot a line at each location specified in data
plt.scatter(data[:3, 0], data[:3, 1], color="blue")
plt.scatter(data[3, 0], data[3, 1], color="red")

plt.show()