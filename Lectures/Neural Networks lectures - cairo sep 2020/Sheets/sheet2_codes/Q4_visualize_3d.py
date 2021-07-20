import numpy as np
import matplotlib.pyplot as plt

# initiliaze a figure
plt.figure()
ax = plt.axes(projection="3d")

# data from Q4
data = np.array([(0, 0, 0), (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255), (255, 0, 255), (255, 255, 255)])
labels = np.array([1, 1, 1, -1, 1, -1, -1, 1])

class1_points = data[np.where(labels==1,1,0).astype(bool)]
class2_points = data[np.where(labels==-1,1,0).astype(bool)]

# plot data points
ax.scatter3D(class1_points[:,0], class1_points[:,1], class1_points[:,2], marker='o', color="red")
ax.scatter3D(class2_points[:,0], class2_points[:,1], class2_points[:,2], marker='o', color="blue")

# add weights
weights = [1, 1, -2]
bias = 100

print(np.sign(np.dot(weights,data.T) + bias))
x1 = np.linspace(-40, 300, 400)
x2 = np.linspace(-40, 300, 400)
X, Y = np.meshgrid(x1, x2)

# w1*x1 + w2*x2 + w3*x3 + bias = 0
Z = (-weights[0] * X - weights[1] * Y - bias) / (weights[2] + 0.0001)
Z[Z>400] = np.nan

# plot classification plane
ax.plot_surface(X, Y, Z, alpha=0.5)
ax.set_zlim(0,300)
ax.set_xlabel("x1")
ax.set_ylabel("x2")
ax.set_zlabel("x3")

plt.show()