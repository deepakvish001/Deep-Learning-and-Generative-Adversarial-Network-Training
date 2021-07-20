import numpy as np
import matplotlib.pyplot as plt


# initiliaze a figure
plt.figure()
ax = plt.axes(projection="3d")


data = [
    [0, 0, 0],
    [255, 0, 0],
    [0, 255, 0],
    [0, 0, 255],
    [255, 255, 0],
    [0, 255, 255],
    [255, 0, 255],
    [255, 255, 255],
]
labels = [1, 1, 1, -1, 1, -1, -1, 1]

data = np.array(data)
labels = np.array(labels)
class1_points = data[np.where(labels == 1, 1, 0).astype(bool)]
class2_points = data[np.where(labels == -1, 1, 0).astype(bool)]

# plot data points
ax.scatter3D(
    class1_points[:, 0],
    class1_points[:, 1],
    class1_points[:, 2],
    marker="o",
    color="red",
)
ax.scatter3D(
    class2_points[:, 0],
    class2_points[:, 1],
    class2_points[:, 2],
    marker="o",
    color="blue",
)


weights = np.array([0, 0, 0, 0])
delta = np.array([0, 0, 0, 0])
epsilon = 0.8

# # bipolar perceptron criterion
# # Online Training
# step = 0
# while np.linalg.norm(delta) > epsilon or step == 0:
#     delta = np.array([0, 0, 0, 0])
#     for i in range(len(data)):
#         pred = labels[i] * np.dot(weights, np.array([1, *data[i]]))
#         if pred <= 0:
#             delta = delta - labels[i] * np.array([1, *data[i]])
#             weights = weights - delta / 8
#             step += 1
#             print("Step{}: delta = {}, weights = {}".format(step, delta, weights))


#############################################################################
# # SVM criterion
# # Online Training
# step = 0
# while (np.linalg.norm(delta) > epsilon or step==0):
#     delta = np.array([0, 0, 0, 0])
#     for i in range(len(data)):
#         pred = labels[i] * np.dot(weights, np.array([1, *data[i]]))
#         if pred <= 1:
#             delta = delta - labels[i] * np.array([1, *data[i]])
#             weights = weights - delta / 8

#     step += 1
#     print("Step{}: delta = {}, weights = {}".format(step, delta, weights))


#############################################################################
# SVM criterion
# Batch Training
step = 0
while np.linalg.norm(delta) > epsilon or step == 0:
    delta = np.array([0, 0, 0, 0])
    for i in range(len(data)):
        pred = labels[i] * np.dot(weights, np.array([1, *data[i]]))
        if pred <= 1:
            delta = delta - labels[i] * np.array([1, *data[i]])
    delta = delta / 8
    weights = weights - delta

    step += 1
    print("Step{}: delta = {}, weights = {}".format(step, delta, weights))

x1 = np.linspace(-40, 300, 400)
x2 = np.linspace(-40, 300, 400)
X, Y = np.meshgrid(x1, x2)

Z = (-weights[1] * X - weights[2] * Y - weights[0]) / (weights[3] + 0.0001)
Z[Z > 400] = np.nan

# plot classification plane
ax.plot_surface(X, Y, Z, alpha=0.5)
ax.set_zlim(0, 300)
ax.set_xlabel("x1")
ax.set_ylabel("x2")
ax.set_zlabel("x3")

plt.show()