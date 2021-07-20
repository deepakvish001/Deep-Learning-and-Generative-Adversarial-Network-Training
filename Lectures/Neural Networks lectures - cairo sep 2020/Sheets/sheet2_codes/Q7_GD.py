import numpy as np

# Choose the data you want to try

# data = [50, 55, 70, 80, 130, 150, 155, 160] # 1D
# data = [[-1, -1], [1, -1], [-1, 1], [1, 1]] # 2D
data = [(0, 0, 0), (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255), (255, 0, 255), (255, 255, 255)] # 3D

# labels = [1, 1, 1, 1, -1, -1, -1, -1] # 1D
# labels = [1, 1, 1, -1] # 2D
labels = [1, 1, 1, -1, 1, -1, -1, 1] # 3D
epsilon = 0.1
data = np.array(data).reshape(len(data), -1)

weights = np.zeros(data.shape[1]+1)
delta = np.zeros(data.shape[1]+1)

# Batch Training
step = 0
while(np.linalg.norm(delta) > epsilon or step==0):
    delta = np.zeros(data.shape[1]+1)
    for i in range(len(data)):
        pred = np.dot(weights, np.array([1,*data[i]]))
        if labels[i]*pred <= 0:
            delta = delta - labels[i]*np.array([1,*data[i]])
    delta = delta / len(data)
    weights = weights - delta

    print("Step{}: delta = {}, weights = {}, ".format(step, delta, weights))
    step += 1
    # input()
