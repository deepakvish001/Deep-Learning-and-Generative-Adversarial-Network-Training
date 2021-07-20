from matplotlib import pyplot as plt
import numpy as np

# initiliaze a figure
plt.figure()

# add weights
weight = 1
bias = 100

plt.vlines(bias,0.8,1.2, color="yellow")

data = np.array([50, 55, 70, 80, 130, 150, 155, 160])
labels = [1, 1, 1, 1, -1, -1, -1, -1]

data = weight*data
plt.hlines(1,0,2*np.max(data))  # Draw a horizontal line (y, xmin, xmax)
plt.xlim(0,2*np.max(data)) # limit x-axis
plt.ylim(0.5,1.5) # limit y-axis

y = np.ones(np.shape(data)) # Make all y values the same

# Plot a line at each location specified in data
plt.plot(data[:4],y[:4],'r|',ms = 20)
plt.plot(data[4:],y[4:],'b|',ms = 30)
plt.show()