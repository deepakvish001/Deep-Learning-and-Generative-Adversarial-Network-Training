import numpy as np
from texttable import Texttable

# Initializing a table
table = Texttable()
table.header(["Iteration", "Weights", "Loss", "delta", "||delta||"])

X = np.arange(0, 9)
Y = np.array([0, 0.81, 0.95, 0.31, -0.59, -1, -0.59, 0.31, 0.95], ndmin=2).T

# y_hat = Xw
def model(x,W):
    return np.dot(x, W)

# L=(1/2N)(y-X.w)_T(y-X.w)
def loss_fn(y, pred):
    return 0.5*np.dot((y-pred).T,(y-pred))*(1/len(y))

# ∇L=(1/N)(X_T.X.w-X_T.y)
def grad_loss(y,pred,x):
    return (np.dot(x.T, pred)-np.dot(x.T, y))*(1/len(y))

# ∇∇L=(1/N)(X_T.X)
def hessian_loss(x):
    return np.dot(x.T, x)*(1/len(x))

# Play with different parameter numbers here
M = 10
W = np.zeros((M,1))
inp = np.array([X**i for i in range(M)]).T

step = 0
epsilon = 0.8
alpha = np.linalg.inv(hessian_loss(inp))
delta = 0
while delta > epsilon or step == 0:
    W = W - np.dot(alpha,grad_loss(Y, model(inp, W), inp))
    delta = np.linalg.norm(grad_loss(Y, model(inp, W), inp))
    table.add_row([step, np.around(W.T, 2), loss_fn(Y, model(inp, W)), np.around(grad_loss(Y, model(inp, W), inp).T, 2), delta])
    step += 1
    # print("Step{}: Weights:{}, Loss:{}, delta:{}, ||delta||:{}".format(*[step, np.around(W.T, 2), loss_fn(Y, model(inp, W)), np.around(grad_loss(Y, model(inp, W), inp).T, 2), delta]))

print(table.draw())