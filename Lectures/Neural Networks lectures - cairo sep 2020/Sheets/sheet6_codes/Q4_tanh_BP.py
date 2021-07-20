import numpy as np


def loss(pred, label):
    return 0.5 * np.dot((pred - label).T, (pred - label))


def grad_loss(pred, label):
    return pred - label


def act_fn(x):
    return np.tanh(x)


def grad_act_fn(x):
    return 1 - act_fn(x) ** 2


if __name__ == "__main__":
    # Play with values
    inp = np.array([1, 0.1], ndmin=2).T

    V = np.array([[0.5, 1], [1, 0.5]])
    b_v = np.array([0.5, 0.5], ndmin=2).T

    W = np.array([[0.5, 1], [1, 0.5]])
    b_w = np.array([0.5, 0.5], ndmin=2).T

    label = np.array([0.6, 0.01], ndmin=2).T

    alpha = 0.5

    # forward pass
    a = np.dot(V, inp) + b_v
    y = act_fn(a)
    z = np.dot(W, y) + b_w
    o = act_fn(z)

    # backward pass
    del_loss_del_output = grad_loss(o, label)  # dL/dO
    del_output_del_z = grad_act_fn(z)  # dO/dZ
    del_loss_del_z = del_loss_del_output * del_output_del_z  # dL/dZ
    del_z_del_W = y.T  # dZ/dW
    del_w = np.dot(del_loss_del_z, del_z_del_W)  # dL/dW
    del_b_w = del_loss_del_z  # dL/db_w

    del_z_del_y = W.T  # dZ/dY
    del_y_del_a = grad_act_fn(a)  # dY/dA
    del_loss_del_a = np.dot(del_z_del_y, del_loss_del_z) * del_y_del_a  # dL/dA
    del_a_del_V = inp.T  # dA/dV
    del_v = np.dot(del_loss_del_a, del_a_del_V)  # dL/dV
    del_b_v = del_loss_del_a  # dL/db_v

    # Update Rule
    W = W - alpha * del_w
    b_w = b_w - alpha * del_b_w
    print("New W:", np.array2string(W, prefix="New W: "))
    print("New b_w:", np.array2string(b_w, prefix="New b_w: "))
    print("===================================")
    V = V - alpha * del_v
    b_v = b_v - alpha * del_b_v
    print("New V:", np.array2string(V, prefix="New V: "))
    print("New b_v:", np.array2string(b_v, prefix="New b_v: "))