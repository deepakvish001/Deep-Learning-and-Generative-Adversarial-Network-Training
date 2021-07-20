import numpy as np
import matplotlib.pyplot as plt


def visualize(func):

    x = np.linspace(-6, 6, 30)
    y = np.linspace(-6, 6, 30)

    X, Y = np.meshgrid(x, y)
    Z = func(X, Y)

    plt.figure()
    ax = plt.axes(projection="3d")
    ax.contour3D(X, Y, Z, 50, cmap="binary")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    plt.show()


def fun(x, y):
    return 3 * (x ** 4) + 3 * (x ** 2) * (y ** 2) + x ** 2 + 2 * (y ** 4)


def fun_x(x, y):
    return 12 * (x ** 3) + 6 * (x) * (y ** 2) + 2 * x


def fun_y(x, y):
    return 6 * (x ** 2) * (y) + 8 * (y ** 3)


def fun_xx(x, y):
    return 36 * (x ** 2) + 6 * (y ** 2) + 2


def fun_yy(x, y):
    return 6 * (x ** 2) + 24 * (y ** 2)


def fun_xy(x, y):
    return 12 * x * y


def fun_yx(x, y):
    return 12 * x * y


def GD_fun(x, y):
    return np.array([fun_x(x, y), fun_y(x, y)])


def hessian_fun(x, y):
    return np.array([[fun_xx(x, y), fun_xy(x, y)], [fun_yx(x, y), fun_yy(x, y)]])


alpha = 0.05
point = [1, 1]
epsilon = 0.0001

visualize(fun)

# # Vanilla Gradient Descent
# gradient_mag = np.linalg.norm(GD_fun(*point))
# step = 0
# print("point\tf(x,y)\tfn_x\tfn_y")
# while(gradient_mag > epsilon):
#     print("Step{}: {} {} {} {}".format(step, point, fun(*point), *GD_fun(*point)))
#     point = point - alpha*GD_fun(*point)
#     gradient_mag = np.linalg.norm(GD_fun(*point))
#     step += 1
#     # print(gradient_mag)

# print("Num of steps:", step)


# # Steepest Gradient Descent
# def fun_alpha(starting_point, alpha):
#     return fun(*(starting_point - alpha*GD_fun(*starting_point)))

# gradient_mag = np.linalg.norm(GD_fun(*point))
# step = 0
# domain_search = np.linspace(0,1,1000)
# print("point\tf(x,y)\talpha\tfn_x\tfn_y")
# while(gradient_mag > epsilon):
#     optimal_alpha = domain_search[np.argmin([fun_alpha(point, x) for x in domain_search])]
#     print("Step{}: {} {} {} {}".format(step, point, fun(*point), optimal_alpha, *GD_fun(*point)))
#     point = point - optimal_alpha*GD_fun(*point)
#     gradient_mag = np.linalg.norm(GD_fun(*point))
#     step += 1
#     # print(gradient_mag)

# print("Num of steps:", step)


# # Newton-Raphson Gradient Descent
# gradient_mag = np.linalg.norm(GD_fun(*point))
# step = 0
# print("point\tf(x,y)\talpha\tfn_x\tfn_y\tfn_xx\tfn_xy\tfn_yx\tfn_yy")
# while(gradient_mag > epsilon):
#     alpha = np.linalg.inv(hessian_fun(*point)) # shape (2,2)
#     print("Step{}: {} {} {} {} {} {} {} {}".format(step, point, fun(*point), *GD_fun(*point), fun_xx(*point), fun_xy(*point), fun_yx(*point), fun_yy(*point)))
#     point = point - np.dot(alpha,GD_fun(*point))
#     gradient_mag = np.linalg.norm(GD_fun(*point))
#     step += 1
#     # print(gradient_mag)

# print("Num of steps:", step)
