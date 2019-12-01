import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def f(x, y):
    # z = 2*np.sin(1.5*x**2 - 0.25*y + 0.25*np.pi) + 3*np.cos(1.5*x*y - 0.5*np.pi)
    z = np.sin(x) + np.cos(y)
    return z

if __name__ == '__main__':
    t = np.linspace(0, np.pi*2, 50)
    x1, y1 = np.meshgrid(t, t)
    z = np.stack([x1.flat, y1.flat], axis=1)
    # print(z.shape)
    x = z[:, 0]
    y = z[:, 1]
    # print(x)
    z = f(x, y)
    # print(z)
    z = z.reshape(x1.shape)
    # print(z.shape)
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.plot_surface(x1, y1, z, rstride=1, cstride=1, cmap='rainbow')
    plt.xlabel("x")
    plt.ylabel("y")
    ax.set_zlabel("z")

    x0 = 2
    y0 = 0.5
    n = 0.5
    xy = []
    for i in range(100):
        xn = x0 - n*np.cos(x0)
        yn = y0 + n*np.sin(y0)
        x0 = xn
        y0 = yn
        xy.append([x0, y0])
    xy = np.array(xy)
    ax.plot(xy[:, 0], xy[:, 1], f(xy[:, 0], xy[:, 1]), 'k-' ,linewidth=5)
    ax.plot(xy[:, 0], xy[:, 1], f(xy[:, 0], xy[:, 1]), 'k*' ,linewidth=20)
    plt.show()