import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt


def main(filename):
    dots = get_data(filename=filename)

    def func(coeff):
        return cost_function(coeff, dots)

    x0 = np.array([6.1, 1.2])
    res = scipy.optimize.minimize(func, x0, method='nelder-mead',
                                  options={'xtol': 1e-10, 'disp': True})

    a = res.x[0]
    b = res.x[1]

    plt.figure()

    plt.xlim(0, 80)
    plt.ylim(0, 150)

    arg = np.linspace(0, 150, 10)
    plt.scatter(dots[:, 0], dots[:, 1], color="blue")
    plt.plot(arg, a*arg + b, color="red")

    plt.show()


def get_data(filename):
    return np.genfromtxt(filename, delimiter=',')


def cost_function(coeff, dots):
    cost = 0
    N = len(dots)

    for i in range(N):
        x = dots[i][0]
        y = dots[i][1]

        cost += (y - (coeff[0] * x + coeff[1])) ** 2

    return cost / (2 * N)


main("linear_regression.csv")
