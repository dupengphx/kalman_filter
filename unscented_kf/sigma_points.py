"""
UFK filters are able to get precise the mean and covariance after non-linear system
with only a few sample points.
"""
from filterpy.kalman import unscented_transform, MerweScaledSigmaPoints
import numpy as np
from numpy.random import multivariate_normal
import matplotlib.pyplot as plt


def f_nonlinear_xy(x, y):
    return np.array([x + y, .1*x**2 + y*y])

def my_unscented_transform(sigmas_f, Wm, Wc):
    # calculate new mean
    mean = np.dot(Wm, sigmas_f)  
    y = sigmas_f - mean[np.newaxis, :]
    # calculate new variance
    variance = np.dot(y.T, np.dot(np.diag(Wc), y))

    return mean, variance

if __name__ == "__main__":
    # initialize state mean and state covariance
    mean = (0., 0.)
    p = np.array([[32., 15], [15., 40.]])

    ### 1.UKF ###
    # create sigma points and weights
    # the generation of sigma points only depends state dim n.
    # (2*n + 1) sigma points will be created.

    # gen points and points weights by state dim.
    points = MerweScaledSigmaPoints(n=2, alpha=.3, beta=2., kappa=.1)
    # gen points positions by given mean and covariance.
    sigmas = points.sigma_points(mean, p)

    # pass sigma points through nonlinear function, get sigmas_f.
    sigmas_f = np.empty((5, 2))
    for i in range(5):
        sigmas_f[i] = f_nonlinear_xy(sigmas[i, 0], sigmas[i ,1])

    # use unscented transform to get new mean and covariance
    ukf_mean, ukf_cov = unscented_transform(sigmas_f, points.Wm, points.Wc)
    print('ukf_mean: ', ukf_mean)
    my_ukf_mean, my_ukf_cov = my_unscented_transform(sigmas_f, points.Wm, points.Wc)
    print('my_ukf_mean: ', my_ukf_mean)

    ### 2. Monte Carlo method ###
    # generate 10000 random points and cal mean and covariance after non-linear transform.
    xs, ys = multivariate_normal(mean=mean, cov=p, size=10000).T
    x_out, y_out = f_nonlinear_xy(xs, ys)
    print('monte carlo mean: ', [np.mean(x_out), np.mean(y_out)])


    ### 3. visualize ###
    # visualize sigma points of UKF and monte carlo.
    fig = plt.figure()
    plt.scatter(xs, ys)
    plt.scatter(sigmas[:, 0], sigmas[:, 1], c='r')
    plt.show()

    pass

    # visualize sigma points after non-linear function of UKF and monte carlo.
    fig2 = plt.figure()
    plt.scatter(x_out, y_out)
    plt.scatter(ukf_mean[0], ukf_mean[1], c='r')
    plt.scatter(np.mean(x_out), np.mean(y_out), c='g')
    plt.show()

    pass



