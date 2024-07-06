from filterpy.kalman import unscented_transform, MerweScaledSigmaPoints
import scipy.stats as stats
import numpy as np
from numpy.random import multivariate_normal
import kf_book
from kf_book.nonlinear_plots import plot_monte_carlo_mean


def f_nonlinear_xy(x, y):
    return np.array([x + y, .1*x**2 + y*y])

if __name__ == "__main__":
    # initialize state mean and state covariance
    mean = (0., 0.)
    p = np.array([[32., 15], [15., 40.]])

    ### 1.UK F###
    # create sigma points and weights
    # the generation of sigma points only depends state dim.
    points = MerweScaledSigmaPoints(n=2, alpha=.3, beta=2., kappa=.1)
    sigmas = points.sigma_points(mean, p)

    # pass sigma points through nonlinear function
    sigmas_f = np.empty((5, 2))
    for i in range(5):
        sigmas_f[i] = f_nonlinear_xy(sigmas[i, 0], sigmas[i ,1])

    # use unscented transform to get new mean and covariance
    ukf_mean, ukf_cov = unscented_transform(sigmas_f, points.Wm, points.Wc)

    #generate random points
    # xs, ys = multivariate_normal(mean=mean, cov=p, size=10000).T