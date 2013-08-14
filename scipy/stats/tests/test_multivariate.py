"""
Test functions for multivariate normal distributions.

"""
from __future__ import division, print_function, absolute_import

from numpy.testing import run_module_suite, assert_allclose

import numpy
import numpy as np

from scipy.stats import multivariate_normal 
from scipy.stats import norm

from scipy.integrate import romb

def test_normal_1D():
    # The probability density function for a 1D normal variable should 
    # agree with the standard normal distribution in scipy.stats.distributions
    x = np.linspace(0, 2, 10)
    mean = 1.2; cov = 0.9; scale=cov**0.5
    d1 = norm.pdf(x, mean, scale)
    d2 = multivariate_normal.pdf(x, mean, cov)
    assert_allclose(d1, d2)

def test_marginalization():
    # Integrating out one of the variables of a 2D Gaussian should
    # yield a 1D Gaussian
    mean = np.array([2.5, 3.5])
    cov = np.array([[.5, 0.2], [0.2, .6]])
    n = 2**8 + 1 # Number of samples
    delta = 6 / (n - 1) # Grid spacing

    v = np.linspace(0, 6, n)
    xv, yv = np.meshgrid(v, v)
    pos = np.empty((n, n, 2))
    pos[:, :, 0] = xv
    pos[:, :, 1] = yv
    pdf = multivariate_normal.pdf(pos, mean, cov)

    # Marginalize over x and y axis
    margin_x = romb(pdf, delta, axis=0)
    margin_y = romb(pdf, delta, axis=1)

    # Compare with standard normal distribution
    gauss_x = norm.pdf(v, loc=mean[0], scale=cov[0,0]**0.5)
    gauss_y = norm.pdf(v, loc=mean[1], scale=cov[1,1]**0.5)
    assert_allclose(margin_x, gauss_x, rtol=1e-2, atol=1e-2)
    assert_allclose(margin_y, gauss_y, rtol=1e-2, atol=1e-2)

if __name__ == "__main__":
    run_module_suite()
