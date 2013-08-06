#
# Author: Joris Vankerschaver 2013
#
from __future__ import division

from functools import wraps
import numpy as np

__all__ = ['multivariate_normal']


_log_2pi = np.log(2 * np.pi)

def _process_arguments(dim, mean, cov):
    """
    Check dimensions mean and covariance matrix, and return full
    covariance matrix, if necessary.

    """
    if mean is None:
        mean = np.zeros(dim)
    mean, cov = map(np.asarray, (mean, cov))

    if len(mean) != dim:
        raise ValueError("Array 'mean' must be vector of length %d." % dim)
    if cov.ndim == 0:
        cov = cov * np.eye(dim)
    elif cov.ndim == 1:
        cov = np.diag(cov)
    else:
        if cov.shape != (dim, dim):
            raise ValueError("Array 'cov' must be at most two-dimensional,"
                                 " but cov.ndim = %d" % cov.ndim)
    return mean, cov


def process_arguments(f):
    """
    Process arguments passed to member functions of `multivariate_normal`.

    """ 
    @wraps(f)
    def _f(x, mean=None, cov=1):
        x = np.asarray(x)
        mean, cov = _process_arguments(x.shape[-1], mean, cov)
        return f(x, mean, cov)
    return _f


def _pseudo_det(mat, eps=1e-5):
    """
    Compute the pseudo-determinant of a symmetric positive semi-definite 
    matrix.

    The pseudo-determinant of a matrix is defined as the product of
    the non-zero eigenvalues, and coincides with the usual determinant
    for a full matrix. For reasons of efficiency, we (implicitly) 
    assume that the matrix is symmetric positive semi-definite and use
    the non-zero singular values to compute the pseudo-determinant, 
    rather than the eigenvalues. 

    Parameters
    ----------
    mat : array_like
        Input array of shape (`m`, `n`)
    eps : float, optional
        Threshold below which a singular value is considered to be zero.

    Returns
    -------
    det : float 
        Pseudo-determinant of the matrix.

    Notes
    -----
    The expression for the pseudo-determinant in terms of singular values
    rather than eigenvalues is only valid for matrices that are 
    symmetric positive semi-definite, but we do not check this.


    """ 
    s = np.linalg.svd(mat, compute_uv=False)
    return np.prod(s[s>eps])


class multivariate_normal(object):
    r"""
    A multivariate normal random variable.

    TODO : need to describe format for mean and cov variables

    Notes
    ----- 
    The covariance matrix `cov` must be a (symmetric) positive
    semi-definite matrix, but `multivariate_normal` will not check for
    this explicitly. The determinant and inverse of `cov` are computed 
    as the pseudo-determinant and pseudo-inverse, respectively, so that
    `cov` does not need to have full rank.

    The probability density function for `multivariate_normal` is

    .. math::

        f(x) = \frac{1}{\sqrt{(2 \pi)^k \det \Sigma}} \exp\left( -\frac{1}{2} (x - \mu)^T \Sigma^{-1} (x - \mu) \right),

    where :math:`\mu` is the mean, :math:`\Sigma` the covariance matrix, 
    and :math:`k` is the dimension of the space where :math:`x` takes values.

    """

    @staticmethod
    def _logpdf(x, mean, inv_cov, log_det_cov):
        """
        Log of the multivariate normal probability density function.

        Parameters
        ----------
        x : array_like
            Points at which to evaluate the log of the probability 
            density function
        mean : array_like
            Mean of the distribution
        inv_cov : array_like
            Inverse of the covariance matrix
        log_det_cov : float
            Logarithm of the determinant of the covariance matrix

        Returns
        -------
        pdf : ndarray
            Log of the probability density function evaluated at `x`

        """
        dim = x.shape[-1]
        dev = x - mean
        maha = np.einsum('...k,...kl,...l->...', dev, inv_cov, dev)
        return -0.5 * (dim * _log_2pi + log_det_cov + maha)

    @staticmethod
    @process_arguments
    def logpdf(x, mean, cov):
        """
        Log of the multivariate normal probability density function.

        Parameters
        ----------
        x : array_like
            Points at which to evaluate the log of the probability 
            density function
        mean : array_like, optional
            Mean of the distribution (default zero)
        cov : array_like, optional
            Covariance matrix of the distribution (default one)

        Returns
        -------
        pdf : ndarray
            Log of the probability density function evaluated at `x`

        """
        inv_cov = np.linalg.pinv(cov)
        log_det_cov = _pseudo_det(cov)
        return multivariate_normal._logpdf(x, mean, inv_cov, log_det_cov)

    @staticmethod
    @process_arguments
    def pdf(x, mean, cov):
        """
        Multivariate normal probability density function.

        Parameters
        ----------
        x : array_like
            Points at which to evaluate the probability density function
        mean : array_like, optional
            Mean of the distribution (default zero)
        cov : array_like, optional
            Covariance matrix of the distribution (default one)

        Returns
        -------
        pdf : ndarray
            Probability density function evaluated at `x`

        """
        return np.exp( multivariate_normal.logpdf(x, mean, cov) )


class multivariate_normal_frozen(object):

    def __init__(self, dim, mean=None, cov=1):

        mean, cov = _process_arguments(dim, mean, cov)

        self.mean = mean
        self.inv_cov = np.linalg.pinv(cov)
        


    def logpdf(self, x):
        return multivariate_normal(x, self.mean, self.cov)
