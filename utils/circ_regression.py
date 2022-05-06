"""
Circular regression tools.
"""
import numpy as np

import scipy.stats
normcdf = scipy.stats.norm.cdf
normpdf = scipy.stats.norm.pdf

from scipy.linalg import cho_factor, cho_solve
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_random_state


class CircularRegression(BaseEstimator):
    """
    Reference
    ---------
    Brett Presnell, Scott P. Morrison and Ramon C. Littell (1998). "Projected Multivariate
    Linear Models for Directional Data". Journal of the American Statistical Association,
    Vol. 93, No. 443. https://www.jstor.org/stable/2669850
    
    Notes
    -----
    Only works for univariate dependent variable.
    """

    def __init__(
            self, alpha=1e-2, tol=1e-5, max_iter=100,
            fit_intercept=True, n_restarts=2, random_state=None):
        """
        Parameters
        ----------
        alpha : float
            Regularization parameter
        tol : float
            Convergence criterion for EM algorithm
        max_iter : int
            Maximimum number of EM iterations.
        fit_intercept : bool
            If True, a column of ones is appended to the independent variables
        """
        self.alpha = alpha
        self.tol = tol
        self.max_iter = max_iter
        self.fit_intercept = fit_intercept
        self.random_state = check_random_state(random_state)
        self.n_restarts = n_restarts

    def fit(self, X, y):
        """
        Uses EM algorithm in Presnell et al. (1998).
        Parameters
        ----------
        X : array
            Independent variables, has shape (n_timepoints x n_neurons)
        y : array
            Circular dependent variable, has shape (n_timepoints x 1),
            all data should lie on the interval [-pi, +pi].
        """

        if y.ndim != 1:
            raise ValueError("Expected 1d array for y.")

        # Add intercept.
        if self.fit_intercept:
            X = np.column_stack([X, np.ones(X.shape[0])])

        # Cache neuron x neuron gram matrix. This is used below
        # in the M-step to solve a linear least squares problem
        # in the form inv(XtX) @ XtY. Add regularization term to
        # the diagonal.
        XtX = X.T @ X
        XtX[np.diag_indices_from(XtX)] += self.alpha
        XtX = cho_factor(XtX)

        # Convert 1d circular variable to 2d representation
        u = np.column_stack([np.sin(y), np.cos(y)])
        best_log_like = -np.inf

        for rep in range(self.n_restarts):
            W, ll_hist = _fit_em(
                X, XtX, u, self.max_iter, self.tol, self.random_state
            )
            if ll_hist[-1] > best_log_like:
                best_log_like = ll_hist[-1]
                Wbest = W
                self.log_like_hist_ = ll_hist

        if self.fit_intercept:
            self.weights_ = Wbest[:-1]
            self.intercept_ = Wbest[-1]
        else:
            self.weights_ = Wbest
            self.intercept_ = None

    def predict(self, X):
        if self.fit_intercept:
            u_pred = X @ self.weights_ + self.intercept_[None, :]
        else:
            u_pred = X @ self.weights_
        return np.arctan2(u_pred[:, 0], u_pred[:, 1])

    def score(self, X, y):
        """
        Returns 1 minus mean angular similarity between y and model prediction.
        score == 1 for perfect predictions
        score == 0 in expectation for random predictions
        score == -1 if predictions are off by 180 degrees.
        """
        y_pred = self.predict(X)
        return np.mean(np.cos(y - y_pred))



def _fit_em(X, XtX, u, max_iter, tol, random_state):

    # Randomly initialize weights. Ensure scaling does
    W = random_state.randn(X.shape[1], 2)
    W /= np.max(np.sum(X @ W, axis=1))

    # Compute model prediction in 2d space, and projection onto
    # each observed u.
    XW = (X @ W)
    t = np.sum(u * XW, axis=1)
    tcdf = normcdf(t)
    tpdf = normpdf(t)

    log_like_hist_ = [
        np.log(2 * np.pi) - 
        0.5 * np.mean(np.sum(XW * XW, axis=1), axis=0) +
        np.mean(np.log(1 + t * tcdf / tpdf))
    ]

    for itr in range(max_iter):

        # E-step.
        m = t + (tcdf / (tpdf + t * tcdf))
        XtY = X.T @ (m[:, None] * u)

        # M-step.
        W = cho_solve(XtX, XtY)

        # Recompute model prediction.
        XW = X @ W
        t = np.sum(u * XW, axis=1)
        tcdf = normcdf(t)
        tpdf = normpdf(t)

        # Store log-likelihood. See sec. 4 of Presnell.
        log_like_hist_.append(
            np.mean(np.log(1 + t * tcdf / tpdf))
            - np.log(2 * np.pi)
            - 0.5 * np.mean(np.sum(XW * XW, axis=1), axis=0)
        )

        # Check convergence.
        if (log_like_hist_[-1] - log_like_hist_[-2]) < tol:
            break

    return W, log_like_hist_