"""Regression tools for Project 1 -- Runge's function.

FYS-STK3155/4155, autumn 2026.

"""

import numpy as np

# Paul Tol's colourblind-safe palette, as used in the weekly slides
BLUE, RED, YELLOW, GREY, GREEN = "#004488", "#BB5566", "#DDAA33", "#777777", "#228833"


# ---------------------------------------------------------------------------
# Step 1: the data
# ---------------------------------------------------------------------------

def runge(x):
    """Runge's function f(x) = 1 / (1 + 25 x^2)."""
    # TODO: one line.
    return 1 / (1 + 25 * x**2)


def make_data(n=100, sigma=0.1, seed=2026, uniform=True):
    """Sample Runge's function on [-1, 1] and add N(0, sigma^2) noise.

    Parameters
    ----------
    n : int
        Number of data points.
    sigma : float
        Standard deviation of the noise.
    seed : int
        Seed for the random number generator.
    uniform : bool
        If True draw x uniformly on [-1, 1]; if False use a fixed step size.

    Returns
    -------
    x, y : ndarray of shape (n,)
    """
    rng = np.random.default_rng(seed)
    # TODO:
    #   1. build x -- rng.uniform(...) if uniform, else np.linspace(...)
    #   2. y = runge(x) + sigma * rng.standard_normal(n)
    #   3. return x, y
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Step 2: the design matrix
# ---------------------------------------------------------------------------

def polynomial_features(x, degree, intercept=False):
    """Vandermonde design matrix in x.

    With ``intercept=True`` the columns are [1, x, x^2, ..., x^degree] and the
    matrix has ``degree + 1`` columns.  With ``intercept=False`` the constant
    column is dropped, leaving ``degree`` columns -- the form we want when the
    data have been centred.
    """
    # TODO: np.vander(x, degree + 1, increasing=True), then drop column 0
    #       if intercept is False.
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Step 3: the metrics
# ---------------------------------------------------------------------------

def MSE(y_data, y_model):
    """Mean squared error."""
    # TODO
    raise NotImplementedError


def R2(y_data, y_model):
    """The R^2 score."""
    # TODO
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Step 4: ordinary least squares
# ---------------------------------------------------------------------------

def ols(X, y):
    """OLS coefficients via the pseudoinverse / SVD.

    The project asks explicitly for our own code here, using ``np.linalg.pinv``
    or the SVD, rather than scikit-learn.
    """
    # TODO: np.linalg.pinv(X) @ y
    raise NotImplementedError
