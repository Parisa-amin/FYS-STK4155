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
    rng= np.random.default_rng(seed)

    if uniform:
        x=rng.uniform(-1, 1, n)
    else:
        x=np.linspace(-1, 1, n)

    y= runge(x) + sigma * rng.standard_normal(n)
    return x, y
    


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
    X= np.vander(x, degree + 1, increasing=True)
    if not intercept:
        X= X[:, 1:]
    return X
   

# ---------------------------------------------------------------------------
# Step 3: the metrics
# ---------------------------------------------------------------------------

def MSE(y_data, y_model):
    """Mean squared error."""
    
    return np.mean((y_data - y_model)**2 )


def R2(y_data, y_model):
    """The R^2 score."""
    numerator = np.sum (( y_data - y_model)**2)
    denominator= np.sum((y_data - np.mean(y_data))**2)

    return 1- numerator/denominator 


# ---------------------------------------------------------------------------
# Step 4: ordinary least squares
# ---------------------------------------------------------------------------

def ols(X, y):
    """OLS coefficients via the pseudoinverse / SVD.

    The project asks explicitly for our own code here, using ``np.linalg.pinv``
    or the SVD, rather than scikit-learn.
    """
    return np.linalg.pinv(X) @ y




# ---------------------------------------------------------------------------
# step 5: Build polynomial features, scale, fit OLS/Ridge, and predict.
# ---------------------------------------------------------------------------

def fit_predict(x_train, x_test, y_train, degree, lam=0.0):
    """Build features, scale on the training set, fit OLS/Ridge, and predict.

    Eq. (3.44) with the 1/n convention of Eq. (3.95): the cost function is
    (1/n)||X theta - y||^2 + lambda ||theta||^2, so the normal equations pick
    up a factor n on the ridge term.

    Returns
    -------
    y_train_pred, y_test_pred, theta
    """
    X_train= polynomial_features(x_train, degree)
    X_test= polynomial_features(x_test, degree)


    X_mean= np.mean(X_train, axis=0)
    X_std= np.std(X_train, axis=0)


    X_train_scaled= (X_train - X_mean) / X_std
    X_test_scaled= (X_test - X_mean) / X_std

    #center y using the training mean
    y_mean= np.mean(y_train)
    y_train_centered= y_train - y_mean

    if lam == 0.0:
        #OLS
        theta= ols(X_train_scaled, y_train_centered)
    else:
        n, p = X_train_scaled.shape

        theta = np.linalg.solve(
            X_train_scaled.T @ X_train_scaled + n * lam * np.eye(p),
            X_train_scaled.T @ y_train_centered,
        )


    #predict and return to the original y 
    y_train_pred = X_train_scaled @ theta + y_mean
    y_test_pred = X_test_scaled @ theta + y_mean

    return y_train_pred, y_test_pred , theta





