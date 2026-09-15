# The teacher's coding conventions

Extracted from the lecture notebooks (`week35tuesday`, `week36tuesday`,
`week37tuesday`, `week38monday`). Matching these makes the report easy for him to
read — and makes it obvious the work follows the course.

## Random numbers

```python
rng = np.random.default_rng(2026)     # modern Generator, seed 2026 everywhere
x = rng.uniform(-1.0, 1.0, n)
noise = rng.standard_normal(n)
```

Not `np.random.seed()` / `np.random.randn()` — that is the legacy API. (He does fall
back to it in `week36tuesday` because `sklearn.utils.resample` reads the global state.)

## Metrics — always `(y_data, y_model)` in that order

```python
def MSE(y_data, y_model):
    return np.mean((y_data - y_model) ** 2)

def R2(y_data, y_model):
    return 1.0 - np.sum((y_data - y_model) ** 2) / np.sum((y_data - np.mean(y_data)) ** 2)
```

## Design matrix

```python
X = np.vander(x, degree + 1, increasing=True)        # [1, x, x^2, ...] WITH intercept
X = np.column_stack([x**k for k in range(1, degree + 1)])   # no intercept column
```

## Fitting

```python
theta = np.linalg.lstsq(X, y, rcond=None)[0]         # OLS, SVD-based and stable

def closed_form(X, y, lam=0.0):
    """Eq. (3.44) with the 1/n convention of Eq. (3.95)."""
    n, p = X.shape
    return np.linalg.solve(X.T @ X + n * lam * np.eye(p), X.T @ y)
```

**The `n * lam` matters.** His cost function is `(1/n)||Xθ - y||² + λθᵀθ`, so the
normal equations pick up a factor `n`. That is also why the scikit-learn check is
`Ridge(alpha=n * lam, fit_intercept=False)`.

## Gradients and gradient descent

```python
def gradient(theta, X, y, lam=0.0):
    """Eqs. (4.13) and (4.17)."""
    n = len(y)
    return (2.0 / n) * X.T @ (X @ theta - y) + 2.0 * lam * theta

def gradient_descent(X, y, gamma, lam=0.0, num_iters=1000, tol=1e-8, theta0=None):
    """Plain gradient descent, Eq. (4.15). Returns (history, n_iterations)."""

def hessian_eigs(X, lam=0.0):
    """Eigenvalues of (2/n) X^T X + 2 lambda I, Eqs. (4.14) and (4.17)."""
    return np.linalg.eigvalsh((2.0 / len(X)) * X.T @ X + 2.0 * lam * np.eye(X.shape[1]))
```

Note `gamma` (γ) is his symbol for the learning rate, and `lam` / `lmbda` for λ.
Functions **return the full history** of iterates, not just the final θ — that is what
lets you plot convergence.

## Plotting — Paul Tol's colourblind-safe palette

```python
BLUE, RED, YELLOW, GREY, GREEN = "#004488", "#BB5566", "#DDAA33", "#777777", "#228833"

fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
ax.plot(degrees, mse_train, "o-", color=BLUE, label="training MSE")
ax.plot(degrees, mse_test,  "s-", color=RED,  label="test MSE")
ax.axhline(sigma**2, color="gray", ls=":", lw=1, label=r"$\sigma^2$")
ax.set_yscale("log")
ax.set_xlabel("polynomial degree"); ax.set_ylabel("MSE")
ax.legend(frameon=False, fontsize=8)
plt.tight_layout(); plt.show()
```

Markers: `"o-"` first series, `"s-"` second, `"d-"` third. Reference lines are always
grey, `ls=":"` or `ls="--"`, `lw=1`.

## Style habits worth copying

- Docstrings cite the lecture-note equation number: `"""Eq. (4.15)."""`
- `np.set_printoptions(precision=4, suppress=True)` at the top
- `np.round(theta, 4)` when printing
- Benchmarks printed as `max |ours - sklearn| = {...:.1e}`
- Helper functions take `(X, y, ...)` and return plain arrays — no classes
