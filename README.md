# FYS-STK3155/4155 — Applied Data Analysis and Machine Learning

Coursework for FYS-STK4155 at the University of Oslo, autumn 2026.

## Project 1 — Parisa and Gard

Project 1 is a joint project by **Parisa Amin** and **Gard Kvalsvik Lilleås**.

Polynomial regression on Runge's function f(x) = 1/(1+25x^2), using OLS, Ridge and
Lasso, with the bootstrap and k-fold cross-validation, and gradient descent as the
optimiser.

```
Project1/
  code/      regression.py and the notebooks
  results/   the figures used in the report
  report/    bibliography (the report itself is written in Overleaf)
```

We keep separate notebooks so they do not clash in git: `Project1.ipynb` is Parisa's
and `Project1_gard.ipynb` is Gard's. Shared functions go in `regression.py`.

## Running the code

```
pip install -r requirements.txt
```

All random seeds are fixed, so the figures can be reproduced from the notebooks.
