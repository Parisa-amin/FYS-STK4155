# FYS-STK3155/4155 — Applied Data Analysis and Machine Learning

Coursework for [FYS-STK4155](https://www.uio.no/studier/emner/matnat/fys/FYS-STK4155/),
University of Oslo, autumn 2026.

**Parisa Amin** — Industrial PhD candidate, Li-Tech AS × UiO Department of Physics.

## Contents

| Folder | Contents |
| --- | --- |
| `Project1/` | Regression, resampling methods and gradient descent |
| `exercises/` | Weekly exercise notebooks (weeks 34–) |

### Project 1 — Regression, resampling and gradient descent

Polynomial regression on Runge's function *f(x) = 1/(1+25x²)* using ordinary least
squares, Ridge and Lasso; model assessment with the bootstrap and k-fold
cross-validation; and gradient-descent optimisers (plain, momentum, AdaGrad, RMSprop,
Adam, and stochastic gradient descent) with gradients computed both analytically and by
automatic differentiation.

```
Project1/
├── report/     LaTeX source and the final report PDF
├── code/       Python modules and notebooks
└── results/    Figures, and selected runs not shown in the report
```

## Reproducing the results

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

All random number generators are seeded, so every figure in the report is reproducible
from the code in `Project1/code/`.
