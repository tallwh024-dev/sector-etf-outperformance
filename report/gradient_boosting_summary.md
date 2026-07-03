# Gradient Boosting Summary

This note summarizes Jackson Wang's Gradient Boosting portion of the sector ETF outperformance project.

## Objective

Predict whether a U.S. sector ETF will outperform SPY in the following month.

The target is:

- `1`: sector ETF next-month return > SPY next-month return
- `0`: otherwise

## Model

I used `sklearn.ensemble.GradientBoostingClassifier` because sector ETF outperformance may depend on nonlinear interactions among momentum, volatility, drawdown, interest rates, VIX, and yield-curve conditions.

The model outputs a predicted probability of next-month outperformance. These probabilities can be used for ranking sector ETF candidates in a sector rotation framework.

## Selected hyperparameters

| Hyperparameter | Value |
|---|---:|
| `n_estimators` | 50 |
| `learning_rate` | 0.03 |
| `max_depth` | 1 |
| `min_samples_leaf` | 50 |
| `subsample` | 0.8 |

## Performance

| Dataset | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Train | 0.5702 | 0.5824 | 0.1970 | 0.2944 | 0.6511 |
| Validation | 0.4458 | 0.4375 | 0.1654 | 0.2400 | 0.4716 |
| Test | 0.5429 | 0.3818 | 0.1429 | 0.2079 | 0.5033 |

## Interpretation

The Gradient Boosting model did not generalize strongly to the 2023-2025 test period. Test ROC-AUC was close to random, and the model had low recall for outperforming ETFs. Feature importance was concentrated in a few variables, especially 12-month ETF momentum, the XLK sector dummy, and 1-month ETF momentum. This suggests the model relied heavily on historical momentum and technology-sector strength, which may not have transferred well across later market regimes.
