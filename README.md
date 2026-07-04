# Sector ETF Outperformance Prediction - Gradient Boosting

This repository contains my Gradient Boosting portion of a CFRM 521 Machine Learning for Finance project.

The objective is to predict whether a U.S. sector ETF will outperform SPY in the following month. The model is framed as a binary classification problem and uses monthly sector ETF features, SPY benchmark features, Treasury-rate variables, VIX, and ETF identity indicators.

## My contribution

**Implemented by:** Jackson Wang  
**Model:** Gradient Boosting Classifier

This repository includes my Gradient Boosting work and the data preparation steps needed to reproduce that model.

## Project overview

The target variable equals:

- `1` if the sector ETF's next-month return is greater than SPY's next-month return
- `0` otherwise

The model uses a chronological split to reduce look-ahead bias:

| Partition | Date Range | Observations |
|---|---:|---:|
| Training | 2016-01-31 to 2020-12-31 | 591 |
| Validation | 2021-01-31 to 2022-12-31 | 240 |
| Test | 2023-01-31 to 2025-11-30 | 350 |

## Features

The feature set includes:

- ETF momentum: 1-month, 3-month, 6-month, and 12-month
- ETF volatility: 3-month, 6-month, and 12-month
- ETF 12-month drawdown
- SPY momentum, volatility, and drawdown features
- Macroeconomic and market-regime indicators:
  - 10-year Treasury yield
  - 2-year Treasury yield
  - Effective federal funds rate
  - VIX
  - 10-year minus 2-year yield spread
  - 1-month changes in macro variables
- ETF dummy variables

## Gradient Boosting model

The model was implemented with `sklearn.ensemble.GradientBoostingClassifier`.

The validation set was used for hyperparameter tuning. The selected model used:

| Hyperparameter | Value |
|---|---:|
| `n_estimators` | 50 |
| `learning_rate` | 0.03 |
| `max_depth` | 1 |
| `min_samples_leaf` | 50 |
| `subsample` | 0.8 |

## Results

### Final model performance

| Dataset | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Train | 0.5702 | 0.5824 | 0.1970 | 0.2944 | 0.6511 |
| Validation | 0.4458 | 0.4375 | 0.1654 | 0.2400 | 0.4716 |
| Test | 0.5429 | 0.3818 | 0.1429 | 0.2079 | 0.5033 |

### Test confusion matrix

| | Predicted 0 | Predicted 1 |
|---|---:|---:|
| Actual 0 | 169 | 34 |
| Actual 1 | 126 | 21 |

The out-of-sample results show that the Gradient Boosting model did not provide strong predictive power on the 2023-2025 test period. Test ROC-AUC was close to random at 0.5033, and recall for outperforming ETFs was low.

## Top feature importances

| Rank | Feature | Importance |
|---:|---|---:|
| 1 | `etf_mom_12m` | 0.2641 |
| 2 | `ETF_XLK` | 0.2407 |
| 3 | `etf_mom_1m` | 0.1832 |
| 4 | `etf_vol_6m` | 0.0841 |
| 5 | `etf_drawdown_12m` | 0.0436 |
| 6 | `VIXCLS_change_1m` | 0.0360 |
| 7 | `DGS10_change_1m` | 0.0284 |
| 8 | `yield_spread_10y_2y` | 0.0223 |

## Repository structure

```text
sector-etf-outperformance/
├── README.md
├── requirements.txt
├── notebooks/
│   └── gradient_boosting_sector_etf.ipynb
├── src/
│   ├── build_panel.py
│   └── gradient_boosting_model.py
├── outputs/
│   ├── gradient_boosting_results.csv
│   └── gradient_boosting_feature_importance.csv
└── report/
    └── gradient_boosting_summary.md
```

## How to run

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Place the FRED CSV files in a local `data/` folder:

```text
data/
├── DGS10.csv
├── DGS2.csv
├── DFF.csv
└── VIXCLS.csv
```

3. Run the scripts:

```bash
python src/build_panel.py
python src/gradient_boosting_model.py
```

The scripts generate the cleaned panel dataset, train/validation/test splits, Gradient Boosting results, and feature-importance output.

## Tools used

- Python
- pandas
- NumPy
- scikit-learn
- yfinance
- matplotlib
- FRED macroeconomic data
