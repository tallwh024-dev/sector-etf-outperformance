"""
Gradient Boosting model for sector ETF outperformance prediction.

This script trains Jackson Wang's Gradient Boosting Classifier using the
chronological train/validation/test splits created by src/build_panel.py.
"""

from pathlib import Path

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)

OUTPUT_DIR = Path("outputs")


def load_splits():
    """Load scaled features and target files."""
    X_train = pd.read_csv(OUTPUT_DIR / "X_train_scaled.csv")
    X_valid = pd.read_csv(OUTPUT_DIR / "X_valid_scaled.csv")
    X_test = pd.read_csv(OUTPUT_DIR / "X_test_scaled.csv")

    y_train = pd.read_csv(OUTPUT_DIR / "y_train.csv").squeeze("columns").astype(int)
    y_valid = pd.read_csv(OUTPUT_DIR / "y_valid.csv").squeeze("columns").astype(int)
    y_test = pd.read_csv(OUTPUT_DIR / "y_test.csv").squeeze("columns").astype(int)

    return X_train, X_valid, X_test, y_train, y_valid, y_test


def evaluate_model(model, X, y, dataset_name):
    """Evaluate a fitted classifier and return a metric dictionary."""
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]

    result = {
        "dataset": dataset_name,
        "accuracy": accuracy_score(y, y_pred),
        "precision": precision_score(y, y_pred, zero_division=0),
        "recall": recall_score(y, y_pred, zero_division=0),
        "f1": f1_score(y, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y, y_proba),
        "pred_0": int((y_pred == 0).sum()),
        "pred_1": int((y_pred == 1).sum()),
    }

    print(f"\n{dataset_name} results")
    print("-" * 60)
    for key in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
        print(f"{key}: {result[key]:.4f}")
    print("\nClassification report:")
    print(classification_report(y, y_pred, zero_division=0))
    print("Confusion matrix:")
    print(confusion_matrix(y, y_pred))

    return result


def tune_gradient_boosting(X_train, y_train, X_valid, y_valid):
    """Tune Gradient Boosting hyperparameters using validation ROC-AUC."""
    param_grid = [
        {"n_estimators": 25, "learning_rate": 0.01, "max_depth": 1, "min_samples_leaf": 20, "subsample": 0.8},
        {"n_estimators": 50, "learning_rate": 0.01, "max_depth": 1, "min_samples_leaf": 20, "subsample": 0.8},
        {"n_estimators": 100, "learning_rate": 0.01, "max_depth": 1, "min_samples_leaf": 20, "subsample": 0.8},
        {"n_estimators": 25, "learning_rate": 0.02, "max_depth": 1, "min_samples_leaf": 30, "subsample": 0.8},
        {"n_estimators": 50, "learning_rate": 0.02, "max_depth": 1, "min_samples_leaf": 30, "subsample": 0.8},
        {"n_estimators": 100, "learning_rate": 0.02, "max_depth": 1, "min_samples_leaf": 30, "subsample": 0.8},
        {"n_estimators": 25, "learning_rate": 0.01, "max_depth": 2, "min_samples_leaf": 30, "subsample": 0.8},
        {"n_estimators": 50, "learning_rate": 0.01, "max_depth": 2, "min_samples_leaf": 30, "subsample": 0.8},
        {"n_estimators": 100, "learning_rate": 0.01, "max_depth": 2, "min_samples_leaf": 30, "subsample": 0.8},
        {"n_estimators": 50, "learning_rate": 0.03, "max_depth": 1, "min_samples_leaf": 50, "subsample": 0.8},
        {"n_estimators": 100, "learning_rate": 0.03, "max_depth": 1, "min_samples_leaf": 50, "subsample": 0.8},
    ]

    rows = []
    for params in param_grid:
        model = GradientBoostingClassifier(**params, random_state=42)
        model.fit(X_train, y_train)

        y_train_pred = model.predict(X_train)
        y_train_proba = model.predict_proba(X_train)[:, 1]
        y_valid_pred = model.predict(X_valid)
        y_valid_proba = model.predict_proba(X_valid)[:, 1]

        rows.append(
            {
                **params,
                "train_accuracy": accuracy_score(y_train, y_train_pred),
                "train_roc_auc": roc_auc_score(y_train, y_train_proba),
                "valid_accuracy": accuracy_score(y_valid, y_valid_pred),
                "valid_roc_auc": roc_auc_score(y_valid, y_valid_proba),
                "valid_precision": precision_score(y_valid, y_valid_pred, zero_division=0),
                "valid_recall": recall_score(y_valid, y_valid_pred, zero_division=0),
                "valid_f1": f1_score(y_valid, y_valid_pred, zero_division=0),
                "valid_pred_0": int((y_valid_pred == 0).sum()),
                "valid_pred_1": int((y_valid_pred == 1).sum()),
            }
        )

    tuning_results = pd.DataFrame(rows).sort_values("valid_roc_auc", ascending=False)
    tuning_results.to_csv(OUTPUT_DIR / "gradient_boosting_tuning_results.csv", index=False)

    best = tuning_results.iloc[0]
    return {
        "n_estimators": int(best["n_estimators"]),
        "learning_rate": float(best["learning_rate"]),
        "max_depth": int(best["max_depth"]),
        "min_samples_leaf": int(best["min_samples_leaf"]),
        "subsample": float(best["subsample"]),
    }


def main():
    X_train, X_valid, X_test, y_train, y_valid, y_test = load_splits()

    best_params = tune_gradient_boosting(X_train, y_train, X_valid, y_valid)
    print("Best Gradient Boosting parameters:", best_params)

    model = GradientBoostingClassifier(**best_params, random_state=42)
    model.fit(X_train, y_train)

    results = pd.DataFrame(
        [
            evaluate_model(model, X_train, y_train, "Train"),
            evaluate_model(model, X_valid, y_valid, "Validation"),
            evaluate_model(model, X_test, y_test, "Test"),
        ]
    )
    results.to_csv(OUTPUT_DIR / "gradient_boosting_results.csv", index=False)

    feature_importance = pd.DataFrame(
        {"feature": X_train.columns, "importance": model.feature_importances_}
    ).sort_values("importance", ascending=False)
    feature_importance.to_csv(OUTPUT_DIR / "gradient_boosting_feature_importance.csv", index=False)

    print("\nTop feature importances:")
    print(feature_importance.head(15))


if __name__ == "__main__":
    main()
