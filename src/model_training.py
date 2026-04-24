"""
Model Training & Evaluation Pipeline
=======================================
Trains three regression models (Linear, Random Forest, XGBoost), performs
hyperparameter tuning via RandomizedSearchCV, and selects the best model
based on cross-validated R² score.

Model Selection Rationale:
  - Linear Regression: Baseline. Shows how much variance is linearly
    explainable. If LR scores high, features are well-engineered.
  - Random Forest: Captures non-linear relationships and feature
    interactions without explicit feature engineering. Robust to noise.
  - XGBoost: State-of-the-art gradient boosting. Handles missing values,
    regularization, and feature importance natively. Expected winner for
    structured tabular data (per "Why do tree-based models still outperform
    deep learning on tabular data?" — Grinsztajn et al., NeurIPS 2022).

Trade-offs:
  - LR is interpretable but underfits non-linear patterns
  - RF is robust but slower to inference at scale
  - XGBoost is best accuracy but requires careful hyperparameter tuning
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import (
    train_test_split, cross_val_score, RandomizedSearchCV
)
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


def train_all_models(X: np.ndarray, y: np.ndarray, feature_names: list = None):
    """Train 3 models with hyperparameter tuning and full evaluation.

    Pipeline:
        1. 80/20 stratified-ish split (random_state=42 for reproducibility)
        2. Train each model (with RandomizedSearchCV for RF & XGB)
        3. Evaluate on test set + 5-fold CV on training set
        4. Select best model by test R² score

    Args:
        X: Feature matrix (n_samples, n_features).
        y: Target vector (continuous scores 0-100).
        feature_names: Optional list of feature names for interpretability.

    Returns:
        results: Dict mapping model_name → metrics dict.
        best_model: The best performing sklearn/xgb model object.
        best_name: Name string of the best model.
        models: Dict of all trained model objects.
        X_train, X_test, y_train, y_test: The data splits used.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    models = {}
    results = {}

    # --- 1. Linear Regression (Baseline) ---
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    models["Linear Regression"] = lr
    results["Linear Regression"] = _evaluate(
        lr, X_test, y_test, X_train, y_train
    )

    # --- 2. Random Forest ---
    rf_params = {
        "n_estimators": [100, 200, 300],
        "max_depth": [10, 15, 20, None],
        "min_samples_split": [2, 5, 10],
    }
    rf_search = RandomizedSearchCV(
        RandomForestRegressor(random_state=42, n_jobs=-1),
        rf_params, n_iter=10, cv=3, scoring="r2",
        random_state=42, n_jobs=-1
    )
    rf_search.fit(X_train, y_train)
    models["Random Forest"] = rf_search.best_estimator_
    results["Random Forest"] = _evaluate(
        rf_search.best_estimator_, X_test, y_test, X_train, y_train
    )

    # --- 3. XGBoost ---
    xgb_params = {
        "n_estimators": [200, 300, 500],
        "max_depth": [4, 6, 8],
        "learning_rate": [0.05, 0.1, 0.15],
        "subsample": [0.8, 0.9, 1.0],
    }
    xgb_search = RandomizedSearchCV(
        XGBRegressor(random_state=42, n_jobs=-1, verbosity=0),
        xgb_params, n_iter=10, cv=3, scoring="r2",
        random_state=42, n_jobs=-1
    )
    xgb_search.fit(X_train, y_train)
    models["XGBoost"] = xgb_search.best_estimator_
    results["XGBoost"] = _evaluate(
        xgb_search.best_estimator_, X_test, y_test, X_train, y_train
    )

    # Select the best model by test-set R²
    best_name = max(results, key=lambda k: results[k]["r2"])
    best_model = models[best_name]

    return results, best_model, best_name, models, X_train, X_test, y_train, y_test


def _evaluate(model, X_test, y_test, X_train, y_train):
    """Compute regression metrics + 5-fold cross-validation.

    Returns dict with: r2, rmse, mae, cv_r2_mean, cv_r2_std.
    """
    y_pred = model.predict(X_test)
    r2 = float(r2_score(y_test, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    mae = float(mean_absolute_error(y_test, y_pred))

    # 5-fold CV on training data to check for overfitting
    cv_scores = cross_val_score(
        model, X_train, y_train, cv=5, scoring="r2", n_jobs=-1
    )

    return {
        "r2": round(r2, 4),
        "rmse": round(rmse, 4),
        "mae": round(mae, 4),
        "cv_r2_mean": round(float(cv_scores.mean()), 4),
        "cv_r2_std": round(float(cv_scores.std()), 4),
    }


def save_model(model, path: str = "models/best_model.joblib"):
    """Serialize trained model to disk using joblib.

    Joblib is preferred over pickle for sklearn models because it
    handles large numpy arrays more efficiently.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_model(path: str = "models/best_model.joblib"):
    """Deserialize a saved model from disk."""
    return joblib.load(path)


def get_feature_importance(model, feature_names: list) -> pd.DataFrame:
    """Extract feature importance from tree-based models.

    Uses the built-in `feature_importances_` attribute (Gini importance
    for RF, gain-based for XGBoost).

    Args:
        model: Trained model with feature_importances_ attribute.
        feature_names: List of feature column names.

    Returns:
        DataFrame with columns ['feature', 'importance'], sorted descending.
    """
    if hasattr(model, "feature_importances_"):
        imp = model.feature_importances_
        names = (
            feature_names[:len(imp)]
            if feature_names
            else [f"feature_{i}" for i in range(len(imp))]
        )
        return (
            pd.DataFrame({"feature": names, "importance": imp})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )
    return pd.DataFrame()
