"""
STEP 6 & 7: Model Training & Evaluation
=========================================
- Trains Linear Regression, Random Forest, and XGBoost
- Includes cross-validation and hyperparameter tuning (RandomizedSearchCV)
- Compares models on R², RMSE, MAE
- Selects and returns the best model
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


def train_all_models(X: np.ndarray, y: np.ndarray, feature_names: list = None):
    """Train 3 models with hyperparameter tuning and full evaluation.

    Args:
        X: Feature matrix.
        y: Target vector.
        feature_names: List of feature names.

    Returns:
        results: Dict with model names → metrics.
        best_model: The best performing model object.
        best_name: Name of the best model.
        X_train, X_test, y_train, y_test: Data splits.
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
    results["Linear Regression"] = _evaluate(lr, X_test, y_test, X_train, y_train)

    # --- 2. Random Forest (with tuning) ---
    rf_params = {
        "n_estimators": [100, 200, 300],
        "max_depth": [10, 15, 20, None],
        "min_samples_split": [2, 5, 10],
    }
    rf_search = RandomizedSearchCV(
        RandomForestRegressor(random_state=42, n_jobs=-1),
        rf_params, n_iter=10, cv=3, scoring="r2", random_state=42, n_jobs=-1
    )
    rf_search.fit(X_train, y_train)
    models["Random Forest"] = rf_search.best_estimator_
    results["Random Forest"] = _evaluate(rf_search.best_estimator_, X_test, y_test, X_train, y_train)

    # --- 3. XGBoost (with tuning) ---
    xgb_params = {
        "n_estimators": [200, 300, 500],
        "max_depth": [4, 6, 8],
        "learning_rate": [0.05, 0.1, 0.15],
        "subsample": [0.8, 0.9, 1.0],
    }
    xgb_search = RandomizedSearchCV(
        XGBRegressor(random_state=42, n_jobs=-1, verbosity=0),
        xgb_params, n_iter=10, cv=3, scoring="r2", random_state=42, n_jobs=-1
    )
    xgb_search.fit(X_train, y_train)
    models["XGBoost"] = xgb_search.best_estimator_
    results["XGBoost"] = _evaluate(xgb_search.best_estimator_, X_test, y_test, X_train, y_train)

    # Select best
    best_name = max(results, key=lambda k: results[k]["r2"])
    best_model = models[best_name]

    return results, best_model, best_name, models, X_train, X_test, y_train, y_test


def _evaluate(model, X_test, y_test, X_train, y_train):
    """Compute regression metrics + 5-fold CV."""
    y_pred = model.predict(X_test)
    r2 = float(r2_score(y_test, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    mae = float(mean_absolute_error(y_test, y_pred))

    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="r2", n_jobs=-1)

    return {
        "r2": round(r2, 4),
        "rmse": round(rmse, 4),
        "mae": round(mae, 4),
        "cv_r2_mean": round(float(cv_scores.mean()), 4),
        "cv_r2_std": round(float(cv_scores.std()), 4),
    }


def save_model(model, path: str = "models/best_model.joblib"):
    """STEP 10: Save model with joblib for reproducibility."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_model(path: str = "models/best_model.joblib"):
    """Load a saved model."""
    return joblib.load(path)


def get_feature_importance(model, feature_names: list) -> pd.DataFrame:
    """Extract feature importance from tree-based models."""
    if hasattr(model, "feature_importances_"):
        imp = model.feature_importances_
        names = feature_names[:len(imp)] if feature_names else [f"f{i}" for i in range(len(imp))]
        return pd.DataFrame({"feature": names, "importance": imp}).sort_values(
            "importance", ascending=False
        ).reset_index(drop=True)
    return pd.DataFrame()
