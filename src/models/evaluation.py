"""Model evaluation metrics."""
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import cross_val_score
from src.utils.logger import get_logger

logger = get_logger("models.evaluation")

def evaluate_regression(model, X_test, y_test, name, X_train=None, y_train=None):
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    mae = float(mean_absolute_error(y_test, y_pred))
    cv_rmse_mean, cv_rmse_std = 0.0, 0.0
    if X_train is not None and y_train is not None:
        try:
            cv = cross_val_score(model, X_train, y_train, cv=5, scoring="neg_root_mean_squared_error", n_jobs=-1)
            cv_rmse_mean, cv_rmse_std = float(-cv.mean()), float(cv.std())
        except Exception:
            pass
    logger.info(f"  {name}: R2={r2:.4f} RMSE={rmse:.4f} MAE={mae:.4f} CV_RMSE={cv_rmse_mean:.4f}")
    return {"r2": float(r2), "rmse": rmse, "mae": mae, "cv_rmse_mean": cv_rmse_mean, "cv_rmse_std": cv_rmse_std}

def evaluate_classification(model, X_test, y_test, name, class_names=None):
    y_pred = model.predict(X_test)
    acc = float(accuracy_score(y_test, y_pred))
    f1 = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))
    prec = float(precision_score(y_test, y_pred, average="weighted", zero_division=0))
    rec = float(recall_score(y_test, y_pred, average="weighted", zero_division=0))
    logger.info(f"  {name}: Acc={acc:.4f} F1={f1:.4f}")
    return {"accuracy": acc, "f1_weighted": f1, "precision": prec, "recall": rec}

def get_feature_importance(model, feature_names):
    if hasattr(model, "feature_importances_"):
        imp = model.feature_importances_
        names = feature_names if feature_names and len(feature_names) == len(imp) else [f"f{i}" for i in range(len(imp))]
        return pd.DataFrame({"feature": names, "importance": imp}).sort_values("importance", ascending=False).reset_index(drop=True)
    return pd.DataFrame()
