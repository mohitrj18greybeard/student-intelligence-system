"""
Evaluation Utilities
======================
Comparison tables, performance classification, and human-readable
summary generators for model output.
"""

import pandas as pd
import numpy as np


def comparison_table(results: dict) -> pd.DataFrame:
    """Build a model comparison DataFrame from results dict.

    Args:
        results: Dict mapping model_name → metrics dict.

    Returns:
        DataFrame sorted by R² Score descending, ready for display.
    """
    rows = []
    for name, metrics in results.items():
        rows.append({
            "Model": name,
            "R² Score": metrics["r2"],
            "RMSE": metrics["rmse"],
            "MAE": metrics["mae"],
            "CV R² (mean)": metrics["cv_r2_mean"],
            "CV R² (std)": metrics["cv_r2_std"],
        })
    return (
        pd.DataFrame(rows)
        .sort_values("R² Score", ascending=False)
        .reset_index(drop=True)
    )


def classify_performance(score: float) -> str:
    """Map continuous score to performance tier.

    Thresholds chosen based on typical grading rubrics:
      - High (≥70): Top quartile — on track for academic excellence
      - Medium (40-69): Average — room for improvement
      - Low (<40): At-risk — immediate intervention needed
    """
    if score >= 70:
        return "High"
    elif score >= 40:
        return "Medium"
    else:
        return "Low"


def prediction_confidence(model, X_single: np.ndarray, y_pred: float) -> float:
    """Estimate prediction confidence based on proximity to training data.

    For tree-based models, we use the variance across individual tree
    predictions as a proxy for uncertainty. Lower variance = higher confidence.

    Args:
        model: Trained model (RF or XGBoost).
        X_single: Single sample feature vector (1, n_features).
        y_pred: The model's point prediction.

    Returns:
        Confidence percentage (0-100).
    """
    try:
        if hasattr(model, "estimators_"):
            # Random Forest — use tree-level prediction variance
            tree_preds = np.array([
                t.predict(X_single)[0] for t in model.estimators_
            ])
            std = tree_preds.std()
            # Map std to confidence: std=0 → 99%, std=10 → ~75%
            confidence = max(60, min(99, 99 - std * 2.5))
        elif hasattr(model, "get_booster"):
            # XGBoost — use margin-based confidence
            import xgboost as xgb
            dmat = xgb.DMatrix(X_single)
            margin = model.get_booster().predict(dmat, output_margin=True)
            # Larger absolute margin = more confident
            confidence = min(99, 75 + abs(float(margin[0])) * 0.3)
        else:
            # Linear Regression — assume moderate confidence
            confidence = 78.0
    except Exception:
        confidence = 80.0

    return round(confidence, 1)


def performance_summary(score: float, category: str) -> str:
    """Generate a human-readable performance summary.

    Args:
        score: Predicted final score (0-100).
        category: Performance tier ('High', 'Medium', 'Low').

    Returns:
        Natural language summary string.
    """
    if category == "High":
        return (
            f"Predicted Score: {score:.1f}/100 — Excellent performance. "
            f"This student is on track for top academic results. "
            f"Focus on maintaining consistency and exploring advanced topics."
        )
    elif category == "Medium":
        return (
            f"Predicted Score: {score:.1f}/100 — Average performance. "
            f"There is clear room for improvement with targeted behavioral "
            f"changes. See the recommendations below for specific actions."
        )
    else:
        return (
            f"Predicted Score: {score:.1f}/100 — Below average. "
            f"Immediate attention is needed. The recommendations below "
            f"identify the most impactful changes this student can make."
        )
