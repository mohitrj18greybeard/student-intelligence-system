"""
STEP 7 (extended): Evaluation Utilities
=========================================
Comparison tables, visualization helpers, and summary generators.
"""

import pandas as pd
import numpy as np


def comparison_table(results: dict) -> pd.DataFrame:
    """Build a model comparison DataFrame from results dict."""
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
    return pd.DataFrame(rows).sort_values("R² Score", ascending=False).reset_index(drop=True)


def classify_performance(score: float) -> str:
    """Classify predicted score into performance category."""
    if score >= 70:
        return "High"
    elif score >= 40:
        return "Medium"
    else:
        return "Low"


def performance_summary(score: float, category: str) -> str:
    """Generate a human-readable performance summary."""
    if category == "High":
        return (f"Predicted Score: {score:.1f}/100 — Excellent Performance! "
                f"You are on track for top academic results.")
    elif category == "Medium":
        return (f"Predicted Score: {score:.1f}/100 — Average Performance. "
                f"There is clear room for improvement with targeted changes.")
    else:
        return (f"Predicted Score: {score:.1f}/100 — Below Average. "
                f"Immediate attention needed. See recommendations below.")
