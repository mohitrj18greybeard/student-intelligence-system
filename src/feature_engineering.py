"""
STEP 4: Feature Engineering
=============================
Creates 12 meaningful derived features that capture hidden patterns
in student behavior. Each feature is justified with domain reasoning.
"""

import numpy as np
import pandas as pd


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer 12 domain-driven features.

    Each feature captures a latent factor not directly observable
    in raw data but strongly correlated with academic outcomes.
    """
    df = df.copy()

    # 1. Study Efficiency Score
    #    Why: Raw study hours don't account for motivation. A motivated
    #    student studying 5 hrs outperforms an unmotivated one studying 8.
    df["study_efficiency"] = (df["study_hours_per_week"] * df["motivation_level"]) / 10

    # 2. Stress Impact Index
    #    Why: Stress alone doesn't predict failure — stress + social media
    #    (avoidance behavior) is the real predictor.
    df["stress_impact"] = (df["stress_level"] * df["social_media_hours"]) / 10

    # 3. Academic Consistency
    #    Why: Students who attend AND complete assignments show discipline,
    #    a stronger predictor than either alone.
    df["academic_consistency"] = (df["attendance_rate"] + df["assignments_completed"]) / 2

    # 4. Lifestyle Balance Score
    #    Why: Sleep vs social media reflects self-regulation ability.
    df["lifestyle_balance"] = df["sleep_hours"] - df["social_media_hours"]

    # 5. Work-Life Balance
    #    Why: Part-time work reduces effective study+sleep time.
    df["work_life_balance"] = df["study_hours_per_week"] + df["sleep_hours"] - df["part_time_job"] * 8

    # 6. Engagement Score
    #    Why: Multi-signal engagement metric combining attendance,
    #    homework, motivation, and study time.
    df["engagement_score"] = (
        df["attendance_rate"] * 0.3
        + df["assignments_completed"] * 0.3
        + df["motivation_level"] * 10 * 0.2
        + df["study_hours_per_week"] * 2.5 * 0.2
    )

    # 7. Risk Score
    #    Why: Identifies at-risk students by combining negative factors.
    df["risk_score"] = np.clip(
        (10 - df["motivation_level"]) * 2
        + df["stress_level"] * 2
        + df["social_media_hours"] * 1.5
        - df["study_hours_per_week"] * 0.5, 0, 100
    )

    # 8. Wellbeing Index
    #    Why: Health and mental state affect cognitive performance.
    df["wellbeing_index"] = np.clip(
        df["sleep_hours"] * 10
        + (10 - df["stress_level"]) * 8
        + df["motivation_level"] * 5
        - df["social_media_hours"] * 3, 0, 150
    )

    # 9. Study-Distraction Ratio
    #    Why: Ratio captures time allocation efficiency better than
    #    absolute hours.
    df["study_distraction_ratio"] = df["study_hours_per_week"] / (df["social_media_hours"] + 0.5)

    # 10. Academic Effort Ratio
    #     Why: Effort vs resistance — high effort + low resistance = success.
    df["academic_effort_ratio"] = (
        (df["study_hours_per_week"] + df["assignments_completed"] / 10)
        / (df["social_media_hours"] + df["stress_level"] / 2 + 0.5)
    )

    # 11. Study Hours Squared (Non-linear)
    #     Why: Diminishing returns — studying 20 hrs/week is not 2× better
    #     than 10. Quadratic captures this.
    df["study_hours_sq"] = df["study_hours_per_week"] ** 2

    # 12. Stress-Motivation Gap
    #     Why: Positive gap = resilient student. Negative = at-risk.
    df["stress_motivation_gap"] = df["motivation_level"] - df["stress_level"]

    return df


def get_feature_columns(df: pd.DataFrame):
    """Return (numerical_cols, categorical_cols) excluding targets."""
    exclude = ["student_id", "final_score", "performance_category"]
    cat_cols = ["gender", "family_income", "parent_education", "internet_quality"]
    # After encoding, categoricals become numeric, so just return all non-excluded
    all_features = [c for c in df.columns if c not in exclude]
    num_cols = [c for c in all_features if c not in cat_cols]
    actual_cats = [c for c in cat_cols if c in df.columns]
    return num_cols, actual_cats
