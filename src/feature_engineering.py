"""
Feature Engineering Module
============================
Creates 14 domain-driven derived features that capture latent patterns
in student behavior invisible in raw data alone.

Each feature is grounded in educational psychology or learning science
research. The goal is NOT to throw in random transformations, but to
encode domain knowledge that helps the model generalize better.

Feature Design Philosophy:
  - Ratio features capture *efficiency*, not just magnitude
  - Interaction features model the compound effect of risk factors
  - Non-linear features handle diminishing returns (e.g., study hours)
  - Composite indices aggregate multiple weak signals into one strong one
"""

import numpy as np
import pandas as pd


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer 14 domain-driven features from raw student data.

    Each feature targets a specific latent construct that raw features
    alone cannot capture. The justification for each is inline below.

    Args:
        df: Raw student DataFrame (before or after preprocessing).

    Returns:
        DataFrame with all original columns plus 14 new features.
    """
    df = df.copy()

    # ---- 1. Study Efficiency Score ----
    # Raw study hours ignore HOW students study. A motivated student
    # retains more per hour than an unmotivated one. Multiplying by
    # motivation captures this — it's the "quality-adjusted" study time.
    df["study_efficiency"] = (
        df["study_hours_per_week"] * df["motivation_level"]
    ) / 10

    # ---- 2. Study-to-Distraction Ratio ----
    # Absolute study hours are misleading if a student also spends 8 hrs
    # on social media. This ratio captures net productive time allocation.
    # The +0.5 prevents division by zero for rare zero-social-media students.
    df["study_distraction_ratio"] = (
        df["study_hours_per_week"] / (df["social_media_hours"] + 0.5)
    )

    # ---- 3. Attendance Impact Score ----
    # Attendance matters MORE when combined with assignment completion.
    # A student who attends but never submits work gains less than one
    # who both attends AND completes assignments. This interaction term
    # captures that synergy.
    df["attendance_impact"] = (
        df["attendance_rate"] * df["assignments_completed"]
    ) / 100

    # ---- 4. Consistency Index ----
    # Students who are consistent across multiple academic behaviors
    # (attendance, assignments, study) outperform those who spike in
    # one area. Low variance = high consistency = better outcomes.
    academic_cols = ["attendance_rate", "assignments_completed"]
    if all(c in df.columns for c in academic_cols):
        df["consistency_index"] = 100 - df[academic_cols].std(axis=1)

    # ---- 5. Engagement Level Score ----
    # Multi-signal engagement metric. Weights reflect educational research:
    # attendance (30%), assignments (30%), motivation (20%), study (20%).
    df["engagement_score"] = (
        df["attendance_rate"] * 0.30
        + df["assignments_completed"] * 0.30
        + df["motivation_level"] * 10 * 0.20
        + df["study_hours_per_week"] * 2.5 * 0.20
    )

    # ---- 6. Stress Impact Index ----
    # Stress alone is manageable. Stress + social media (avoidance coping)
    # is the real predictor of academic decline. This interaction captures
    # the "stress spiral" documented in educational psychology.
    df["stress_impact"] = (
        df["stress_level"] * df["social_media_hours"]
    ) / 10

    # ---- 7. Lifestyle Balance Score ----
    # Sleep vs social media reveals self-regulation ability.
    # Positive = healthy balance, Negative = poor self-regulation.
    df["lifestyle_balance"] = df["sleep_hours"] - df["social_media_hours"]

    # ---- 8. Work-Life Balance ----
    # Part-time work competes directly with study and sleep time.
    # This captures the total "productive + restorative" time budget
    # minus the hours lost to employment.
    df["work_life_balance"] = (
        df["study_hours_per_week"] + df["sleep_hours"]
        - df["part_time_job"] * 8
    )

    # ---- 9. Risk Score ----
    # Composite risk indicator combining all negative factors.
    # High risk = low motivation + high stress + high screen time + low study.
    df["risk_score"] = np.clip(
        (10 - df["motivation_level"]) * 2
        + df["stress_level"] * 2
        + df["social_media_hours"] * 1.5
        - df["study_hours_per_week"] * 0.5,
        0, 100
    )

    # ---- 10. Wellbeing Index ----
    # Cognitive performance depends on physical and mental health.
    # This index combines sleep quality, stress management, motivation,
    # and screen time into a single health metric.
    df["wellbeing_index"] = np.clip(
        df["sleep_hours"] * 10
        + (10 - df["stress_level"]) * 8
        + df["motivation_level"] * 5
        - df["social_media_hours"] * 3,
        0, 150
    )

    # ---- 11. Academic Effort Ratio ----
    # Effort vs resistance. Numerator = productive behaviors,
    # denominator = impediments. High ratio = student is winning
    # the battle against distractions.
    df["academic_effort_ratio"] = (
        (df["study_hours_per_week"] + df["assignments_completed"] / 10)
        / (df["social_media_hours"] + df["stress_level"] / 2 + 0.5)
    )

    # ---- 12. Study Hours Squared (Non-linear) ----
    # Diminishing returns: studying 20 hrs/week is NOT 2× better than 10.
    # The quadratic term lets the model learn the point of diminishing returns
    # without requiring manual threshold engineering.
    df["study_hours_sq"] = df["study_hours_per_week"] ** 2

    # ---- 13. Stress-Motivation Gap ----
    # The gap between motivation and stress is a resilience indicator.
    # Positive gap = student has internal drive exceeding external pressure.
    # Negative gap = at-risk for burnout or dropout.
    df["stress_motivation_gap"] = (
        df["motivation_level"] - df["stress_level"]
    )

    # ---- 14. Academic Consistency (mean-based) ----
    # Simple average of attendance and assignments as a discipline proxy.
    df["academic_consistency"] = (
        df["attendance_rate"] + df["assignments_completed"]
    ) / 2

    return df


def get_feature_columns(df: pd.DataFrame):
    """Return (numerical_cols, categorical_cols) excluding targets.

    Args:
        df: DataFrame after feature engineering.

    Returns:
        Tuple of (numerical_column_names, categorical_column_names).
    """
    exclude = {"student_id", "final_score", "performance_category"}
    cat_cols = ["gender", "family_income", "parent_education", "internet_quality"]

    all_features = [c for c in df.columns if c not in exclude]
    num_cols = [c for c in all_features if c not in cat_cols]
    actual_cats = [c for c in cat_cols if c in df.columns]

    return num_cols, actual_cats
