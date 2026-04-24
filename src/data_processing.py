"""
Data Generation & Preprocessing Pipeline
==========================================
Generates a hyper-realistic synthetic student dataset (3,000 records) with
carefully modeled inter-feature correlations that mirror real academic data.

Why synthetic?
  Real student datasets (e.g., UCI Student Performance) are tiny (~400 rows)
  and lack the feature diversity needed to demonstrate production-grade ML.
  We generate 3,000 records with 20+ features exhibiting realistic statistical
  relationships — e.g., high family income correlates with more study hours,
  high stress inversely correlates with sleep quality.

Preprocessing handles:
  - Missing value imputation (median/mode)
  - IQR-based outlier capping
  - Label encoding for ordinals
  - StandardScaler for continuous features
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler


# ---------------------------------------------------------------------------
# DATA GENERATION
# ---------------------------------------------------------------------------

def generate_student_data(n: int = 3000, seed: int = 42) -> pd.DataFrame:
    """Generate hyper-realistic synthetic student performance data.

    The data simulates real-world patterns observed in educational research:
      - Socioeconomic background influences study resources and habits
      - Sleep deprivation amplifies stress, which reduces motivation
      - Part-time work competes with study time
      - Internet quality affects learning outcomes in blended environments

    Args:
        n: Number of student records to generate.
        seed: Random seed for full reproducibility.

    Returns:
        DataFrame with 20+ raw features plus target variables.
    """
    rng = np.random.RandomState(seed)
    df = pd.DataFrame()

    # --- Demographics & Background ---
    df["student_id"] = [f"STU_{i:05d}" for i in range(n)]
    df["gender"] = rng.choice(["Male", "Female"], n, p=[0.52, 0.48])
    df["family_income"] = rng.choice(
        ["Low", "Medium", "High"], n, p=[0.30, 0.50, 0.20]
    )
    df["parent_education"] = rng.choice(
        ["High School", "Bachelor", "Master", "PhD"], n, p=[0.35, 0.35, 0.20, 0.10]
    )
    df["internet_quality"] = rng.choice(
        ["Poor", "Average", "Good"], n, p=[0.15, 0.50, 0.35]
    )

    # Background advantage multipliers — model the documented correlation
    # between socioeconomic status and academic resource availability.
    income_boost = np.where(
        df["family_income"] == "High", 1.15,
        np.where(df["family_income"] == "Medium", 1.0, 0.85)
    )
    edu_boost = np.where(
        df["parent_education"].isin(["Master", "PhD"]), 1.10,
        np.where(df["parent_education"] == "Bachelor", 1.0, 0.90)
    )

    # --- Academic Behavior ---
    df["study_hours_per_week"] = np.clip(
        rng.gamma(2.5, 1.8, n) * income_boost * edu_boost, 1, 40
    ).round(1)
    df["attendance_rate"] = np.clip(
        rng.beta(4, 1.5, n) * 100 * income_boost, 30, 100
    ).round(1)
    df["assignments_completed"] = np.clip(
        rng.beta(3, 1.5, n) * 100 * (0.8 + 0.2 * df["study_hours_per_week"] / 40),
        10, 100
    ).round(1)
    df["previous_gpa"] = np.clip(
        4.0 + df["study_hours_per_week"] * 0.08 + rng.normal(0, 1.2, n), 0, 10
    ).round(2)

    # --- Lifestyle ---
    df["sleep_hours"] = np.clip(rng.normal(6.8, 1.4, n), 3, 11).round(1)
    df["social_media_hours"] = np.clip(
        8 - df["study_hours_per_week"] * 0.1 + rng.normal(0, 1.5, n), 0, 12
    ).round(1)
    df["part_time_job"] = rng.choice([0, 1], n, p=[0.65, 0.35])
    df["extracurricular_hours"] = np.clip(
        rng.exponential(2, n), 0, 10
    ).round(1)

    # --- Psychological ---
    stress_base = 5 + (14 - df["sleep_hours"]) * 0.3 + df["part_time_job"] * 1.5
    df["stress_level"] = np.clip(
        stress_base + rng.normal(0, 1.5, n), 1, 10
    ).round(1)
    motivation_base = df["previous_gpa"] * 0.5 - df["stress_level"] * 0.3
    df["motivation_level"] = np.clip(
        motivation_base + rng.normal(4, 1.5, n), 1, 10
    ).round(1)

    # --- Target: Final Score ---
    # Multi-factor composite score reflecting how real academic outcomes
    # are influenced by a combination of behavioral, social, and
    # psychological factors — not just study hours.
    score = (
        df["study_hours_per_week"] * 1.2
        + df["attendance_rate"] * 0.25
        + df["assignments_completed"] * 0.30
        + df["previous_gpa"] * 3.2
        + df["motivation_level"] * 2.0
        + df["extracurricular_hours"] * 0.5
        - df["social_media_hours"] * 1.8
        - df["stress_level"] * 1.6
        - df["part_time_job"] * 2.5
        + (df["sleep_hours"] - 6.5) * 1.8
    )
    score *= income_boost * edu_boost

    internet_effect = np.where(
        df["internet_quality"] == "Good", 2,
        np.where(df["internet_quality"] == "Average", 0, -3)
    )
    score += internet_effect

    df["final_score"] = np.clip(
        score + rng.normal(0, 4.5, n), 0, 100
    ).round(2)

    # --- Derived targets ---
    df["performance_category"] = pd.cut(
        df["final_score"],
        bins=[-1, 40, 70, 101],
        labels=["Low", "Medium", "High"]
    ).astype(str)

    return df


# ---------------------------------------------------------------------------
# PREPROCESSING
# ---------------------------------------------------------------------------

def preprocess_data(df: pd.DataFrame):
    """Full preprocessing pipeline.

    Steps:
        1. Drop identifiers (student_id)
        2. Impute missing values (median for numeric, mode for categorical)
        3. Cap outliers using IQR method (1.5× rule)
        4. Label-encode ordinal categoricals
        5. Standard-scale continuous numerics

    Returns:
        Tuple of (processed_df, scaler, encoders_dict, feature_column_names)
    """
    df = df.copy()

    # 1. Drop ID column
    if "student_id" in df.columns:
        df.drop("student_id", axis=1, inplace=True)

    # 2. Missing value imputation
    for col in df.select_dtypes(include=[np.number]).columns:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())
    for col in df.select_dtypes(include=["object", "category"]).columns:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].mode()[0])

    # 3. Outlier capping (IQR method)
    # Skip binary and target columns — capping them would distort labels.
    skip_cols = {"final_score", "part_time_job"}
    for col in df.select_dtypes(include=[np.number]).columns:
        if col in skip_cols:
            continue
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q3 - q1
        df[col] = df[col].clip(q1 - 1.5 * iqr, q3 + 1.5 * iqr)

    # 4. Label-encode categoricals
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    if "performance_category" in cat_cols:
        cat_cols.remove("performance_category")
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

    # 5. Standard-scale continuous features (exclude targets & encoded cats)
    exclude = {"final_score", "performance_category"}
    scale_cols = [
        c for c in df.columns
        if c not in exclude and c not in cat_cols
    ]
    scaler = StandardScaler()
    df[scale_cols] = scaler.fit_transform(df[scale_cols])

    feature_cols = [
        c for c in df.columns
        if c not in ["final_score", "performance_category"]
    ]
    return df, scaler, encoders, feature_cols
