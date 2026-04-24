"""Feature engineering: 14 derived features."""
import pandas as pd
import numpy as np
from src.utils.logger import get_logger

logger = get_logger("data.features")

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["study_efficiency"] = df["study_hours"] * df["motivation_level"] / 10
    df["stress_impact"] = df["stress_level"] * df["social_media_hours"] / 10
    df["academic_consistency"] = (df["attendance"] + df["assignments_completed"]) / 2
    df["lifestyle_balance"] = df["sleep_hours"] - df["social_media_hours"]
    df["work_life_balance"] = df["study_hours"] + df["sleep_hours"] - df["part_time_job"] * 3
    df["engagement_score"] = (df["attendance"] * 0.3 + df["assignments_completed"] * 0.3
                              + df["motivation_level"] * 10 * 0.2 + df["study_hours"] * 5 * 0.2)
    df["risk_score"] = np.clip((10 - df["motivation_level"]) * 2 + df["stress_level"] * 2
                               + (12 - df["attendance"] / 10) * 1.5
                               + df["social_media_hours"] * 1.5 - df["study_hours"] * 2, 0, 100)
    df["wellbeing_index"] = np.clip(df["sleep_hours"] * 10 + (10 - df["stress_level"]) * 8
                                    + df["motivation_level"] * 5 - df["social_media_hours"] * 3, 0, 100)
    df["study_distraction_ratio"] = df["study_hours"] / (df["social_media_hours"] + 0.5)
    df["academic_effort_ratio"] = ((df["study_hours"] + df["assignments_completed"] / 10)
                                   / (df["social_media_hours"] + df["stress_level"] / 2 + 0.5))
    df["study_hours_sq"] = df["study_hours"] ** 2
    df["stress_motivation_gap"] = df["motivation_level"] - df["stress_level"]
    logger.info(f"Created 14 engineered features -> total columns: {len(df.columns)}")
    return df

def get_feature_columns(df: pd.DataFrame):
    categorical_cols = ["internet_quality", "family_income", "parent_education"]
    exclude = ["student_id", "final_score", "risk_category", "performance_category",
               "math_score", "science_score", "english_score", "programming_score"]
    all_cols = [c for c in df.columns if c not in exclude]
    numerical_cols = [c for c in all_cols if c not in categorical_cols]
    return numerical_cols, categorical_cols
