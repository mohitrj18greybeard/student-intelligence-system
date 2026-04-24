"""Hyper-realistic synthetic student data generator."""
import pandas as pd
import numpy as np
from src.utils.logger import get_logger

logger = get_logger("data.generator")

def generate_student_data(n: int = 3000, seed: int = 42) -> pd.DataFrame:
    logger.info(f"Generating {n} synthetic student records (seed={seed})...")
    np.random.seed(seed)
    data = pd.DataFrame()
    data["student_id"] = [f"STU_{i:05d}" for i in range(n)]

    data["family_income"] = np.random.choice(["low", "medium", "high"], n, p=[0.30, 0.50, 0.20])
    data["parent_education"] = np.random.choice(["school", "graduate", "postgraduate"], n, p=[0.35, 0.40, 0.25])
    data["internet_quality"] = np.random.choice(["poor", "average", "good"], n, p=[0.15, 0.50, 0.35])

    income_boost = np.where(data["family_income"] == "high", 1.15, np.where(data["family_income"] == "medium", 1.0, 0.85))
    edu_boost = np.where(data["parent_education"] == "postgraduate", 1.10, np.where(data["parent_education"] == "graduate", 1.0, 0.90))

    data["study_hours"] = np.clip(np.random.gamma(2.5, 1.8, n) * income_boost * edu_boost, 0.5, 14)
    data["attendance"] = np.clip(np.random.beta(4, 1.5, n) * 100 * income_boost, 30, 100)
    data["assignments_completed"] = np.clip(np.random.beta(3, 1.5, n) * 100 * (0.8 + 0.2 * (data["study_hours"] / 14)), 10, 100)
    data["previous_gpa"] = np.clip(4.0 + data["study_hours"] * 0.3 + np.random.normal(0, 1.2, n), 0, 10)
    data["sleep_hours"] = np.clip(np.random.normal(6.8, 1.4, n), 3, 11)
    data["social_media_hours"] = np.clip(8 - data["study_hours"] * 0.3 + np.random.normal(0, 1.5, n), 0, 12)
    data["part_time_job"] = np.random.choice([0, 1], n, p=[0.65, 0.35])
    data["extracurricular_hours"] = np.clip(np.random.exponential(2, n), 0, 10)

    stress_base = 5 + (14 - data["sleep_hours"]) * 0.3 + data["part_time_job"] * 1.5
    data["stress_level"] = np.clip(stress_base + np.random.normal(0, 1.5, n), 1, 10)
    motivation_base = data["previous_gpa"] * 0.5 - data["stress_level"] * 0.3
    data["motivation_level"] = np.clip(motivation_base + np.random.normal(4, 1.5, n), 1, 10)
    data["week"] = np.random.randint(1, 17, n)

    for subj in ["math", "science", "english", "programming"]:
        base = (data["study_hours"] * np.random.uniform(3, 6) + data["attendance"] * 0.15
                + data["previous_gpa"] * 2.5 + data["motivation_level"] * 1.5
                - data["stress_level"] * 1.2 - data["social_media_hours"] * 1.0
                + np.random.normal(0, 8, n))
        data[f"{subj}_score"] = np.clip(base, 0, 100).astype(float)

    score = (data["study_hours"] * 4.5 + data["attendance"] * 0.25
             + data["assignments_completed"] * 0.30 + data["previous_gpa"] * 3.2
             + data["motivation_level"] * 2.0 + data["extracurricular_hours"] * 0.5
             - data["social_media_hours"] * 1.8 - data["stress_level"] * 1.6
             - data["part_time_job"] * 2.5 + (data["sleep_hours"] - 6.5) * 1.8)
    score *= income_boost * edu_boost
    internet_effect = np.where(data["internet_quality"] == "good", 2.0, np.where(data["internet_quality"] == "average", 0, -3.0))
    score += internet_effect - np.abs(data["week"] - 8) * 0.3
    data["final_score"] = np.clip(score + np.random.normal(0, 4.5, n), 0, 100).astype(float)

    data["risk_category"] = pd.cut(data["final_score"], bins=[-1, 35, 50, 65, 80, 101],
                                    labels=["critical", "high", "moderate", "low", "excellent"]).astype(str)
    data["performance_category"] = pd.cut(data["final_score"], bins=[-1, 40, 70, 101],
                                           labels=["low", "medium", "high"]).astype(str)
    logger.info(f"Generated {len(data)} records, score mean={data['final_score'].mean():.1f}")
    return data
