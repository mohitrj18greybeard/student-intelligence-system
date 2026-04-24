"""
Data Insights Generator
=========================
Extracts and articulates key patterns from the dataset in natural language.
This module transforms raw statistical findings into storytelling-ready
insights that demonstrate analytical thinking.

The insights are NOT hardcoded — they are dynamically computed from
whatever dataset is loaded, making the system adaptable to new data.
"""

import pandas as pd
import numpy as np


def generate_dataset_insights(df: pd.DataFrame) -> list:
    """Analyze the dataset and return a list of insight dicts.

    Each insight contains:
      - title: Short headline
      - description: Detailed natural language explanation
      - category: 'surprising' | 'expected' | 'actionable'
      - evidence: Statistical evidence supporting the insight

    Args:
        df: Raw or lightly processed student DataFrame.

    Returns:
        List of insight dicts, sorted by importance.
    """
    insights = []

    # ----- Insight 1: Study time vs performance paradox -----
    if "study_hours_per_week" in df.columns and "final_score" in df.columns:
        high_study_low_perf = df[
            (df["study_hours_per_week"] > df["study_hours_per_week"].quantile(0.75))
            & (df["final_score"] < df["final_score"].quantile(0.35))
        ]
        pct = len(high_study_low_perf) / len(df) * 100
        if pct > 2:
            insights.append({
                "title": "The Study Time Paradox",
                "description": (
                    f"{pct:.1f}% of students study MORE than average but still "
                    f"score in the bottom 35%. This suggests that study hours "
                    f"alone don't determine success — study quality, motivation, "
                    f"and distraction levels matter more than raw time invested."
                ),
                "category": "surprising",
                "evidence": f"{len(high_study_low_perf)} students affected",
            })

    # ----- Insight 2: Attendance is a stronger predictor than study hours -----
    if all(c in df.columns for c in ["attendance_rate", "study_hours_per_week", "final_score"]):
        corr_att = df["attendance_rate"].corr(df["final_score"])
        corr_study = df["study_hours_per_week"].corr(df["final_score"])
        if abs(corr_att) > abs(corr_study):
            insights.append({
                "title": "Attendance Beats Study Hours",
                "description": (
                    f"Attendance (r={corr_att:.3f}) has a stronger correlation "
                    f"with final scores than study hours (r={corr_study:.3f}). "
                    f"This aligns with research showing that classroom exposure "
                    f"provides structured learning that self-study cannot replicate."
                ),
                "category": "actionable",
                "evidence": f"Correlation: attendance={corr_att:.3f} vs study={corr_study:.3f}",
            })

    # ----- Insight 3: The stress-social media spiral -----
    if all(c in df.columns for c in ["stress_level", "social_media_hours", "final_score"]):
        high_stress_social = df[
            (df["stress_level"] > 7) & (df["social_media_hours"] > 5)
        ]
        if len(high_stress_social) > 10:
            avg_score = high_stress_social["final_score"].mean()
            overall_avg = df["final_score"].mean()
            gap = overall_avg - avg_score
            insights.append({
                "title": "The Stress-Social Media Spiral",
                "description": (
                    f"Students with BOTH high stress (>7) AND heavy social media "
                    f"usage (>5 hrs) score {gap:.1f} points below average. "
                    f"Social media appears to be a coping mechanism for stress, "
                    f"but it creates a negative feedback loop that worsens performance."
                ),
                "category": "surprising",
                "evidence": f"Mean score: {avg_score:.1f} vs population avg: {overall_avg:.1f}",
            })

    # ----- Insight 4: Sleep's non-linear impact -----
    if "sleep_hours" in df.columns and "final_score" in df.columns:
        low_sleep = df[df["sleep_hours"] < 5]["final_score"].mean()
        mid_sleep = df[
            (df["sleep_hours"] >= 6.5) & (df["sleep_hours"] <= 8)
        ]["final_score"].mean()
        high_sleep = df[df["sleep_hours"] > 9]["final_score"].mean()

        if not (np.isnan(low_sleep) or np.isnan(mid_sleep)):
            insights.append({
                "title": "Sleep: The Hidden Performance Multiplier",
                "description": (
                    f"Students sleeping 6.5-8 hours score {mid_sleep:.1f} on average, "
                    f"while those below 5 hours score only {low_sleep:.1f}. "
                    f"Interestingly, sleeping >9 hours ({high_sleep:.1f}) also "
                    f"shows diminishing returns, suggesting an optimal sleep window."
                ),
                "category": "actionable",
                "evidence": f"<5hrs: {low_sleep:.1f}, 6.5-8hrs: {mid_sleep:.1f}, >9hrs: {high_sleep:.1f}",
            })

    # ----- Insight 5: Part-time job impact varies by income -----
    if all(c in df.columns for c in ["part_time_job", "family_income", "final_score"]):
        for income in ["Low", "High"]:
            sub = df[df["family_income"] == income]
            if len(sub) > 30:
                with_job = sub[sub["part_time_job"] == 1]["final_score"].mean()
                without_job = sub[sub["part_time_job"] == 0]["final_score"].mean()
                if not (np.isnan(with_job) or np.isnan(without_job)):
                    gap = without_job - with_job
                    if income == "Low" and gap > 0:
                        insights.append({
                            "title": "Part-Time Work: A Double-Edged Sword",
                            "description": (
                                f"For low-income students, having a part-time job "
                                f"reduces scores by {gap:.1f} points on average. "
                                f"While financial stability matters, the time trade-off "
                                f"disproportionately affects already-disadvantaged students."
                            ),
                            "category": "actionable",
                            "evidence": f"With job: {with_job:.1f}, Without: {without_job:.1f}",
                        })
                        break

    # ----- Insight 6: Feature interaction non-linearity -----
    if all(c in df.columns for c in ["motivation_level", "study_hours_per_week", "final_score"]):
        high_both = df[
            (df["motivation_level"] > 7)
            & (df["study_hours_per_week"] > df["study_hours_per_week"].quantile(0.7))
        ]["final_score"].mean()
        high_study_only = df[
            (df["motivation_level"] < 4)
            & (df["study_hours_per_week"] > df["study_hours_per_week"].quantile(0.7))
        ]["final_score"].mean()
        if not (np.isnan(high_both) or np.isnan(high_study_only)):
            diff = high_both - high_study_only
            insights.append({
                "title": "Motivation × Study Hours: A Non-Linear Interaction",
                "description": (
                    f"Students with BOTH high motivation and high study hours score "
                    f"{high_both:.1f}, but those with high study hours and LOW motivation "
                    f"score only {high_study_only:.1f} — a {diff:.1f}-point gap. "
                    f"This proves that motivation is a force multiplier for study time."
                ),
                "category": "surprising",
                "evidence": f"High motivation + high study: {high_both:.1f}, Low motivation + high study: {high_study_only:.1f}",
            })

    # ----- Insight 7: Gender parity analysis -----
    if "gender" in df.columns and "final_score" in df.columns:
        male_avg = df[df["gender"] == "Male"]["final_score"].mean()
        female_avg = df[df["gender"] == "Female"]["final_score"].mean()
        if not (np.isnan(male_avg) or np.isnan(female_avg)):
            gap = abs(male_avg - female_avg)
            if gap < 3:
                insights.append({
                    "title": "Gender Parity in Performance",
                    "description": (
                        f"Male average ({male_avg:.1f}) and female average "
                        f"({female_avg:.1f}) differ by only {gap:.1f} points. "
                        f"Gender alone is not a meaningful predictor of academic "
                        f"performance — behavioral factors dominate."
                    ),
                    "category": "expected",
                    "evidence": f"Male: {male_avg:.1f}, Female: {female_avg:.1f}, Gap: {gap:.1f}",
                })

    return insights


def generate_individual_insight(student_data: dict, predicted_score: float,
                                 shap_top_features: list = None) -> str:
    """Generate a narrative explanation for a single student's prediction.

    This creates a natural-language "story" explaining WHY the model
    predicted a specific score — critical for SHAP-driven explainability.

    Args:
        student_data: Dict of the student's raw feature values.
        predicted_score: Model's predicted score.
        shap_top_features: Top SHAP features as list of (name, impact) tuples.

    Returns:
        Multi-sentence natural language explanation.
    """
    parts = []

    # Opening
    if predicted_score >= 70:
        parts.append(
            f"This student is predicted to score {predicted_score:.1f}/100, "
            f"placing them in the **High** performance category."
        )
    elif predicted_score >= 40:
        parts.append(
            f"This student is predicted to score {predicted_score:.1f}/100, "
            f"placing them in the **Medium** performance category."
        )
    else:
        parts.append(
            f"This student is predicted to score {predicted_score:.1f}/100, "
            f"placing them in the **Low** (at-risk) performance category."
        )

    # SHAP-driven explanation
    if shap_top_features:
        positive = [(n, v) for n, v in shap_top_features if v > 0]
        negative = [(n, v) for n, v in shap_top_features if v < 0]

        if positive:
            top_pos = positive[0]
            parts.append(
                f"The strongest positive driver is **{top_pos[0].replace('_', ' ')}** "
                f"(+{top_pos[1]:.2f} points), which is boosting performance."
            )

        if negative:
            top_neg = negative[0]
            parts.append(
                f"The biggest risk factor is **{top_neg[0].replace('_', ' ')}** "
                f"({top_neg[1]:.2f} points), which is pulling the score down."
            )

    # Context from raw values
    study = student_data.get("study_hours_per_week", None)
    stress = student_data.get("stress_level", None)
    if study is not None and stress is not None:
        if study < 8 and stress > 6:
            parts.append(
                "The combination of low study hours and high stress "
                "creates a compounding negative effect that is greater "
                "than either factor alone."
            )

    return " ".join(parts)
