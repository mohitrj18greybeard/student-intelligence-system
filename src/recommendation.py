"""
STEP 9: Personalized Recommendation Engine
=============================================
Generates targeted, actionable improvement suggestions based on:
  - Predicted score
  - SHAP feature impact analysis
  - Student input data thresholds
"""


def generate_recommendations(student_data: dict, predicted_score: float,
                              shap_factors: list = None) -> list:
    """Generate personalized recommendations.

    Args:
        student_data: Dict of student features.
        predicted_score: Model predicted score.
        shap_factors: Optional list of SHAP factors (feature, impact, direction).

    Returns:
        List of recommendation dicts sorted by priority.
    """
    recs = []

    # --- Study Time ---
    study = student_data.get("study_hours_per_week", 0)
    if study < 8:
        recs.append({
            "category": "📚 Study Time",
            "priority": "High" if study < 4 else "Medium",
            "message": f"Your study time ({study:.0f} hrs/week) is low. Aim for 15-20 hrs/week.",
            "expected_impact": "+8-15 points",
        })

    # --- Attendance ---
    att = student_data.get("attendance_rate", 0)
    if att < 65:
        recs.append({
            "category": "🏫 Attendance",
            "priority": "Critical",
            "message": f"Attendance ({att:.0f}%) is critically low. Every missed class costs ~0.5 pts.",
            "expected_impact": "+10-18 points",
        })
    elif att < 80:
        recs.append({
            "category": "🏫 Attendance",
            "priority": "Medium",
            "message": f"Attendance ({att:.0f}%) needs improvement. Target 85%+.",
            "expected_impact": "+5-8 points",
        })

    # --- Sleep ---
    sleep = student_data.get("sleep_hours", 0)
    if sleep < 6:
        recs.append({
            "category": "😴 Sleep",
            "priority": "High",
            "message": f"Only {sleep:.1f} hrs sleep. Sleep deprivation reduces memory consolidation by 40%.",
            "expected_impact": "+5-10 points",
        })

    # --- Social Media ---
    social = student_data.get("social_media_hours", 0)
    if social > 5:
        recs.append({
            "category": "📱 Screen Time",
            "priority": "High",
            "message": f"Social media ({social:.1f} hrs/day) is excessive. Set app time limits.",
            "expected_impact": "+6-12 points",
        })

    # --- Stress ---
    stress = student_data.get("stress_level", 0)
    if stress > 7:
        recs.append({
            "category": "🧘 Stress Management",
            "priority": "High",
            "message": f"High stress ({stress:.0f}/10). Practice mindfulness or seek counseling.",
            "expected_impact": "+4-8 points",
        })

    # --- Motivation ---
    motivation = student_data.get("motivation_level", 0)
    if motivation < 4:
        recs.append({
            "category": "🎯 Motivation",
            "priority": "High",
            "message": "Low motivation detected. Set small achievable goals and reward yourself.",
            "expected_impact": "+5-10 points",
        })

    # --- Assignments ---
    assignments = student_data.get("assignments_completed", 0)
    if assignments < 50:
        recs.append({
            "category": "📝 Assignments",
            "priority": "Critical",
            "message": f"Only {assignments:.0f}% assignments done. Each missing assignment costs ~1 pt.",
            "expected_impact": "+8-15 points",
        })

    # --- Positive reinforcement ---
    if predicted_score >= 80:
        recs.append({
            "category": "🌟 Excellence",
            "priority": "Low",
            "message": "Excellent performance! Consider peer tutoring to reinforce your knowledge.",
            "expected_impact": "Maintain A grade",
        })
    elif predicted_score >= 60 and len(recs) == 0:
        recs.append({
            "category": "✅ On Track",
            "priority": "Low",
            "message": "You're doing well. Small improvements in consistency will push you higher.",
            "expected_impact": "+3-5 points",
        })

    # --- SHAP-based (if available) ---
    if shap_factors:
        negative = [f for f in shap_factors if f["direction"] == "negative"]
        for f in negative[:2]:
            feat = f["feature"].replace("_", " ").title()
            recs.append({
                "category": f"📊 {feat}",
                "priority": "Medium",
                "message": f"'{feat}' is negatively impacting your score (SHAP impact: {f['impact']:.2f}).",
                "expected_impact": "+3-6 points",
            })

    # Sort by priority
    priority_order = {"Critical": 0, "High": 1, "Medium": 2, "Low": 3}
    recs.sort(key=lambda r: priority_order.get(r["priority"], 99))

    return recs
