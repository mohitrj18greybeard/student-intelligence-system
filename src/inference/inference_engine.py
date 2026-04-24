"""Inference engine with SHAP explainability and recommendations."""
import json, joblib
import numpy as np, pandas as pd
from pathlib import Path
from typing import Dict, List, Any
from src.data.feature_engineering import create_features
from src.utils.logger import get_logger

logger = get_logger("inference.engine")

class InferenceEngine:
    def __init__(self, models_dir="models"):
        self.models_dir = Path(models_dir)
        self.model = self.preprocessor = self.clf_model = self.clf_preprocessor = None
        self.label_encoder = self.feature_names = self.explainer = None
        self._loaded = False

    def load(self):
        try:
            mp = self.models_dir / "best_regression_model.joblib"
            if not mp.exists():
                logger.warning(f"Model not found: {mp}"); return False
            self.model = joblib.load(mp)
            self.preprocessor = joblib.load(self.models_dir / "preprocessor.joblib")
            fn = self.models_dir / "feature_names.json"
            if fn.exists():
                with open(fn) as f: self.feature_names = json.load(f)
            cp = self.models_dir / "risk_classifier.joblib"
            if cp.exists():
                self.clf_model = joblib.load(cp)
                self.clf_preprocessor = joblib.load(self.models_dir / "clf_preprocessor.joblib")
                self.label_encoder = joblib.load(self.models_dir / "label_encoder.joblib")
            self._loaded = True
            logger.info("All models loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Load failed: {e}"); return False

    def predict_student(self, student_data: Dict[str, Any], use_shap=True) -> Dict:
        if not self._loaded:
            raise RuntimeError("Models not loaded. Run training pipeline first.")
        input_df = pd.DataFrame([student_data])
        input_df = create_features(input_df)
        X = self.preprocessor.transform(input_df)
        score = float(np.clip(self.model.predict(X)[0], 0, 100))
        risk = self._risk(score)
        if self.clf_model and self.clf_preprocessor:
            try:
                Xc = self.clf_preprocessor.transform(input_df)
                risk = self.label_encoder.inverse_transform([self.clf_model.predict(Xc)[0]])[0]
            except Exception:
                pass
        factors = self._shap_factors(X) if use_shap else []
        recs = self._recommendations(student_data, score)
        return {"predicted_score": round(score, 2), "risk_level": risk, "risk_category": risk,
                "grade": self._grade(score), "confidence": self._confidence(score),
                "top_factors": factors, "recommendations": recs}

    def what_if_simulation(self, student_data, changes):
        original = self.predict_student(student_data, use_shap=False)
        modified = {**student_data, **changes}
        simulated = self.predict_student(modified, use_shap=False)
        return {"original_score": original["predicted_score"], "simulated_score": simulated["predicted_score"],
                "score_delta": round(simulated["predicted_score"] - original["predicted_score"], 2),
                "original_risk": original["risk_level"], "simulated_risk": simulated["risk_level"]}

    def _risk(self, s):
        if s >= 80: return "excellent"
        if s >= 65: return "low"
        if s >= 50: return "moderate"
        if s >= 35: return "high"
        return "critical"

    def _grade(self, s):
        if s >= 93: return "A+"
        if s >= 85: return "A"
        if s >= 77: return "B+"
        if s >= 70: return "B"
        if s >= 60: return "C"
        if s >= 50: return "D"
        return "F"

    def _confidence(self, s):
        if 30 <= s <= 80: return "high"
        if 20 <= s <= 90: return "medium"
        return "low"

    def _shap_factors(self, X):
        try:
            import shap
            from sklearn.ensemble import StackingRegressor, StackingClassifier
            if self.explainer is None:
                m = self.model
                if isinstance(m, (StackingRegressor, StackingClassifier)):
                    m = m.estimators_[0]
                self.explainer = shap.Explainer(m, feature_names=self.feature_names)
            sv = self.explainer(X)
            vals = sv.values[0]
            names = self.feature_names or [f"f{i}" for i in range(len(vals))]
            pairs = sorted(zip(names, vals), key=lambda x: abs(x[1]), reverse=True)[:8]
            return [{"feature": f.replace("num__","").replace("cat__",""), "impact": round(float(v),3),
                     "direction": "positive" if v > 0 else "negative"} for f, v in pairs]
        except Exception as e:
            logger.warning(f"SHAP failed: {e}")
            return []

    def _recommendations(self, d, score):
        recs = []
        if d.get("study_hours", 0) < 3:
            recs.append({"category": "study", "priority": "high", "message": "Increase daily study time to 4-5 hours.", "expected_impact": "+8-12 points", "icon": "📚"})
        if d.get("attendance", 0) < 65:
            recs.append({"category": "attendance", "priority": "critical", "message": "Attendance critically low. Aim for 80%+.", "expected_impact": "+10-15 points", "icon": "🏫"})
        elif d.get("attendance", 0) < 80:
            recs.append({"category": "attendance", "priority": "medium", "message": "Improve attendance to 85%+.", "expected_impact": "+5-8 points", "icon": "🏫"})
        if d.get("sleep_hours", 0) < 6:
            recs.append({"category": "health", "priority": "high", "message": "Sleep deprivation detected. Target 7-8 hours.", "expected_impact": "+5-8 points", "icon": "😴"})
        if d.get("social_media_hours", 0) > 5:
            recs.append({"category": "lifestyle", "priority": "high", "message": "Excessive social media. Reduce to under 3 hrs.", "expected_impact": "+6-10 points", "icon": "📱"})
        if d.get("stress_level", 0) > 7:
            recs.append({"category": "wellbeing", "priority": "high", "message": "High stress. Practice mindfulness, seek help.", "expected_impact": "+4-7 points", "icon": "🧘"})
        if d.get("motivation_level", 0) < 4:
            recs.append({"category": "motivation", "priority": "high", "message": "Set small goals. Join study groups.", "expected_impact": "+5-8 points", "icon": "🎯"})
        if d.get("assignments_completed", 0) < 50:
            recs.append({"category": "academics", "priority": "critical", "message": "Assignment completion critically low.", "expected_impact": "+8-12 points", "icon": "📝"})
        if score >= 80:
            recs.append({"category": "excellence", "priority": "low", "message": "Excellent! Consider tutoring peers.", "expected_impact": "Maintain A grade", "icon": "🌟"})
        order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        recs.sort(key=lambda r: order.get(r["priority"], 99))
        return recs

_engine = None
def get_inference_engine(models_dir="models"):
    global _engine
    if _engine is None:
        _engine = InferenceEngine(models_dir)
        _engine.load()
    return _engine
