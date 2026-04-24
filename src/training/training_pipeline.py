"""Training pipeline: generate -> engineer -> train -> evaluate -> save."""
import os, json, time, joblib
import numpy as np, pandas as pd
from pathlib import Path
from datetime import datetime
from src.data.data_generator import generate_student_data
from src.data.feature_engineering import create_features
from src.data.preprocessing import preprocess_data, preprocess_classification_data, save_preprocessor
from src.models.model_definitions import (create_xgboost_regressor, create_lightgbm_regressor,
    create_random_forest_regressor, create_stacking_regressor, create_xgboost_classifier, create_stacking_classifier)
from src.models.evaluation import evaluate_regression, evaluate_classification, get_feature_importance
from src.utils.logger import get_logger

logger = get_logger("training.pipeline")

try:
    import mlflow, mlflow.sklearn
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

class TrainingPipeline:
    def __init__(self, output_dir="models"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.best_model = self.best_model_name = self.preprocessor = None
        self.clf_preprocessor = self.label_encoder = self.feature_names = self.clf_feature_names = None
        self.results = {}

    def run(self, n_students=3000, seed=42, use_mlflow=True):
        start = time.time()
        logger.info("=" * 50 + " TRAINING PIPELINE START " + "=" * 50)
        if use_mlflow and MLFLOW_AVAILABLE:
            mlflow.set_tracking_uri("mlruns")
            mlflow.set_experiment("edupulse")

        df = generate_student_data(n_students, seed)
        Path("data").mkdir(exist_ok=True)
        df.to_csv("data/student_data.csv", index=False)
        df = create_features(df)

        reg_results = self._train_regression(df, use_mlflow)
        clf_results = self._train_classification(df, use_mlflow)
        self._save_artifacts()

        elapsed = time.time() - start
        self.results = {"timestamp": datetime.now().isoformat(), "n_students": n_students,
                        "n_features": len(self.feature_names) if self.feature_names else 0,
                        "regression": reg_results, "classification": clf_results,
                        "best_regression_model": self.best_model_name, "elapsed_seconds": round(elapsed, 2)}
        with open(self.output_dir / "training_results.json", "w") as f:
            json.dump(self._ser(self.results), f, indent=2, default=str)
        logger.info(f"PIPELINE COMPLETE in {elapsed:.1f}s | Best: {self.best_model_name}")
        return self.results

    def _train_regression(self, df, use_mlflow):
        X_train, X_test, y_train, y_test, preprocessor, feature_names = preprocess_data(df)
        self.preprocessor, self.feature_names = preprocessor, feature_names
        save_preprocessor(preprocessor, str(self.output_dir / "preprocessor.joblib"))
        models = {"xgboost": create_xgboost_regressor(), "lightgbm": create_lightgbm_regressor(), "random_forest": create_random_forest_regressor()}
        results, best_r2 = {}, -999
        for name, model in models.items():
            logger.info(f"  Training {name}...")
            model.fit(X_train, y_train)
            metrics = evaluate_regression(model, X_test, y_test, name, X_train, y_train)
            results[name] = metrics
            if metrics["r2"] > best_r2:
                best_r2, self.best_model, self.best_model_name = metrics["r2"], model, name
            fi = get_feature_importance(model, feature_names)
            if not fi.empty:
                fi.to_csv(self.output_dir / f"feature_importance_{name}.csv", index=False)
        try:
            logger.info("  Training stacking ensemble...")
            stacker = create_stacking_regressor()
            stacker.fit(X_train, y_train)
            sm = evaluate_regression(stacker, X_test, y_test, "stacking_ensemble", X_train, y_train)
            results["stacking_ensemble"] = sm
            if sm["r2"] > best_r2:
                self.best_model, self.best_model_name = stacker, "stacking_ensemble"
        except Exception as e:
            logger.warning(f"Stacking failed: {e}")
        return results

    def _train_classification(self, df, use_mlflow):
        try:
            X_train, X_test, y_train, y_test, preprocessor, le, feature_names = preprocess_classification_data(df)
            self.clf_preprocessor, self.label_encoder, self.clf_feature_names = preprocessor, le, feature_names
            save_preprocessor(preprocessor, str(self.output_dir / "clf_preprocessor.joblib"))
            joblib.dump(le, str(self.output_dir / "label_encoder.joblib"))
            models = {"xgboost_clf": create_xgboost_classifier(), "stacking_clf": create_stacking_classifier()}
            results = {}
            for name, model in models.items():
                logger.info(f"  Training {name}...")
                model.fit(X_train, y_train)
                results[name] = evaluate_classification(model, X_test, y_test, name, le.classes_.tolist())
            best_name = max(results, key=lambda k: results[k]["f1_weighted"])
            joblib.dump(models[best_name], str(self.output_dir / "risk_classifier.joblib"))
            return results
        except Exception as e:
            logger.error(f"Classification failed: {e}")
            return {}

    def _save_artifacts(self):
        if self.best_model:
            joblib.dump(self.best_model, str(self.output_dir / "best_regression_model.joblib"))
        if self.feature_names:
            with open(self.output_dir / "feature_names.json", "w") as f:
                json.dump(self.feature_names, f)
        if self.clf_feature_names:
            with open(self.output_dir / "clf_feature_names.json", "w") as f:
                json.dump(self.clf_feature_names, f)

    @staticmethod
    def _ser(obj):
        if isinstance(obj, dict): return {k: TrainingPipeline._ser(v) for k, v in obj.items()}
        if isinstance(obj, list): return [TrainingPipeline._ser(v) for v in obj]
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return obj

def run_training_pipeline(n_students=3000, seed=42, use_mlflow=True, output_dir="models"):
    return TrainingPipeline(output_dir=output_dir).run(n_students, seed, use_mlflow)
