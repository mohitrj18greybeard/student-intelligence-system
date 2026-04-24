"""Model definitions: XGBoost, LightGBM, RF, Stacking ensembles."""
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, StackingRegressor, StackingClassifier
from sklearn.linear_model import Ridge, LogisticRegression
from xgboost import XGBRegressor, XGBClassifier
from lightgbm import LGBMRegressor, LGBMClassifier

def create_xgboost_regressor():
    return XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.1, random_state=42, n_jobs=-1, verbosity=0)

def create_lightgbm_regressor():
    return LGBMRegressor(n_estimators=300, num_leaves=31, learning_rate=0.1, random_state=42, n_jobs=-1, verbose=-1)

def create_random_forest_regressor():
    return RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1)

def create_stacking_regressor():
    estimators = [("xgb", create_xgboost_regressor()), ("lgbm", create_lightgbm_regressor()), ("rf", create_random_forest_regressor())]
    return StackingRegressor(estimators=estimators, final_estimator=Ridge(alpha=1.0), cv=5, n_jobs=-1)

def create_xgboost_classifier():
    return XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42, n_jobs=-1, verbosity=0, eval_metric="mlogloss")

def create_stacking_classifier():
    estimators = [("xgb", create_xgboost_classifier()), ("lgbm", LGBMClassifier(n_estimators=200, verbose=-1, random_state=42, n_jobs=-1))]
    return StackingClassifier(estimators=estimators, final_estimator=LogisticRegression(max_iter=1000), cv=5, n_jobs=-1)

def get_all_regression_models():
    return {"xgboost": create_xgboost_regressor(), "lightgbm": create_lightgbm_regressor(), "random_forest": create_random_forest_regressor()}
