"""Sklearn preprocessing pipelines."""
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from src.data.feature_engineering import get_feature_columns
from src.utils.logger import get_logger

logger = get_logger("data.preprocessing")

def preprocess_data(df):
    from src.data.feature_engineering import create_features
    if "study_efficiency" not in df.columns:
        df = create_features(df)
    numerical_cols, categorical_cols = get_feature_columns(df)
    X = df[numerical_cols + categorical_cols]
    y = df["final_score"]
    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), numerical_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols),
    ], remainder="drop")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train_t = preprocessor.fit_transform(X_train)
    X_test_t = preprocessor.transform(X_test)
    cat_features = []
    if hasattr(preprocessor.named_transformers_["cat"], "get_feature_names_out"):
        cat_features = list(preprocessor.named_transformers_["cat"].get_feature_names_out(categorical_cols))
    feature_names = numerical_cols + cat_features
    logger.info(f"Regression preprocessing: {X_train_t.shape[1]} features, {len(X_train)} train, {len(X_test)} test")
    return X_train_t, X_test_t, y_train.values, y_test.values, preprocessor, feature_names

def preprocess_classification_data(df):
    from src.data.feature_engineering import create_features
    if "study_efficiency" not in df.columns:
        df = create_features(df)
    numerical_cols, categorical_cols = get_feature_columns(df)
    X = df[numerical_cols + categorical_cols]
    le = LabelEncoder()
    y = le.fit_transform(df["risk_category"])
    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), numerical_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols),
    ], remainder="drop")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_train_t = preprocessor.fit_transform(X_train)
    X_test_t = preprocessor.transform(X_test)
    cat_features = []
    if hasattr(preprocessor.named_transformers_["cat"], "get_feature_names_out"):
        cat_features = list(preprocessor.named_transformers_["cat"].get_feature_names_out(categorical_cols))
    feature_names = numerical_cols + cat_features
    logger.info(f"Classification preprocessing: {X_train_t.shape[1]} features, classes={le.classes_}")
    return X_train_t, X_test_t, y_train, y_test, preprocessor, le, feature_names

def save_preprocessor(preprocessor, path):
    joblib.dump(preprocessor, path)
