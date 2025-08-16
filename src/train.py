import argparse, json, os, joblib
import numpy as np
import pandas as pd
from typing import Tuple, List

from sklearn.model_selection import train_test_split, TimeSeriesSplit, RandomizedSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, f1_score, roc_auc_score
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from xgboost import XGBRegressor, XGBClassifier

RANDOM_STATE = 42

def infer_feature_types(df: pd.DataFrame, target: str) -> Tuple[List[str], List[str]]:
    features = df.drop(columns=[target])
    numeric_cols = features.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = [c for c in features.columns if c not in numeric_cols]
    return numeric_cols, categorical_cols

def build_preprocessor(numeric_cols, categorical_cols):
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    preprocessor = ColumnTransformer([
        ("num", numeric_pipe, numeric_cols),
        ("cat", categorical_pipe, categorical_cols)
    ])
    return preprocessor

def get_model(task: str, model_name: str):
    if task == "regression":
        if model_name == "random_forest":
            return RandomForestRegressor(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1)
        elif model_name == "xgboost":
            return XGBRegressor(
                n_estimators=500,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=RANDOM_STATE,
                n_jobs=-1
            )
    elif task == "classification":
        if model_name == "random_forest":
            return RandomForestClassifier(n_estimators=400, random_state=RANDOM_STATE, n_jobs=-1, class_weight="balanced")
        elif model_name == "xgboost":
            return XGBClassifier(
                n_estimators=600,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=RANDOM_STATE,
                n_jobs=-1,
                eval_metric="logloss"
            )
    raise ValueError("Invalid (task, model_name) combination")

def evaluate(task: str, y_true, y_pred, y_proba=None):
    if task == "regression":
        rmse = mean_squared_error(y_true, y_pred) ** 0.5
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        return {"rmse": rmse, "mae": mae, "r2": r2}
    else:
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average="weighted")
        metrics = {"accuracy": acc, "f1_weighted": f1}
        if y_proba is not None and (y_proba.ndim == 1 or y_proba.shape[1] == 2):
            # binary AUC if available
            proba_pos = y_proba if y_proba.ndim == 1 else y_proba[:, 1]
            try:
                auc = roc_auc_score(y_true, proba_pos)
                metrics["roc_auc"] = auc
            except ValueError:
                pass
        return metrics

def main(args):
    os.makedirs("artifacts", exist_ok=True)
    df = pd.read_csv(args.data_path)
    if args.target not in df.columns:
        raise ValueError(f"Target '{args.target}' not in dataset columns: {df.columns.tolist()}")

    numeric_cols, categorical_cols = infer_feature_types(df, args.target)
    preprocessor = build_preprocessor(numeric_cols, categorical_cols)

    X = df.drop(columns=[args.target])
    y = df[args.target]

    # Train/validation split (for time series, override with --timeseries)
    if args.timeseries:
        # Use last 20% as test to respect chronology
        split_idx = int(len(df) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y if args.task=="classification" else None
        )

    model = get_model(args.task, args.model)
    pipe = Pipeline([
        ("prep", preprocessor),
        ("model", model)
    ])

    pipe.fit(X_train, y_train)

    # Predictions & metrics
    if args.task == "classification":
        y_pred = pipe.predict(X_test)
        y_proba = None
        # proba if supported
        try:
            y_proba = pipe.predict_proba(X_test)
        except Exception:
            pass
        metrics = evaluate(args.task, y_test, y_pred, y_proba)
    else:
        y_pred = pipe.predict(X_test)
        metrics = evaluate(args.task, y_test, y_pred)

    # Save model pipeline & metadata
    model_path = os.path.join("artifacts", "model.joblib")
    joblib.dump(pipe, model_path)

    meta = {
        "task": args.task,
        "model": args.model,
        "target": args.target,
        "numeric_features": numeric_cols,
        "categorical_features": categorical_cols,
        "metrics": metrics
    }
    with open(os.path.join("artifacts", "metrics.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print("=== Training complete ===")
    print(json.dumps(meta, indent=2))
    print(f"Saved pipeline -> {model_path}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", type=str, default="data/train.csv")
    p.add_argument("--task", type=str, choices=["regression", "classification"], required=True)
    p.add_argument("--target", type=str, required=True)
    p.add_argument("--model", type=str, choices=["random_forest", "xgboost"], default="random_forest")
    p.add_argument("--timeseries", action="store_true", help="Use chronological split instead of random split")
    args = p.parse_args()
    main(args)
