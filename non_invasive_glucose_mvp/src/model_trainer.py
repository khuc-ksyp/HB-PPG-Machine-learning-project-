"""
Model Training, Cross-Validation, Hyperparameter Optimization, and Metric Evaluation Engine.
Uses GroupKFold splits on Subject ID to prevent data leakage.
"""

from pathlib import Path
from typing import Dict, Any, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GroupKFold, RandomizedSearchCV
import xgboost as xgb
import lightgbm as lgb

from src.config import (
    TARGET_COLUMN,
    GROUP_CV_COLUMN,
    MODEL_SAVE_PATH,
    N_CV_SPLITS,
    RANDOM_SEED,
)


def calculate_mard(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Computes Mean Absolute Relative Difference (MARD) percentage:
    MARD = (1/N) * sum(|y_true - y_pred| / y_true) * 100%
    """
    eps = 1e-8
    relative_diff = np.abs(y_true - y_pred) / (np.abs(y_true) + eps)
    return float(np.mean(relative_diff) * 100.0)


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Computes MAE, RMSE, and MARD (%) metrics.
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mard = calculate_mard(y_true, y_pred)
    return {"MAE": float(mae), "RMSE": float(rmse), "MARD": float(mard)}


def train_and_benchmark_models(
    df: pd.DataFrame,
    target_col: str = TARGET_COLUMN,
    group_col: str = GROUP_CV_COLUMN,
    n_splits: int = N_CV_SPLITS,
    seed: int = RANDOM_SEED,
) -> Tuple[Dict[str, Dict[str, float]], Any, np.ndarray, np.ndarray]:
    """
    Executes GroupKFold Cross Validation across standard ML regression models:
      - RandomForestRegressor
      - ExtraTreesRegressor
      - LGBMRegressor
      - XGBRegressor

    Returns:
        benchmark_results: Dictionary of CV metrics for each candidate model.
        best_model: Best performing model instance.
        oof_y_true: Out-of-fold ground truth target values.
        oof_y_pred: Out-of-fold predictions for the best model.
    """
    # Separate features, target, and grouping variable
    drop_cols = [target_col, group_col]
    feature_cols = [c for c in df.columns if c not in drop_cols]

    X = df[feature_cols].copy()
    y = df[target_col].to_numpy()
    groups = df[group_col].to_numpy()

    candidate_models = {
        "RandomForest": RandomForestRegressor(n_estimators=100, random_state=seed, n_jobs=-1),
        "ExtraTrees": ExtraTreesRegressor(n_estimators=100, random_state=seed, n_jobs=-1),
        "LightGBM": lgb.LGBMRegressor(n_estimators=100, random_state=seed, verbose=-1),
        "XGBoost": xgb.XGBRegressor(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=5,
            random_state=seed,
            n_jobs=-1,
        ),
    }

    gkf = GroupKFold(n_splits=n_splits)
    benchmark_results = {}
    oof_predictions = {name: np.zeros(len(df)) for name in candidate_models.keys()}

    for name, model in candidate_models.items():
        print(f"[ModelTrainer] Benchmarking {name} with GroupKFold (k={n_splits})...")
        mae_scores, rmse_scores, mard_scores = [], [], []

        for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups=groups)):
            X_train, y_train = X.iloc[train_idx], y[train_idx]
            X_val, y_val = X.iloc[val_idx], y[val_idx]

            model.fit(X_train, y_train)
            preds = model.predict(X_val)

            oof_predictions[name][val_idx] = preds

            fold_metrics = evaluate_predictions(y_val, preds)
            mae_scores.append(fold_metrics["MAE"])
            rmse_scores.append(fold_metrics["RMSE"])
            mard_scores.append(fold_metrics["MARD"])

        overall_metrics = evaluate_predictions(y, oof_predictions[name])
        benchmark_results[name] = overall_metrics
        print(
            f"  -> {name} OOF Results | MAE: {overall_metrics['MAE']:.2f} mg/dL | "
            f"RMSE: {overall_metrics['RMSE']:.2f} mg/dL | MARD: {overall_metrics['MARD']:.2f}%"
        )

    # Select best baseline model based on lowest MARD
    best_model_name = min(benchmark_results, key=lambda k: benchmark_results[k]["MARD"])
    print(f"\n[ModelTrainer] Best baseline model selected: {best_model_name}")

    # Define hyperparameter distribution maps
    param_dists = {
        "XGBoost": (
            xgb.XGBRegressor(random_state=seed, n_jobs=-1),
            {
                "n_estimators": [100, 150, 200, 300],
                "max_depth": [3, 4, 5, 6, 7],
                "learning_rate": [0.01, 0.03, 0.05, 0.1],
                "subsample": [0.7, 0.8, 0.9, 1.0],
                "colsample_bytree": [0.6, 0.7, 0.8, 1.0],
                "reg_alpha": [0.0, 0.1, 0.5, 1.0],
                "reg_lambda": [0.5, 1.0, 2.0],
            }
        ),
        "LightGBM": (
            lgb.LGBMRegressor(random_state=seed, verbose=-1),
            {
                "n_estimators": [100, 150, 200, 300],
                "max_depth": [-1, 3, 5, 7, 10],
                "num_leaves": [15, 31, 63],
                "learning_rate": [0.01, 0.03, 0.05, 0.1],
                "subsample": [0.7, 0.8, 0.9, 1.0],
                "colsample_bytree": [0.6, 0.7, 0.8, 1.0],
            }
        ),
        "ExtraTrees": (
            ExtraTreesRegressor(random_state=seed, n_jobs=-1),
            {
                "n_estimators": [100, 150, 200, 300],
                "max_depth": [None, 10, 15, 20],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
                "max_features": ["sqrt", "log2", 1.0],
            }
        ),
        "RandomForest": (
            RandomForestRegressor(random_state=seed, n_jobs=-1),
            {
                "n_estimators": [100, 150, 200, 300],
                "max_depth": [None, 10, 15, 20],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
                "max_features": ["sqrt", "log2", 1.0],
            }
        ),
    }

    # Fallback to XGBoost if unknown
    base_estimator, param_dist = param_dists.get(
        best_model_name,
        param_dists["XGBoost"]
    )

    print(f"[ModelTrainer] Running RandomizedSearchCV hyperparameter tuning on {best_model_name}...")
    cv_splits = list(gkf.split(X, y, groups=groups))
    search = RandomizedSearchCV(
        estimator=base_estimator,
        param_distributions=param_dist,
        n_iter=15,
        cv=cv_splits,
        scoring="neg_mean_absolute_error",
        random_state=seed,
        n_jobs=-1,
    )
    search.fit(X, y)
    best_tuned_model = search.best_estimator_

    # Final Out-Of-Fold Evaluation for Tuned Model
    tuned_oof_preds = np.zeros(len(df))
    for train_idx, val_idx in cv_splits:
        X_train, y_train = X.iloc[train_idx], y[train_idx]
        X_val = X.iloc[val_idx]

        model_fold = search.estimator.__class__(**best_tuned_model.get_params())
        model_fold.fit(X_train, y_train)
        tuned_oof_preds[val_idx] = model_fold.predict(X_val)

    tuned_metrics = evaluate_predictions(y, tuned_oof_preds)
    tuned_key = f"Tuned_{best_model_name}"
    benchmark_results[tuned_key] = tuned_metrics
    print(
        f"[ModelTrainer] {tuned_key} OOF | MAE: {tuned_metrics['MAE']:.2f} mg/dL | "
        f"RMSE: {tuned_metrics['RMSE']:.2f} mg/dL | MARD: {tuned_metrics['MARD']:.2f}%"
    )

    # Fit best model on entire dataset for production export
    best_tuned_model.fit(X, y)

    return benchmark_results, best_tuned_model, y, tuned_oof_preds



def save_model(model: Any, path: Path = MODEL_SAVE_PATH) -> None:
    """
    Saves trained ML model pipeline binary using joblib.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    print(f"[ModelTrainer] Best model pipeline saved to: {path}")
