"""
Model Training, Cross-Validation, Hyperparameter Optimization, and Metric Evaluation Engine.
Uses GroupKFold splits on Subject ID to prevent data leakage.
Implements Log-Target Optimization (z = ln(y)) and Non-Negative Stacking Meta-Learner.
"""

from pathlib import Path
from typing import Dict, Any, Tuple, List
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, HuberRegressor, ElasticNet
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, GridSearchCV
import lightgbm as lgb

from src.config import (
    TARGET_COLUMN,
    GROUP_CV_COLUMN,
    MODEL_SAVE_PATH,
    N_CV_SPLITS,
    RANDOM_SEED,
)
from src.clinical_evaluator import evaluate_iso_15197_compliance, evaluate_clarke_error_grid


class NonNegativeStackedEnsemblePipeline:
    """
    Production wrapper for Log-Target Base Models + Non-Negative Stacking Meta-Learner.
    Base models predict z = ln(y), meta-learner blends z_base, and final prediction = exp(z_stacked).
    """
    def __init__(self, ridge_model: Any, huber_model: Any, svr_model: Any, lgbm_model: Any, meta_model: Any):
        self.ridge_model = ridge_model
        self.huber_model = huber_model
        self.svr_model = svr_model
        self.lgbm_model = lgbm_model
        self.meta_model = meta_model

        if hasattr(ridge_model, "feature_names_in_"):
            self.feature_names_in_ = ridge_model.feature_names_in_

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "NonNegativeStackedEnsemblePipeline":
        z = np.log(y)
        self.ridge_model.fit(X, z)
        self.huber_model.fit(X, z)
        self.svr_model.fit(X, z)
        self.lgbm_model.fit(X, z)

        z_matrix = np.column_stack([
            self.ridge_model.predict(X),
            self.huber_model.predict(X),
            self.svr_model.predict(X),
            self.lgbm_model.predict(X),
        ])
        self.meta_model.fit(z_matrix, z)

        if hasattr(X, "columns"):
            self.feature_names_in_ = np.array(X.columns)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        z_preds = np.column_stack([
            self.ridge_model.predict(X),
            self.huber_model.predict(X),
            self.svr_model.predict(X),
            self.lgbm_model.predict(X),
        ])
        z_stacked = self.meta_model.predict(z_preds)
        return np.exp(z_stacked)


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
    Computes MAE, RMSE, MARD (%), R2, ISO 15197 Compliance (%), and Clarke Zone A (%) metrics.
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mard = calculate_mard(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    iso = evaluate_iso_15197_compliance(y_true, y_pred)
    zones_pct, _ = evaluate_clarke_error_grid(y_true, y_pred)
    return {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "MARD": float(mard),
        "R2": float(r2),
        "ISO_15197_Compliance": float(iso),
        "Clarke_Zone_A": float(zones_pct["A"]),
        "Clarke_Zone_A_B": float(zones_pct["A_B_Combined"]),
    }


def train_and_benchmark_models(
    df: pd.DataFrame,
    target_col: str = TARGET_COLUMN,
    group_col: str = GROUP_CV_COLUMN,
    n_splits: int = N_CV_SPLITS,
    seed: int = RANDOM_SEED,
) -> Tuple[Dict[str, Dict[str, float]], Any, np.ndarray, np.ndarray]:
    """
    Executes Log-Target Optimization (z = ln(y)) and Non-Negative Meta-Learner Stacking:
      1. Ridge (CV tuned alpha in [0.1, 1.0, 10.0, 100.0])
      2. HuberRegressor
      3. SVR_Linear (kernel="linear", C=1.0, epsilon=0.01)
      4. SVR_RBF (kernel="rbf", C=1.0, epsilon=0.01)
      5. ElasticNet (alpha=0.1, l1_ratio=0.5)
      6. LightGBM_Tuned
      7. Stacked_Ensemble (Ridge positive=True meta-learner)
    """
    drop_cols = [target_col, group_col]
    feature_cols = [c for c in df.columns if c not in drop_cols]

    X = df[feature_cols].copy()
    y = df[target_col].to_numpy()
    z = np.log(y)  # Log-space target transformation
    groups = df[group_col].to_numpy()
    gkf = GroupKFold(n_splits=n_splits)

    # 1. Tune Ridge Alpha in Log Space via CV in [0.1, 1.0, 10.0, 100.0]
    ridge_grid = GridSearchCV(
        estimator=Ridge(random_state=seed),
        param_grid={"alpha": [0.1, 1.0, 10.0, 50.0, 100.0]},
        cv=list(gkf.split(X, z, groups=groups)),
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
    )
    ridge_grid.fit(X, z)
    best_ridge_alpha = float(ridge_grid.best_params_["alpha"])
    print(f"[ModelTrainer] Optimal Log-Target Ridge Alpha selected: {best_ridge_alpha}")

    candidate_models = {
        "Ridge": Ridge(alpha=best_ridge_alpha, random_state=seed),
        "Huber": make_pipeline(StandardScaler(), HuberRegressor(max_iter=1000)),
        "SVR_Linear": make_pipeline(StandardScaler(), SVR(kernel="linear", C=1.0, epsilon=0.01)),
        "SVR_RBF": make_pipeline(StandardScaler(), SVR(kernel="rbf", C=1.0, epsilon=0.01)),
        "ElasticNet": make_pipeline(StandardScaler(), ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=seed)),
        "LightGBM_Tuned": lgb.LGBMRegressor(
            n_estimators=75,
            max_depth=3,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.7,
            reg_alpha=0.8,
            reg_lambda=1.5,
            random_state=seed,
            verbose=-1,
        ),
    }

    benchmark_results = {}
    oof_preds_log = {name: np.zeros(len(df)) for name in candidate_models.keys()}
    train_r2_scores = {name: [] for name in candidate_models.keys()}

    for name, model in candidate_models.items():
        for fold, (train_idx, val_idx) in enumerate(gkf.split(X, z, groups=groups)):
            X_train, z_train = X.iloc[train_idx], z[train_idx]
            X_val, z_val = X.iloc[val_idx], z[val_idx]

            model.fit(X_train, z_train)
            tr_z_pred = model.predict(X_train)
            val_z_pred = model.predict(X_val)

            oof_preds_log[name][val_idx] = val_z_pred
            # Evaluate train R2 on exponentiated glucose scale
            tr_y_pred = np.exp(tr_z_pred)
            train_r2_scores[name].append(r2_score(y[train_idx], tr_y_pred))

        oof_y_pred = np.exp(oof_preds_log[name])
        overall_metrics = evaluate_predictions(y, oof_y_pred)
        overall_metrics["Train_R2"] = float(np.mean(train_r2_scores[name]))
        overall_metrics["Test_OOF_R2"] = overall_metrics["R2"]
        overall_metrics["Gap"] = overall_metrics["Train_R2"] - overall_metrics["Test_OOF_R2"]
        benchmark_results[name] = overall_metrics

    # 2. Fit Non-Negative Ridge Stacking Meta-Learner
    meta_base_names = ["Ridge", "Huber", "SVR_Linear", "LightGBM_Tuned"]
    meta_X_oof = np.column_stack([oof_preds_log[m] for m in meta_base_names])

    meta_ridge = Ridge(alpha=1.0, positive=True, fit_intercept=False, random_state=seed)
    meta_ridge.fit(meta_X_oof, z)
    learned_weights = meta_ridge.coef_
    print(f"[ModelTrainer] Non-Negative Meta-Stacking Weights: {dict(zip(meta_base_names, np.round(learned_weights, 4)))}")

    oof_z_stacked = meta_ridge.predict(meta_X_oof)
    oof_y_stacked = np.exp(oof_z_stacked)

    # Compute GroupKFold train R2 for Stacked Ensemble
    stacked_tr_r2_list = []
    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, z, groups=groups)):
        X_train, z_train = X.iloc[train_idx], z[train_idx]
        tr_z_mat = []
        for m_name in meta_base_names:
            m_inst = candidate_models[m_name]
            m_inst.fit(X_train, z_train)
            tr_z_mat.append(m_inst.predict(X_train))
        tr_z_matrix = np.column_stack(tr_z_mat)
        m_meta = Ridge(alpha=1.0, positive=True, fit_intercept=False, random_state=seed)
        m_meta.fit(tr_z_matrix, z_train)
        tr_y_st = np.exp(m_meta.predict(tr_z_matrix))
        stacked_tr_r2_list.append(r2_score(y[train_idx], tr_y_st))

    stacked_metrics = evaluate_predictions(y, oof_y_stacked)
    stacked_metrics["Train_R2"] = float(np.mean(stacked_tr_r2_list))
    stacked_metrics["Test_OOF_R2"] = stacked_metrics["R2"]
    stacked_metrics["Gap"] = stacked_metrics["Train_R2"] - stacked_metrics["Test_OOF_R2"]
    benchmark_results["Stacked_Ensemble"] = stacked_metrics

    # 3. Train final NonNegativeStackedEnsemblePipeline on entire dataset
    ensemble_pipeline = NonNegativeStackedEnsemblePipeline(
        ridge_model=candidate_models["Ridge"],
        huber_model=candidate_models["Huber"],
        svr_model=candidate_models["SVR_Linear"],
        lgbm_model=candidate_models["LightGBM_Tuned"],
        meta_model=meta_ridge,
    )
    ensemble_pipeline.fit(X, y)

    return benchmark_results, ensemble_pipeline, y, oof_y_stacked


def save_model(model: Any, path: Path = MODEL_SAVE_PATH) -> None:
    """
    Saves trained ML model pipeline binary using joblib.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    print(f"[ModelTrainer] Best model pipeline saved to: {path}")
