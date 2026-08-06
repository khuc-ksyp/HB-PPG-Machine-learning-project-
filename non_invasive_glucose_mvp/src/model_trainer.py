"""
Model Training, Cross-Validation, Hyperparameter Optimization, and Metric Evaluation Engine.
Uses GroupKFold splits on Subject ID to prevent data leakage.
Implements Log-Target Optimization (z = ln(y)), Base Model Pruning, and MARD-Direct Loss Optimization.
"""

from pathlib import Path
from typing import Dict, Any, Tuple, List
import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize
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


def mard_loss_function(weights: np.ndarray, z_matrix: np.ndarray, y_true: np.ndarray) -> float:
    """
    Direct MARD Loss Objective Function:
    Loss(w) = (1/N) * sum( |y_i - exp( sum_m w_m * z_{i,m} )| / y_i ) * 100
    """
    z_blend = np.dot(z_matrix, weights)
    y_pred = np.exp(z_blend)
    return float(np.mean(np.abs(y_true - y_pred) / y_true) * 100.0)


class MARDOptimizedEnsemblePipeline:
    """
    Production wrapper for Log-Target Base Models + MARD-Direct Loss SLSQP Meta-Learner.
    Predicts y = exp( sum_m w_m * z_{base, m} ).
    """
    def __init__(self, models_dict: Dict[str, Any], weights_dict: Dict[str, float]):
        self.models_dict = models_dict
        self.weights_dict = weights_dict
        self.model_names = list(models_dict.keys())
        self.weights_arr = np.array([weights_dict[m] for m in self.model_names])

        first_m = list(models_dict.values())[0]
        if hasattr(first_m, "feature_names_in_"):
            self.feature_names_in_ = first_m.feature_names_in_

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "MARDOptimizedEnsemblePipeline":
        z = np.log(y)
        z_cols = []
        for name, m in self.models_dict.items():
            m.fit(X, z)
            z_cols.append(m.predict(X))

        z_mat = np.column_stack(z_cols)
        n_m = len(self.model_names)
        init_w = np.ones(n_m) / n_m
        bounds = [(0.0, 1.0) for _ in range(n_m)]
        constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})

        res = minimize(mard_loss_function, init_w, args=(z_mat, y), method='SLSQP', bounds=bounds, constraints=constraints)
        self.weights_arr = res.x
        self.weights_dict = dict(zip(self.model_names, self.weights_arr))

        if hasattr(X, "columns"):
            self.feature_names_in_ = np.array(X.columns)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        z_cols = [self.models_dict[m].predict(X) for m in self.model_names]
        z_mat = np.column_stack(z_cols)
        z_blend = np.dot(z_mat, self.weights_arr)
        return np.exp(z_blend)


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
    Executes Log-Target Optimization (z = ln(y)), Base Model MARD Pruning, and MARD-Direct Loss Meta-Optimization:
      1. Ridge (CV tuned alpha in [0.1, 1.0, 10.0, 100.0])
      2. HuberRegressor
      3. SVR_Linear (kernel="linear", C=1.0, epsilon=0.01)
      4. SVR_RBF (kernel="rbf", C=1.0, epsilon=0.01)
      5. ElasticNet (alpha=0.1, l1_ratio=0.5)
      6. LightGBM_Tuned
      7. MARD_Opt_Ensemble (SciPy SLSQP direct MARD loss minimization)
    """
    drop_cols = [target_col, group_col]
    feature_cols = [c for c in df.columns if c not in drop_cols]

    X = df[feature_cols].copy()
    y = df[target_col].to_numpy()
    z = np.log(y)
    groups = df[group_col].to_numpy()
    gkf = GroupKFold(n_splits=n_splits)

    # 1. Tune Ridge Alpha in Log Space
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
            tr_y_pred = np.exp(tr_z_pred)
            train_r2_scores[name].append(r2_score(y[train_idx], tr_y_pred))

        oof_y_pred = np.exp(oof_preds_log[name])
        overall_metrics = evaluate_predictions(y, oof_y_pred)
        overall_metrics["Train_R2"] = float(np.mean(train_r2_scores[name]))
        overall_metrics["Test_OOF_R2"] = overall_metrics["R2"]
        overall_metrics["Gap"] = overall_metrics["Train_R2"] - overall_metrics["Test_OOF_R2"]
        benchmark_results[name] = overall_metrics

    # 2. Base Model Pruning: Retain candidate models with OOF MARD <= 10.20% (always keep top linear models)
    retained_models = [m for m in candidate_models.keys() if benchmark_results[m]["MARD"] <= 10.20 or m in ["Ridge", "Huber", "SVR_Linear"]]
    print(f"[ModelTrainer] Retained Base Models for MARD Meta-Stacking: {retained_models}")

    meta_X_oof = np.column_stack([oof_preds_log[m] for m in retained_models])
    n_m = len(retained_models)
    init_w = np.ones(n_m) / n_m
    bounds = [(0.0, 1.0) for _ in range(n_m)]
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})

    res = minimize(mard_loss_function, init_w, args=(meta_X_oof, y), method='SLSQP', bounds=bounds, constraints=constraints)
    learned_weights = dict(zip(retained_models, np.round(res.x, 4)))
    print(f"[ModelTrainer] MARD-Optimized Meta Weights: {learned_weights}")

    oof_z_opt = np.dot(meta_X_oof, res.x)
    oof_y_opt = np.exp(oof_z_opt)

    # Compute GroupKFold train R2 for MARD-Optimized Ensemble
    opt_tr_r2_list = []
    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, z, groups=groups)):
        X_train, z_train = X.iloc[train_idx], z[train_idx]
        tr_z_mat = []
        for m_name in retained_models:
            m_inst = candidate_models[m_name]
            m_inst.fit(X_train, z_train)
            tr_z_mat.append(m_inst.predict(X_train))
        tr_z_matrix = np.column_stack(tr_z_mat)
        res_fold = minimize(mard_loss_function, init_w, args=(tr_z_matrix, y[train_idx]), method='SLSQP', bounds=bounds, constraints=constraints)
        tr_y_st = np.exp(np.dot(tr_z_matrix, res_fold.x))
        opt_tr_r2_list.append(r2_score(y[train_idx], tr_y_st))

    opt_metrics = evaluate_predictions(y, oof_y_opt)
    opt_metrics["Train_R2"] = float(np.mean(opt_tr_r2_list))
    opt_metrics["Test_OOF_R2"] = opt_metrics["R2"]
    opt_metrics["Gap"] = opt_metrics["Train_R2"] - opt_metrics["Test_OOF_R2"]
    benchmark_results["MARD_Opt_Ensemble"] = opt_metrics

    # 3. Fit final MARDOptimizedEnsemblePipeline on full dataset
    retained_dict = {m: candidate_models[m] for m in retained_models}
    ensemble_pipeline = MARDOptimizedEnsemblePipeline(
        models_dict=retained_dict,
        weights_dict=learned_weights,
    )
    ensemble_pipeline.fit(X, y)

    return benchmark_results, ensemble_pipeline, y, oof_y_opt


def save_model(model: Any, path: Path = MODEL_SAVE_PATH) -> None:
    """
    Saves trained ML model pipeline binary using joblib.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    print(f"[ModelTrainer] Best model pipeline saved to: {path}")
