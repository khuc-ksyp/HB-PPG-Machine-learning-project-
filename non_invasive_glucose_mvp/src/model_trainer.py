"""
Model Training, Cross-Validation, Hyperparameter Optimization, and Metric Evaluation Engine.
Uses GroupKFold splits on Subject ID to prevent data leakage.
Implements Log-Target Optimization (z = ln(y)), Tail-Weighted Fitting, Base Model Pruning, and ISO Safety Loss Optimization.
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
from sklearn.pipeline import Pipeline, make_pipeline
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
from src.clinical_evaluator import (
    evaluate_iso_15197_compliance,
    evaluate_clarke_error_grid,
    apply_iso_range_expansion,
)


def compute_tail_sample_weights(y_arr: np.ndarray) -> np.ndarray:
    """
    Computes dynamic tail sample weights to mitigate regression shrinkage in hypo/hyper regimes:
      - y < 100: w = 1.0 + 2.0 * ((100 - y) / 100) ** 2
      - 100 <= y <= 180: w = 1.0
      - y > 180: w = 1.0 + 2.0 * ((y - 180) / 180) ** 2
    """
    w = np.ones(len(y_arr))
    mask_low = y_arr < 100.0
    mask_high = y_arr > 180.0
    w[mask_low] = 1.0 + 2.0 * ((100.0 - y_arr[mask_low]) / 100.0) ** 2
    w[mask_high] = 1.0 + 2.0 * ((y_arr[mask_high] - 180.0) / 180.0) ** 2
    return w


def fit_model_with_sample_weights(model: Any, X: pd.DataFrame, z: np.ndarray, sw: np.ndarray) -> Any:
    """
    Fits scikit-learn estimator or Pipeline passing sample_weight to final estimator step.
    """
    if hasattr(model, "steps"):
        final_step_name = model.steps[-1][0]
        model.fit(X, z, **{f"{final_step_name}__sample_weight": sw})
    elif hasattr(model, "fit"):
        try:
            model.fit(X, z, sample_weight=sw)
        except TypeError:
            model.fit(X, z)
    return model


def iso_safety_loss_function(weights: np.ndarray, z_matrix: np.ndarray, y_true: np.ndarray) -> float:
    """
    ISO-Aware MAE, Variance Expansion & Slope Penalty Custom Meta-Learner Loss Function:
      - Base Loss: MAE = mean(abs(y_true - y_pred))
      - Variance Expansion Loss: std_loss = abs(1.0 - (std(y_pred) / (std(y_true) + 1e-8)))
      - Slope Penalty Loss: slope_loss = (1.0 - slope)^2
      - Total Loss: MAE + 10.0 * std_loss + 10.0 * slope_loss
    """
    z_blend = np.dot(z_matrix, weights)
    y_pred = np.exp(z_blend)
    abs_diff = np.abs(y_pred - y_true)

    mae = float(np.mean(abs_diff))

    std_true = float(np.std(y_true))
    std_pred = float(np.std(y_pred))
    std_loss = float(np.abs(1.0 - (std_pred / (std_true + 1e-8))))

    var_true = float(np.var(y_true))
    cov = float(np.cov(y_true, y_pred)[0, 1]) if var_true > 1e-8 else 0.0
    slope = cov / (var_true + 1e-8)
    slope_loss = float((1.0 - slope) ** 2)

    return mae + 10.0 * std_loss + 10.0 * slope_loss


class MARDOptimizedEnsemblePipeline:
    """
    Production wrapper for Log-Target Base Models + ISO Safety SLSQP Meta-Learner + Monotonic Target Expander.
    Predicts y = apply_iso_range_expansion(exp( sum_m w_m * z_{base, m} )).
    """
    def __init__(
        self,
        models_dict: Dict[str, Any],
        weights_dict: Dict[str, float],
        low_threshold: float = 98.0,
        high_threshold: float = 165.0,
        low_gain: float = 1.22,
        high_gain: float = 1.18,
    ):
        self.models_dict = models_dict
        self.weights_dict = weights_dict
        self.model_names = list(models_dict.keys())
        self.weights_arr = np.array([weights_dict[m] for m in self.model_names])
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
        self.low_gain = low_gain
        self.high_gain = high_gain

        first_m = list(models_dict.values())[0]
        if hasattr(first_m, "feature_names_in_"):
            self.feature_names_in_ = first_m.feature_names_in_

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "MARDOptimizedEnsemblePipeline":
        if hasattr(X, "columns"):
            self.feature_names_in_ = np.array(X.columns)
        z = np.log(y)
        sw = compute_tail_sample_weights(y)
        z_cols = []
        for name, m in self.models_dict.items():
            fit_model_with_sample_weights(m, X, z, sw)
            z_cols.append(np.asarray(m.predict(X), dtype=float))

        z_mat = np.column_stack(z_cols)
        n_m = len(self.model_names)
        init_w = np.ones(n_m) / n_m
        bounds = [(0.0, 1.0) for _ in range(n_m)]
        constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})

        res = minimize(iso_safety_loss_function, init_w, args=(z_mat, y), method='SLSQP', bounds=bounds, constraints=constraints)
        self.weights_arr = res.x
        self.weights_dict = dict(zip(self.model_names, [float(w) for w in np.round(res.x, 4)]))

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if hasattr(self, "feature_names_in_") and hasattr(X, "columns"):
            missing_cols = [c for c in self.feature_names_in_ if c not in X.columns]
            if missing_cols:
                X = X.copy()
                for c in missing_cols:
                    X[c] = 0.0
            X = X[self.feature_names_in_]

        z_cols = [np.asarray(self.models_dict[m].predict(X), dtype=float) for m in self.model_names]
        z_mat = np.column_stack(z_cols)
        z_blend = np.dot(z_mat, self.weights_arr)
        y_raw = np.exp(z_blend)
        return apply_iso_range_expansion(
            y_raw,
            low_threshold=self.low_threshold,
            high_threshold=self.high_threshold,
            low_gain=self.low_gain,
            high_gain=self.high_gain,
        )


def calculate_mard(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Computes Mean Absolute Relative Difference (MARD) percentage.
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
        "MAE": mae,
        "RMSE": rmse,
        "MARD": mard,
        "R2": r2,
        "ISO_15197_Compliance": iso,
        "Clarke_Zone_A": zones_pct["A"],
        "Clarke_Zone_A_B": zones_pct["A_B_Combined"],
    }


def train_and_benchmark_models(
    df: pd.DataFrame,
    target_col: str = TARGET_COLUMN,
    group_col: str = GROUP_CV_COLUMN,
    n_splits: int = N_CV_SPLITS,
    seed: int = RANDOM_SEED,
) -> Tuple[Dict[str, Dict[str, float]], Any, np.ndarray, np.ndarray]:
    """
    Executes Log-Target Optimization (z = ln(y)), Tail-Weighted Fitting, Base Model Pruning, and ISO Safety Loss Meta-Optimization:
      1. Ridge (alpha=[0.01, 0.1, 1.0, 10.0])
      2. HuberRegressor (alpha=[0.0001, 0.001, 0.01], epsilon=1.35)
      3. SVR_Linear (C=[1.0, 10.0, 100.0], epsilon=0.01)
      4. SVR_RBF (kernel="rbf", C=1.0, epsilon=0.01)
      5. ElasticNet (alpha=0.1, l1_ratio=0.5)
      6. LightGBM_Tuned (objective="regression_l1")
      7. MARD_Opt_Ensemble (ISO Safety & Slope Penalty SLSQP minimization)
    """
    drop_cols = [target_col, group_col]
    feature_cols = [c for c in df.columns if c not in drop_cols]

    X = df[feature_cols].copy()
    y = np.asarray(df[target_col].values, dtype=float)
    z = np.log(y)
    groups = np.asarray(df[group_col].values)
    gkf = GroupKFold(n_splits=n_splits)
    cv_splits = list(gkf.split(X, z, groups=groups))

    # 1. Un-constrain Base Model Regularization via GridSearchCV
    ridge_grid = GridSearchCV(
        estimator=Ridge(random_state=seed),
        param_grid={"alpha": [0.01, 0.1, 1.0, 10.0]},
        cv=cv_splits,
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
    )
    ridge_grid.fit(X, z)
    best_ridge_alpha = float(ridge_grid.best_params_["alpha"])

    huber_grid = GridSearchCV(
        estimator=Pipeline([("scaler", StandardScaler()), ("model", HuberRegressor(epsilon=1.35, max_iter=1000))]),
        param_grid={"model__alpha": [0.0001, 0.001, 0.01]},
        cv=cv_splits,
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
    )
    huber_grid.fit(X, z)
    best_huber_alpha = float(huber_grid.best_params_["model__alpha"])

    svr_grid = GridSearchCV(
        estimator=Pipeline([("scaler", StandardScaler()), ("model", SVR(kernel="linear", epsilon=0.01))]),
        param_grid={"model__C": [1.0, 10.0, 100.0]},
        cv=cv_splits,
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
    )
    svr_grid.fit(X, z)
    best_svr_c = float(svr_grid.best_params_["model__C"])

    print(f"[ModelTrainer] Tuned Base Regularization: Ridge alpha={best_ridge_alpha}, Huber alpha={best_huber_alpha}, SVR C={best_svr_c}")

    candidate_models = {
        "Ridge": Ridge(alpha=best_ridge_alpha, random_state=seed),
        "Huber": Pipeline([("scaler", StandardScaler()), ("model", HuberRegressor(alpha=best_huber_alpha, epsilon=1.35, max_iter=1000))]),
        "SVR_Linear": Pipeline([("scaler", StandardScaler()), ("model", SVR(kernel="linear", C=best_svr_c, epsilon=0.01))]),
        "SVR_RBF": Pipeline([("scaler", StandardScaler()), ("model", SVR(kernel="rbf", C=1.0, epsilon=0.01))]),
        "ElasticNet": Pipeline([("scaler", StandardScaler()), ("model", ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=seed))]),
        "LightGBM_Tuned": lgb.LGBMRegressor(
            objective="regression_l1",
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
            X_train, z_train, y_train = X.iloc[train_idx], z[train_idx], y[train_idx]
            X_val, z_val = X.iloc[val_idx], z[val_idx]
            sw_train = compute_tail_sample_weights(y_train)

            fit_model_with_sample_weights(model, X_train, z_train, sw_train)
            tr_z_pred = np.asarray(model.predict(X_train), dtype=float)
            val_z_pred = np.asarray(model.predict(X_val), dtype=float)

            oof_preds_log[name][val_idx] = val_z_pred
            tr_y_pred = np.exp(tr_z_pred)
            train_r2_scores[name].append(r2_score(y_train, tr_y_pred))

        oof_y_pred = np.exp(oof_preds_log[name])
        overall_metrics = evaluate_predictions(y, oof_y_pred)
        overall_metrics["Train_R2"] = float(np.mean(train_r2_scores[name]))
        overall_metrics["Test_OOF_R2"] = overall_metrics["R2"]
        overall_metrics["Gap"] = overall_metrics["Train_R2"] - overall_metrics["Test_OOF_R2"]
        benchmark_results[name] = overall_metrics

    # 2. Base Model Pruning: Retain candidate models with OOF MAE <= 12.0 mg/dL or core L1 models
    retained_models = [m for m in candidate_models.keys() if benchmark_results[m]["MAE"] <= 12.0 or m in ["Ridge", "Huber", "SVR_Linear", "LightGBM_Tuned"]]
    print(f"[ModelTrainer] Retained Base Models for ISO Meta-Stacking: {retained_models}")

    meta_X_oof = np.column_stack([oof_preds_log[m] for m in retained_models])
    n_m = len(retained_models)
    init_w = np.ones(n_m) / n_m
    bounds = [(0.0, 1.0) for _ in range(n_m)]
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})

    res = minimize(iso_safety_loss_function, init_w, args=(meta_X_oof, y), method='SLSQP', bounds=bounds, constraints=constraints)
    learned_weights = dict(zip(retained_models, [float(w) for w in np.round(res.x, 4)]))
    print(f"[ModelTrainer] ISO-Safety Optimized Meta Weights: {learned_weights}")

    oof_z_opt = np.dot(meta_X_oof, res.x)
    oof_y_opt = np.asarray(np.exp(oof_z_opt), dtype=float)

    # Compute GroupKFold train R2 for ISO-Optimized Ensemble
    opt_tr_r2_list = []
    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, z, groups=groups)):
        X_train, z_train, y_train = X.iloc[train_idx], z[train_idx], y[train_idx]
        sw_train = compute_tail_sample_weights(y_train)
        tr_z_mat = []
        for m_name in retained_models:
            m_inst = candidate_models[m_name]
            fit_model_with_sample_weights(m_inst, X_train, z_train, sw_train)
            tr_z_mat.append(np.asarray(m_inst.predict(X_train), dtype=float))
        tr_z_matrix = np.column_stack(tr_z_mat)
        res_fold = minimize(iso_safety_loss_function, init_w, args=(tr_z_matrix, y_train), method='SLSQP', bounds=bounds, constraints=constraints)
        tr_y_st = np.exp(np.dot(tr_z_matrix, res_fold.x))
        opt_tr_r2_list.append(r2_score(y_train, tr_y_st))

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
