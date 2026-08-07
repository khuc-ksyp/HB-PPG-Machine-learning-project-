"""
Feature Selection Engine for Non-Invasive Blood Glucose Estimator.
Performs:
  1. Pairwise Correlation Filtering (|r| > 0.85): Removes collinear features, keeping the feature with higher target correlation.
  2. RFECV / Feature Importance Ranking: Isolates the top core predictive features (15-35 features) using GroupKFold.
  3. Saves selected feature names to artifacts/selected_features.json.
"""

import json
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.feature_selection import RFECV
from sklearn.model_selection import GroupKFold

from src.config import ARTIFACTS_DIR, TARGET_COLUMN, GROUP_CV_COLUMN, RANDOM_SEED, N_CV_SPLITS


SELECTED_FEATURES_JSON_PATH = ARTIFACTS_DIR / "selected_features.json"


def filter_collinear_features(
    X: pd.DataFrame,
    y: np.ndarray,
    threshold: float = 0.85
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Step A: Pairwise Correlation Filter.
    Removes features with pairwise correlation |r| > threshold.
    For correlated pairs, retains the feature with higher absolute correlation to target y.
    """
    print(f"[FeatureSelection] Step A: Pairwise Correlation Filter (|r| > {threshold})...")
    
    # Calculate target correlation
    target_corrs = X.apply(lambda col: float(np.abs(np.corrcoef(col.to_numpy(), y)[0, 1]))).to_dict()

    corr_matrix = X.corr().abs()
    triu_mask = pd.DataFrame(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool), index=corr_matrix.index, columns=corr_matrix.columns)
    upper = corr_matrix.where(triu_mask)

    to_drop = set()
    for col in upper.columns:
        # Find features correlated higher than threshold
        high_corr_cols = upper.index[upper[col] > threshold].tolist()
        for c in high_corr_cols:
            # Compare correlation with target y
            if target_corrs.get(c, 0.0) >= target_corrs.get(col, 0.0):
                to_drop.add(col)
            else:
                to_drop.add(c)

    selected_cols = [col for col in X.columns if col not in to_drop]
    dropped_cols = list(to_drop)

    print(f"  - Initial features: {X.shape[1]}")
    print(f"  - Removed collinear features: {len(dropped_cols)}")
    print(f"  - Remaining features after Step A: {len(selected_cols)}")

    return X[selected_cols].copy(), selected_cols


def select_top_features_rfecv(
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    target_num_features: int = 14,
    n_splits: int = N_CV_SPLITS,
    seed: int = RANDOM_SEED
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Step B: RFECV & Feature Importance Ranking.
    Uses ExtraTrees / RandomForest regressor with GroupKFold cross-validation
    to isolate top target_num_features core predictive features (12-15 features).
    """
    print(f"[FeatureSelection] Step B: RFECV / Importance Selection (Target top {target_num_features} features)...")
    
    gkf = GroupKFold(n_splits=n_splits)
    estimator = ExtraTreesRegressor(n_estimators=100, max_depth=8, random_state=seed, n_jobs=-1)

    cv_splits = list(gkf.split(X, y, groups=groups))

    try:
        rfecv = RFECV(
            estimator=estimator,
            step=1,
            cv=cv_splits,
            scoring="neg_mean_absolute_error",
            min_features_to_select=min(target_num_features, X.shape[1]),
            n_jobs=-1,
        )
        rfecv.fit(X, y)

        if np.sum(rfecv.support_) > target_num_features:
            importances = estimator.fit(X.iloc[:, rfecv.support_], y).feature_importances_
            selected_indices = np.where(rfecv.support_)[0]
            top_order = np.argsort(importances)[::-1][:target_num_features]
            final_indices = selected_indices[top_order]
            selected_features = list(X.columns[final_indices.tolist()])
        else:
            selected_features = list(X.columns[rfecv.support_])
    except Exception as e:
        print(f"  [Warning] RFECV fallback to feature importances due to: {e}")
        estimator.fit(X, y)
        importances = estimator.feature_importances_
        top_idx = np.argsort(importances)[::-1][:min(target_num_features, X.shape[1])]
        selected_features = list(X.columns[top_idx.tolist()])

    if len(selected_features) < min(target_num_features, X.shape[1]):
        estimator.fit(X, y)
        importances = estimator.feature_importances_
        top_idx = np.argsort(importances)[::-1][:min(target_num_features, X.shape[1])]
        selected_features = list(X.columns[top_idx.tolist()])

    print(f"  - Final core features selected: {len(selected_features)}")
    
    # Save selected feature list to JSON artifact
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(SELECTED_FEATURES_JSON_PATH, "w") as f:
        json.dump(selected_features, f, indent=2)

    print(f"[FeatureSelection] Selected features saved to: {SELECTED_FEATURES_JSON_PATH.resolve()}")

    return X[selected_features].copy(), selected_features


from scipy.stats import spearmanr


def audit_and_filter_correlations(
    X: pd.DataFrame,
    y: np.ndarray,
    min_corr: float = 0.10,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Computes Spearman rank correlation (|r|) for all extracted PPG features vs target y across all N=217 subjects.
    Prints top 10 most correlated features and enforces a STRICT GATE (|r| >= 0.10).
    """
    print(f"[FeatureSelection] Step 0: Target Spearman Correlation Audit (Strict Gate |r| >= {min_corr})...")
    spearman_scores = {}
    raw_spearman = {}
    for col in X.columns:
        arr = np.asarray(X[col].values, dtype=float)
        try:
            res_s = spearmanr(arr, y)
            r_val = float(res_s.statistic) if hasattr(res_s, "statistic") else 0.0
        except Exception:
            r_val = 0.0
        raw_spearman[col] = r_val
        spearman_scores[col] = abs(r_val)

    # Sort and print Top 10 Most Correlated Features
    sorted_features = sorted(spearman_scores.keys(), key=lambda c: spearman_scores[c], reverse=True)
    print("--- TOP 10 SPEARMAN CORRELATED PPG & PHYSIOLOGICAL FEATURES ---")
    for rank, col in enumerate(sorted_features[:10], start=1):
        print(f"  {rank:2d}. {col:<35} | Spearman r = {raw_spearman[col]:+.4f} (|r| = {spearman_scores[col]:.4f})")
    print("----------------------------------------------------------------")

    # STRICT GATE: Drop all features where |r| < 0.10
    selected = [c for c in X.columns if spearman_scores[c] >= min_corr]
    print(f"  - Initial extracted features: {X.shape[1]}")
    print(f"  - Features passing strict Spearman correlation gate (|r| >= {min_corr}): {len(selected)}")

    if len(selected) < 5:
        print(f"[FeatureSelection] WARNING: Only {len(selected)} features passed strict correlation gate (|r| >= {min_corr}). PPG feature extraction needs Beer-Lambert log-ratio re-engineering!")
        if len(selected) == 0:
            selected = sorted_features[:5]

    return X[selected].copy(), selected


def run_feature_selection(
    df: pd.DataFrame,
    target_col: str = TARGET_COLUMN,
    group_col: str = GROUP_CV_COLUMN,
    corr_threshold: float = 0.85,
    target_num_features: int = 14,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Main entry point for feature reduction pipeline.
    Accepts full DataFrame, applies Step 0 (Correlation Audit), Step A (Correlation Filter) & Step B (RFECV/Importance Selection).
    Returns reduced feature DataFrame (including Target & Group columns) and list of selected feature names.
    """
    drop_cols = [target_col, group_col]
    feature_cols = [c for c in df.columns if c not in drop_cols]

    X = df[feature_cols].copy()
    y = np.asarray(df[target_col].values, dtype=float)
    groups = np.asarray(df[group_col].values)

    X_audit, _ = audit_and_filter_correlations(X, y, min_corr=0.10)
    X_filtered, _ = filter_collinear_features(X_audit, y, threshold=corr_threshold)
    X_selected, selected_features = select_top_features_rfecv(
        X_filtered, y, groups, target_num_features=target_num_features
    )

    reduced_df = pd.concat([X_selected, df[drop_cols]], axis=1)
    return reduced_df, selected_features


