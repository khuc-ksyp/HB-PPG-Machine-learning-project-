"""
Unit tests for model training, GroupKFold validation, metrics calculation, and Clarke Error Grid evaluation.
"""

import numpy as np
import pandas as pd
import pytest

from src.model_trainer import calculate_mard, train_and_benchmark_models
from src.clinical_evaluator import assign_clarke_zone, evaluate_clarke_error_grid
from src.config import TARGET_COLUMN, GROUP_CV_COLUMN


def test_calculate_mard():
    y_true = np.array([100.0, 200.0, 150.0])
    y_pred = np.array([110.0, 180.0, 150.0])
    # Relative diffs: |10/100| = 0.10, |-20/200| = 0.10, |0/150| = 0.0 -> Mean = 0.06667 -> 6.67%
    mard = calculate_mard(y_true, y_pred)
    assert np.isclose(mard, 6.6666666, atol=1e-3)


def test_assign_clarke_zone():
    # Zone A: Perfect fit or within 20%
    assert assign_clarke_zone(100.0, 105.0) == "A"
    assert assign_clarke_zone(60.0, 65.0) == "A"

    # Zone E: Severe opposite misclassification
    assert assign_clarke_zone(60.0, 200.0) == "E"
    assert assign_clarke_zone(200.0, 60.0) == "E"


def test_group_kfold_training_pipeline():
    # Mock feature dataset with 10 distinct subjects (2 rows each)
    np.random.seed(42)
    n_samples = 20
    mock_df = pd.DataFrame({
        GROUP_CV_COLUMN: np.repeat(np.arange(1, 11), 2),
        "Age": np.random.uniform(20, 60, n_samples),
        "BMI": np.random.uniform(18, 30, n_samples),
        "R_660_940": np.random.uniform(0.5, 1.5, n_samples),
        "660nm_mean": np.random.uniform(-1, 1, n_samples),
        TARGET_COLUMN: np.random.uniform(80, 180, n_samples),
    })

    results, best_model, y_true, y_pred = train_and_benchmark_models(
        mock_df,
        target_col=TARGET_COLUMN,
        group_col=GROUP_CV_COLUMN,
        n_splits=3,
    )

    assert "Tuned_XGBoost" in results
    assert len(y_pred) == n_samples
    assert hasattr(best_model, "predict")
