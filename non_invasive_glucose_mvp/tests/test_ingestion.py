"""
Unit tests for data ingestion, metadata cleaning, missing value imputation, and signal loading.
"""

import numpy as np
import pandas as pd
import pytest

from src.data_ingestion import clean_and_impute_metadata, load_subject_signal
from src.config import TARGET_COLUMN, CHANNEL_NAMES, MMOL_TO_MGDL, DATA_CSV_DIR


def test_clean_and_impute_metadata():
    # Mock raw metadata with missing values, '/' strings, and varied cases
    raw_data = {
        "ID": [1, 2, 3, 4],
        "Gender": ["male", "female", "/", "male"],
        "Age (year)": [25, np.nan, 40, 50],
        "Height (cm)": [175, 160, "/", 180],
        "Weight (kg)": [70, 55, 65, np.nan],
        "Hemoglobin (g/L)": [140, np.nan, 150, 145],
        "Blood glucose (mmol/L)": [5.0, 6.0, "/", 7.0],
    }
    raw_df = pd.DataFrame(raw_data)

    clean_df = clean_and_impute_metadata(raw_df)

    # 1. Check row 3 (missing target '/') was dropped
    assert len(clean_df) == 3
    assert 3 not in clean_df["ID"].values

    # 2. Check glucose conversion
    expected_glucose_1 = 5.0 * MMOL_TO_MGDL
    assert np.isclose(clean_df.loc[clean_df["ID"] == 1, TARGET_COLUMN].values[0], expected_glucose_1)

    # 3. Check continuous imputation
    assert clean_df["Age"].isna().sum() == 0
    assert clean_df["Height"].isna().sum() == 0
    assert clean_df["Weight"].isna().sum() == 0
    assert clean_df["Hemoglobin"].isna().sum() == 0

    # 4. Check BMI column creation
    assert "BMI" in clean_df.columns


def test_load_subject_signal():
    # Test loading real subject CSV 1 if dataset present
    sig_df = load_subject_signal(1, DATA_CSV_DIR)
    if sig_df is not None:
        assert isinstance(sig_df, pd.DataFrame)
        assert list(sig_df.columns) == CHANNEL_NAMES
        assert len(sig_df) > 1000
