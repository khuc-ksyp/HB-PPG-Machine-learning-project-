"""
Subject-Wise Feature Standardization & Delta Transformation Engine.
Converts raw physiological features into Subject-Normalized Delta Features:
  X_delta = X_i - Mean(X_subject)
  X_ratio = X_i / Mean(X_subject)
Retains demographics (Age, Gender, Height, Weight, Hemoglobin, BMI) in raw form while
transforming optical AC/DC, THAI index, derivative, and ratio features into subject-centered delta space.
"""

from typing import List, Tuple
import numpy as np
import pandas as pd

from src.config import TARGET_COLUMN, GROUP_CV_COLUMN

DEMOGRAPHIC_COLUMNS = ["ID", "Age", "Gender_male", "Height", "Weight", "Hemoglobin", "BMI", TARGET_COLUMN]


def robust_z_scale(series: pd.Series) -> pd.Series:
    """
    Computes Robust Median / IQR Standardization:
      x_norm = (x - median(x_subject)) / (iqr(x_subject) + 1e-6)
    """
    med = series.median()
    q75 = series.quantile(0.75)
    q25 = series.quantile(0.25)
    iqr = q75 - q25
    if iqr < 1e-8:
        return series - med
    return (series - med) / (iqr + 1e-6)


def transform_subject_delta_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Transforms physiological, optical, derivative, and morphological features in df
    into Subject-Level Robust Z-Score Normalized Features.

    Demographic factors (Age, Gender_male, Height, Weight, Hemoglobin, BMI) are retained in raw form.

    Returns:
        transformed_df: DataFrame with demographic features + transformed robust scaled features.
        feature_names: List of all predictor feature column names (excluding ID & Target).
    """
    transformed_df = df.copy()

    phys_cols = [col for col in df.columns if col not in DEMOGRAPHIC_COLUMNS]

    # Calculate Subject-Wise or Group-Wise baseline metrics using Robust Median/IQR scaling
    if GROUP_CV_COLUMN in df.columns and df[GROUP_CV_COLUMN].nunique() > 1:
        # Check if multiple window observations exist per subject
        subject_counts = df[GROUP_CV_COLUMN].value_counts()
        if subject_counts.max() > 1:
            # Multi-window per subject ID
            for col in phys_cols:
                transformed_df[col] = df.groupby(GROUP_CV_COLUMN)[col].transform(robust_z_scale)
        else:
            # Single observation per subject: Center relative to demographic cohort baseline (Age decile & Gender)
            if "Age" in df.columns and "Gender_male" in df.columns:
                age_group = pd.qcut(df["Age"], q=min(5, df["Age"].nunique()), labels=False, duplicates="drop")
                for col in phys_cols:
                    transformed_df[col] = df.groupby([age_group, df["Gender_male"]])[col].transform(robust_z_scale)
            else:
                for col in phys_cols:
                    transformed_df[col] = robust_z_scale(df[col])

    feature_names = [col for col in transformed_df.columns if col not in [TARGET_COLUMN, GROUP_CV_COLUMN]]
    return transformed_df, feature_names
