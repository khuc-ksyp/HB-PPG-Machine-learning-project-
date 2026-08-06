"""
Data Ingestion Module.
Handles metadata extraction from Excel, targeted missing value imputation,
column pruning, signal file alignment, and signal loading.
"""

from pathlib import Path
from typing import Dict, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer

from src.config import (
    METADATA_PATH,
    DATA_CSV_DIR,
    MMOL_TO_MGDL,
    TARGET_COLUMN,
    CHANNEL_NAMES,
    SAMPLING_FREQ,
)


def load_metadata_excel(path: Path = METADATA_PATH) -> pd.DataFrame:
    """
    Reads the subject information Excel file and returns the raw metadata DataFrame.
    """
    if not path.exists():
        raise FileNotFoundError(f"Metadata Excel file not found at: {path}")

    raw_df = pd.read_excel(path)
    return raw_df


def clean_and_impute_metadata(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies specific column pruning, coercion, target conversion, and imputation strategies:
      - Coerces missing placeholders (e.g., '/') to NaN.
      - Drops rows with missing target blood glucose.
      - Converts blood glucose from mmol/L to mg/dL.
      - Imputes Age, Height, Weight using Median grouped by Gender.
      - Imputes Gender using Mode if missing.
      - Imputes Hemoglobin (Hb) using KNN Imputer (k=5) based on correlated physiological features.
      - Drops irrelevant administrative columns.
    """
    df = raw_df.copy()

    # Column Mapping
    column_rename = {
        "ID": "ID",
        "Gender": "Gender",
        "Age (year)": "Age",
        "Height (cm)": "Height",
        "Weight (kg)": "Weight",
        "Hemoglobin (g/L)": "Hemoglobin",
        "Blood glucose (mmol/L)": "Glucose_mmol_L",
    }

    # Identify essential columns
    essential_cols = [c for c in column_rename.keys() if c in df.columns]
    df = df[essential_cols].rename(columns=column_rename)

    # 1. Clean Gender
    if "Gender" in df.columns:
        df["Gender"] = df["Gender"].astype(str).str.strip().str.lower()
        df["Gender"] = df["Gender"].replace({'nan': np.nan, 'none': np.nan, '/': np.nan, '': np.nan})
        if df["Gender"].isna().sum() > 0:
            gender_mode = df["Gender"].mode()[0]
            df["Gender"] = df["Gender"].fillna(gender_mode)

    # 2. Coerce numeric columns (handles '/' or text strings as NaN)
    numeric_cols = ["ID", "Age", "Height", "Weight", "Hemoglobin", "Glucose_mmol_L"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # 3. Target Handling: Drop rows missing Blood Glucose (NEVER impute ground truth target)
    df = df.dropna(subset=["Glucose_mmol_L"]).reset_index(drop=True)

    # Convert Glucose from mmol/L to mg/dL
    df[TARGET_COLUMN] = df["Glucose_mmol_L"] * MMOL_TO_MGDL
    df = df.drop(columns=["Glucose_mmol_L"])

    # 4. Continuous Demographics Imputation (Age, Height, Weight) via Median by Gender
    demo_cols = ["Age", "Height", "Weight"]
    for col in demo_cols:
        if col in df.columns and df[col].isna().sum() > 0:
            df[col] = df.groupby("Gender")[col].transform(lambda x: x.fillna(x.median()))
            # Fallback overall median if group is all NaN
            if df[col].isna().sum() > 0:
                df[col] = df[col].fillna(df[col].median())

    # 5. Hemoglobin (Hb) Imputation using KNN Imputer (k=5)
    if "Hemoglobin" in df.columns and df["Hemoglobin"].isna().sum() > 0:
        knn_features = ["Age", "Height", "Weight", "Hemoglobin"]
        gender_dummy = pd.get_dummies(df["Gender"], drop_first=True)
        knn_df = pd.concat([df[knn_features], gender_dummy], axis=1)

        imputer = KNNImputer(n_neighbors=5)
        imputed_matrix = imputer.fit_transform(knn_df)
        df["Hemoglobin"] = imputed_matrix[:, 3]

    # Calculate BMI as engineered demographic feature
    if "Height" in df.columns and "Weight" in df.columns:
        height_m = df["Height"] / 100.0
        df["BMI"] = df["Weight"] / (height_m ** 2)

    df["ID"] = df["ID"].astype(int)
    return df


def load_subject_signal(subject_id: int, csv_dir: Path = DATA_CSV_DIR) -> Optional[pd.DataFrame]:
    """
    Loads and validates the 4-wavelength PPG signal CSV for a given Subject ID.
    Returns DataFrame with columns ['660nm', '730nm', '850nm', '940nm'] or None if corrupted/missing.
    """
    file_path = csv_dir / f"{subject_id}.csv"
    if not file_path.exists():
        return None

    try:
        signal_df = pd.read_csv(file_path)
        # Verify required channels exist
        if not all(col in signal_df.columns for col in CHANNEL_NAMES):
            # If CSV has no headers, fallback to column indexing
            if signal_df.shape[1] >= 4:
                signal_df = signal_df.iloc[:, :4]
                signal_df.columns = CHANNEL_NAMES
            else:
                return None
        
        # Select target channels
        signal_df = signal_df[CHANNEL_NAMES].apply(pd.to_numeric, errors="coerce")
        
        # Check for NaN values or empty signal
        if signal_df.isna().sum().sum() > 0:
            signal_df = signal_df.ffill().bfill()
        
        # Signal length validation: minimum 5 seconds of data (1000 samples)
        if len(signal_df) < int(5 * SAMPLING_FREQ):
            return None

        return signal_df
    except Exception:
        return None


def ingest_dataset(
    metadata_path: Path = METADATA_PATH,
    csv_dir: Path = DATA_CSV_DIR
) -> Tuple[pd.DataFrame, Dict[int, pd.DataFrame]]:
    """
    Main ingestion interface. Loads metadata, cleans/imputes attributes,
    matches with CSV signals, and filters out subjects with missing or corrupted signals.

    Returns:
        valid_metadata: Cleaned metadata DataFrame for valid subjects.
        signals_dict: Dictionary mapping Subject ID -> 4-channel signal DataFrame.
    """
    raw_metadata = load_metadata_excel(metadata_path)
    clean_meta = clean_and_impute_metadata(raw_metadata)

    valid_signals = {}
    valid_subject_ids = []

    for _, row in clean_meta.iterrows():
        sub_id = int(row["ID"])
        sig_df = load_subject_signal(sub_id, csv_dir)
        if sig_df is not None:
            valid_signals[sub_id] = sig_df
            valid_subject_ids.append(sub_id)

    valid_metadata = clean_meta[clean_meta["ID"].isin(valid_subject_ids)].reset_index(drop=True)
    return valid_metadata, valid_signals
