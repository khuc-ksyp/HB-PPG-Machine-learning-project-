"""
System configuration constants, signal processing parameters, and file path definitions.
"""

import os
from pathlib import Path

# Base Directory Resolution
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Dataset Paths
# Check root or parent for Hb_PPG_Dataset
DATASET_DIR = PROJECT_ROOT / "Hb_PPG_Dataset"
if not DATASET_DIR.exists() and (PROJECT_ROOT.parent / "Hb_PPG_Dataset").exists():
    DATASET_DIR = PROJECT_ROOT.parent / "Hb_PPG_Dataset"

METADATA_PATH = DATASET_DIR / "subject information.xlsx"
DATA_CSV_DIR = DATASET_DIR / "data_csv"

# Artifact Output Paths
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
CLEANED_FEATURES_PATH = ARTIFACTS_DIR / "cleaned_features.csv"
MODEL_SAVE_PATH = ARTIFACTS_DIR / "glucose_model.pkl"
CLARKE_GRID_PLOT_PATH = ARTIFACTS_DIR / "clarke_error_grid.png"

# Signal Processing Parameters
SAMPLING_FREQ = 200.0  # Hz
BANDPASS_LOW = 0.5     # Hz
BANDPASS_HIGH = 4.0    # Hz
FILTER_ORDER = 2

# Optical Wavelength Channels (nm)
WAVELENGTHS = [660, 730, 850, 940]
CHANNEL_NAMES = [f"{wl}nm" for wl in WAVELENGTHS]

# Unit Conversion Constants
MMOL_TO_MGDL = 18.0182  # Blood Glucose conversion factor mmol/L -> mg/dL

# Machine Learning & Cross Validation
RANDOM_SEED = 42
N_CV_SPLITS = 5
GROUP_CV_COLUMN = "ID"
TARGET_COLUMN = "Blood_Glucose_mg_dL"
