"""
Main Command Line Interface (CLI) Pipeline for Non-Invasive Blood Glucose Estimator.
Orchestrates: Ingestion -> Preprocessing & Feature Extraction -> Model Training -> Clinical Evaluation -> SHAP Explainability -> 1-Point Calibration.
"""

import sys
import time
from pathlib import Path

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    METADATA_PATH,
    DATA_CSV_DIR,
    CLEANED_FEATURES_PATH,
    MODEL_SAVE_PATH,
    CLARKE_GRID_PLOT_PATH,
    ARTIFACTS_DIR,
    TARGET_COLUMN,
    GROUP_CV_COLUMN,
)
from src.data_ingestion import ingest_dataset
from src.feature_extraction import build_feature_dataset
from src.model_trainer import (
    train_and_benchmark_models,
    save_model,
)
from src.clinical_evaluator import (
    evaluate_clarke_error_grid,
    plot_clarke_error_grid,
)
from src.explainability import generate_shap_analysis
from src.calibration import evaluate_calibrated_performance


def run_pipeline():
    start_time = time.time()
    print("================================================================================")
    print("    NON-INVASIVE CONTINUOUS BLOOD GLUCOSE ESTIMATION PIPELINE")
    print("    Multi-Wavelength Photoplethysmogram (PPG) Analysis (660/730/850/940 nm)")
    print("================================================================================\n")

    # STEP 1: DATA INGESTION & CLEANING
    print(">>> STEP 1: Metadata Ingestion & Missing Value Cleaning...")
    metadata_df, signals_dict = ingest_dataset(METADATA_PATH, DATA_CSV_DIR)
    print(f"    - Metadata subjects ingested: {len(metadata_df)}")
    print(f"    - Synchronized PPG signal files matched: {len(signals_dict)}\n")

    # STEP 2 & 3: SIGNAL PROCESSING & FEATURE EXTRACTION
    print(">>> STEP 2 & 3: Signal Preprocessing & Feature Extraction Engine...")
    features_df = build_feature_dataset(
        metadata_df=metadata_df,
        signals_dict=signals_dict,
        save_csv=True,
        output_path=CLEANED_FEATURES_PATH,
    )
    print(f"    - Extracted Feature Matrix shape: {features_df.shape}\n")

    # STEP 4: MODEL TRAINING & SELECTION
    print(">>> STEP 4: Machine Learning Model Benchmarking & GroupKFold Tuning...")
    benchmark_results, best_model, y_true, y_pred = train_and_benchmark_models(
        df=features_df,
        target_col=TARGET_COLUMN,
        group_col=GROUP_CV_COLUMN,
    )
    save_model(best_model, MODEL_SAVE_PATH)
    print()

    # STEP 5: CLINICAL EVALUATION & CLARKE ERROR GRID
    print(">>> STEP 5: Clinical Safety Evaluation & Clarke Error Grid Generation...")
    zone_percentages, _ = evaluate_clarke_error_grid(y_true, y_pred)
    plot_path = plot_clarke_error_grid(y_true, y_pred, save_path=CLARKE_GRID_PLOT_PATH)
    print()

    # STEP 6: SHAP EXPLAINABILITY MODULE
    print(">>> STEP 6: SHAP Interpretability & Feature Importance Analysis...")
    drop_cols = [TARGET_COLUMN, GROUP_CV_COLUMN]
    feature_cols = [c for c in features_df.columns if c not in drop_cols]
    X_matrix = features_df[feature_cols]

    _, top_features_df = generate_shap_analysis(
        model=best_model,
        X_train=X_matrix,
        X_test=X_matrix,
        feature_names=feature_cols,
        output_dir=ARTIFACTS_DIR,
    )

    # STEP 7: 1-POINT PERSONALIZATION CALIBRATION
    print(">>> STEP 7: 1-Point Personalization Calibration Impact Evaluation...")
    uncal_metrics, cal_metrics = evaluate_calibrated_performance(y_true, y_pred, calibration_index=0)

    # SUMMARY CONSOLE REPORT
    elapsed = time.time() - start_time
    tuned_key = [k for k in benchmark_results.keys() if k.startswith("Tuned_")][0]
    best_metrics = benchmark_results[tuned_key]

    print("================================================================================")
    print("                         PIPELINE EXECUTION SUMMARY                             ")
    print("================================================================================")
    print(f"Total Execution Time: {elapsed:.2f} seconds")
    print(f"Cleaned Feature File: {CLEANED_FEATURES_PATH.resolve()}")
    print(f"Trained Model Binary: {MODEL_SAVE_PATH.resolve()}")
    print(f"Clarke Grid Image:    {plot_path.resolve()}")
    print(f"SHAP Summary Image:   {(ARTIFACTS_DIR / 'shap_summary.png').resolve()}")
    print("--------------------------------------------------------------------------------")
    print(f"Model Performance Metrics ({tuned_key} - GroupKFold CV):")
    print(f"  - Mean Absolute Error (MAE):     {best_metrics['MAE']:.2f} mg/dL")
    print(f"  - Root Mean Squared Error (RMSE): {best_metrics['RMSE']:.2f} mg/dL")
    print(f"  - Baseline MARD (Uncalibrated):   {best_metrics['MARD']:.2f}%")
    print(f"  - Calibrated MARD (1-Point Cal):  {cal_metrics['MARD']:.2f}%")
    print("--------------------------------------------------------------------------------")
    print("Clarke Error Grid Clinical Safety Breakdown:")
    print(f"  - Zone A (Clinically Accurate): {zone_percentages['A']:.2f}%")
    print(f"  - Zone B (Benign Errors):       {zone_percentages['B']:.2f}%")
    print(f"  - Zones A + B Combined:         {zone_percentages['A_B_Combined']:.2f}%")
    print(f"  - Calibrated Zones A + B:        {cal_metrics['Zone_A_B_Combined']:.2f}%")
    print("================================================================================")


if __name__ == "__main__":
    run_pipeline()
