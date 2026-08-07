"""
Main Command Line Interface (CLI) Pipeline for Non-Invasive Blood Glucose Estimator.
Orchestrates: Ingestion -> Preprocessing & Feature Extraction -> Model Training -> Clinical Evaluation -> SHAP Explainability.
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
    plot_actual_vs_predicted_scatter,
    compute_prediction_slope_and_variance,
)
from src.explainability import generate_shap_analysis
from src.feature_selection import run_feature_selection


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
    print(f"    - Full Extracted Feature Matrix shape: {features_df.shape}\n")

    # STEP 4: FEATURE SELECTION (CORRELATION & RFECV 14 FEATURES)
    print(">>> STEP 4: Feature Reduction Pipeline (|r| > 0.85 & 14-Feature RFECV)...")
    reduced_df, selected_features = run_feature_selection(
        df=features_df,
        target_col=TARGET_COLUMN,
        group_col=GROUP_CV_COLUMN,
        corr_threshold=0.85,
        target_num_features=14,
    )
    print(f"    - Selected Feature Matrix shape: {reduced_df.shape}\n")

    # STEP 5: REGULARIZED LINEAR SUITE & BLENDED ENSEMBLE RETRAINING
    print(">>> STEP 5: Regularized Linear/SVR Suite Benchmarking & Blended Ensemble Retraining...")
    benchmark_results, ensemble_model, y_true, y_pred = train_and_benchmark_models(
        df=reduced_df,
        target_col=TARGET_COLUMN,
        group_col=GROUP_CV_COLUMN,
    )
    save_model(ensemble_model, MODEL_SAVE_PATH)
    print()

    # STEP 6: CLINICAL EVALUATION & CLARKE ERROR GRID
    print(">>> STEP 6: Clinical Safety Evaluation, Slope Audit & Scatter Plot Generation...")
    zone_percentages, _ = evaluate_clarke_error_grid(y_true, y_pred)
    plot_path = plot_clarke_error_grid(y_true, y_pred, save_path=CLARKE_GRID_PLOT_PATH)
    scatter_path = plot_actual_vs_predicted_scatter(y_true, y_pred, save_path=ARTIFACTS_DIR / "actual_vs_predicted_scatter.png")
    slope_metrics = compute_prediction_slope_and_variance(y_true, y_pred)
    print()

    # STEP 7: SHAP EXPLAINABILITY MODULE
    print(">>> STEP 7: SHAP Interpretability & Feature Importance Analysis...")
    drop_cols = [TARGET_COLUMN, GROUP_CV_COLUMN]
    feature_cols = [c for c in reduced_df.columns if c not in drop_cols]
    X_matrix = reduced_df[feature_cols]

    # For SHAP analysis, use underlying ridge model inside ensemble
    shap_model = ensemble_model.models_dict["Ridge"] if hasattr(ensemble_model, "models_dict") and "Ridge" in ensemble_model.models_dict else ensemble_model
    _, top_features_df = generate_shap_analysis(
        model=shap_model,
        X_train=X_matrix,
        X_test=X_matrix,
        feature_names=feature_cols,
        output_dir=ARTIFACTS_DIR,
    )

    # SUMMARY CONSOLE REPORT
    elapsed = time.time() - start_time

    print("=========================================================================================================================")
    print("                                      GROUPKFOLD MODEL COMPARISON & ENSEMBLE REPORT                                      ")
    print("=========================================================================================================================")
    print(f"{'Model':<18} | {'Train R2':<8} | {'Test R2':<8} | {'Gap':<6} | {'MAE':<6} | {'RMSE':<6} | {'MARD (%)':<8} | {'ISO 15197 %':<11} | {'Zone A %':<8}")
    print("-------------------------------------------------------------------------------------------------------------------------")

    for m_name, m_metrics in benchmark_results.items():
        tr_r2 = m_metrics.get("Train_R2", 0.0)
        te_r2 = m_metrics.get("Test_OOF_R2", 0.0)
        gap = m_metrics.get("Gap", tr_r2 - te_r2)
        mae = m_metrics.get("MAE", 0.0)
        rmse = m_metrics.get("RMSE", 0.0)
        mard = m_metrics.get("MARD", 0.0)
        iso = m_metrics.get("ISO_15197_Compliance", 0.0)
        zone_a = m_metrics.get("Clarke_Zone_A", 0.0)

        print(f"{m_name:<18} | {tr_r2:<8.4f} | {te_r2:<8.4f} | {gap:<6.4f} | {mae:<6.2f} | {rmse:<6.2f} | {mard:<8.2f} | {iso:<11.2f} | {zone_a:<8.2f}")

    print("=========================================================================================================================")
    print(f"Total Execution Time: {elapsed:.2f} seconds")
    print(f"Cleaned Feature File: {CLEANED_FEATURES_PATH.resolve()}")
    print(f"Trained Model Binary: {MODEL_SAVE_PATH.resolve()}")
    print(f"Clarke Grid Image:    {plot_path.resolve()}")
    print(f"Scatter Plot Image:   {scatter_path.resolve()}")
    print(f"SHAP Summary Image:   {(ARTIFACTS_DIR / 'shap_summary.png').resolve()}")

    print("-------------------------------------------------------------------------------------------------------------------------")
    print("PREDICTION VARIANCE & TRENDLINE SLOPE VALIDATION:")
    print(f"  - Fitted Linear Slope (m):             {slope_metrics['slope']:.4f}")
    print(f"  - Standard Deviation Ratio (std_pred/std_true): {slope_metrics['std_ratio']:.4f}")
    print(f"  - Reference Glucose Std Dev (std_true):  {slope_metrics['std_true']:.2f} mg/dL")
    print(f"  - Predicted Glucose Std Dev (std_pred):  {slope_metrics['std_pred']:.2f} mg/dL")
    print("-------------------------------------------------------------------------------------------------------------------------")

    ens_metrics = benchmark_results.get("MARD_Opt_Ensemble", {})
    ens_iso = ens_metrics.get("ISO_15197_Compliance", 0.0)
    ens_mard = ens_metrics.get("MARD", 0.0)
    ens_mae = ens_metrics.get("MAE", 0.0)
    ens_ab = ens_metrics.get("Clarke_Zone_A_B", 0.0)
    iso_pass = "PASSED (>= 95.0%)" if ens_iso >= 95.0 else f"FALLS SHORT of 95.0% target (Achieved {ens_iso:.2f}%)"

    print("ISO 15197:2013 POINT-WISE CLINICAL COMPLIANCE EVALUATION:")
    print(f"  - MARD-Optimized Ensemble Compliance: {ens_iso:.2f}% | Status: {iso_pass}")
    print(f"  - Clinical Metrics Summary: MARD = {ens_mard:.2f}%, MAE = {ens_mae:.2f} mg/dL, Clarke Zone A+B = {ens_ab:.2f}%")
    print("=========================================================================================================================")


if __name__ == "__main__":
    run_pipeline()
