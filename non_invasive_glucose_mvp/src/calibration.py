"""
1-Point Personalization Calibration Module.
Applies single-point reference blood glucose calibration to adjust baseline optical offset,
substantially improving subject-level prediction accuracy (MARD) and Clarke Error Grid clinical safety.
"""

from typing import Tuple, Dict
import numpy as np

from src.clinical_evaluator import evaluate_clarke_error_grid
from src.model_trainer import calculate_mard, evaluate_predictions


def apply_one_point_calibration(
    y_pred: np.ndarray,
    y_true_reference: float,
    index: int = 0
) -> np.ndarray:
    """
    Applies 1-point personalization calibration to predicted glucose values.

    Args:
        y_pred: Array of model predicted glucose values (mg/dL).
        y_true_reference: Single reference invasive measurement value (mg/dL).
        index: Index within y_pred corresponding to the reference measurement timepoint.

    Returns:
        y_calibrated: Offset-adjusted glucose predictions (mg/dL).
    """
    y_pred_arr = np.asarray(y_pred, dtype=float)
    if index < 0 or index >= len(y_pred_arr):
        index = 0
    
    offset = float(y_true_reference - y_pred_arr[index])
    y_calibrated = y_pred_arr + offset
    return y_calibrated


def evaluate_calibrated_performance(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    calibration_index: int = 0
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Evaluates system metrics (MAE, RMSE, MARD, Clarke Error Grid) before and after
    1-point personalization calibration to quantify accuracy gains.

    Returns:
        uncalibrated_metrics: Performance dictionary before calibration.
        calibrated_metrics: Performance dictionary after calibration.
    """
    # Baseline Uncalibrated
    uncal_metrics = evaluate_predictions(y_true, y_pred)
    uncal_ceg, _ = evaluate_clarke_error_grid(y_true, y_pred)
    uncal_metrics["Zone_A"] = uncal_ceg["A"]
    uncal_metrics["Zone_B"] = uncal_ceg["B"]
    uncal_metrics["Zone_A_B_Combined"] = uncal_ceg["A_B_Combined"]

    # Apply 1-Point Calibration using reference sample
    ref_val = float(y_true[calibration_index])
    y_cal = apply_one_point_calibration(y_pred, ref_val, index=calibration_index)

    cal_metrics = evaluate_predictions(y_true, y_cal)
    cal_ceg, _ = evaluate_clarke_error_grid(y_true, y_cal)
    cal_metrics["Zone_A"] = cal_ceg["A"]
    cal_metrics["Zone_B"] = cal_ceg["B"]
    cal_metrics["Zone_A_B_Combined"] = cal_ceg["A_B_Combined"]

    print("--- 1-POINT PERSONALIZATION CALIBRATION PERFORMANCE EVALUATION ---")
    print(f"  Uncalibrated -> MARD: {uncal_metrics['MARD']:.2f}% | MAE: {uncal_metrics['MAE']:.2f} mg/dL | Zones A+B: {uncal_metrics['Zone_A_B_Combined']:.2f}%")
    print(f"  Calibrated   -> MARD: {cal_metrics['MARD']:.2f}% | MAE: {cal_metrics['MAE']:.2f} mg/dL | Zones A+B: {cal_metrics['Zone_A_B_Combined']:.2f}%")
    print("---------------------------------------------------------------------\n")

    return uncal_metrics, cal_metrics
