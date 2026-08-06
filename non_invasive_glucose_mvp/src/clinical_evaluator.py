"""
Clinical Evaluation Engine & Clarke Error Grid (CEG) Analysis.
Calculates clinical accuracy zone distribution (A, B, C, D, E) and renders
production-quality Clarke Error Grid plots for glucose predictions.
"""

from pathlib import Path
from typing import Dict, Tuple, List
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import CLARKE_GRID_PLOT_PATH


def assign_clarke_zone(y_ref: float, y_pred: float) -> str:
    """
    Evaluates a single reference and predicted blood glucose pair (in mg/dL)
    and assigns it to one of the 5 Clarke Error Grid zones (A, B, C, D, E).

    Zone Definitions:
      - Zone A: Clinically accurate (within 20% or <70 mg/dL when ref <70).
      - Zone B: Benign errors (outside 20% but no inappropriate treatment).
      - Zone C: Overcorrection errors (leads to unnecessary treatment).
      - Zone D: Failure to detect hypoglycemia or hyperglycemia.
      - Zone E: Erroneous treatment (confuses hypo with hyper or vice versa).
    """
    # Zone A
    if abs(y_pred - y_ref) <= 0.20 * y_ref or (y_ref <= 70 and y_pred <= 70):
        return "A"

    # Zone E
    if (y_ref <= 70 and y_pred >= 180) or (y_ref >= 180 and y_pred <= 70):
        return "E"

    # Zone D
    if (y_ref <= 70 and 70 < y_pred < 180) or (y_ref >= 240 and 70 < y_pred < 180):
        return "D"

    # Zone C
    if (70 < y_ref < 290 and y_pred > y_ref + 110) or (130 < y_ref < 180 and y_pred < (7.0 / 5.0) * y_ref - 182):
        return "C"

    # Zone B
    return "B"


def evaluate_clarke_error_grid(
    y_true: np.ndarray, y_pred: np.ndarray
) -> Tuple[Dict[str, float], List[str]]:
    """
    Computes zone breakdown counts and percentages for Clarke Error Grid analysis.
    """
    zones = [assign_clarke_zone(ref, pred) for ref, pred in zip(y_true, y_pred)]
    total = len(zones)

    zone_counts = {z: zones.count(z) for z in ["A", "B", "C", "D", "E"]}
    zone_percentages = {z: (cnt / total) * 100.0 for z, cnt in zone_counts.items()}

    zone_percentages["A_B_Combined"] = zone_percentages["A"] + zone_percentages["B"]
    return zone_percentages, zones


def plot_clarke_error_grid(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: Path = CLARKE_GRID_PLOT_PATH,
    title: str = "Clarke Error Grid Analysis - Non-Invasive Glucose Estimator",
) -> Path:
    """
    Renders and saves a high-resolution Clarke Error Grid plot with reference lines,
    zone labels, and prediction scatter points.
    """
    percentages, zones = evaluate_clarke_error_grid(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(9, 9), dpi=300)

    # Upper bound of plot (mg/dL)
    max_val = max(float(np.max(y_true)), float(np.max(y_pred)), 400.0) + 20.0

    # Grid background & Axis bounds
    ax.set_xlim([0, max_val])
    ax.set_ylim([0, max_val])
    ax.set_xlabel("Reference Blood Glucose (mg/dL)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Estimated Blood Glucose (mg/dL)", fontsize=12, fontweight="bold")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)

    # 1. Ideal line y = x
    ax.plot([0, max_val], [0, max_val], color="black", linestyle="--", linewidth=1.5, label="y = x (Perfect Fit)")

    # 2. Zone A limits: y = 1.2x, y = 0.8x
    ax.plot([0, max_val], [0, 1.2 * max_val], color="gray", linestyle=":", linewidth=1.2)
    ax.plot([0, max_val], [0, 0.8 * max_val], color="gray", linestyle=":", linewidth=1.2)

    # 3. Zone boundaries for C, D, E
    ax.plot([70, 70], [84, max_val], color="black", linestyle="-", linewidth=1.0)
    ax.plot([0, 70], [70, 70], color="black", linestyle="-", linewidth=1.0)
    ax.plot([70, 290], [180, max_val], color="black", linestyle="-", linewidth=1.0)

    ax.plot([180, max_val], [70, 70], color="black", linestyle="-", linewidth=1.0)
    ax.plot([180, 180], [0, 70], color="black", linestyle="-", linewidth=1.0)
    ax.plot([240, 240], [70, 180], color="black", linestyle="-", linewidth=1.0)
    ax.plot([130, 180], [0, 70], color="black", linestyle="-", linewidth=1.0)

    # Add Zone Text Annotations
    ax.text(30, 15, "A", fontsize=15, fontweight="bold", color="green")
    ax.text(150, 260, "B", fontsize=15, fontweight="bold", color="darkorange")
    ax.text(280, 120, "B", fontsize=15, fontweight="bold", color="darkorange")
    ax.text(160, 370, "C", fontsize=15, fontweight="bold", color="red")
    ax.text(160, 20, "C", fontsize=15, fontweight="bold", color="red")
    ax.text(30, 130, "D", fontsize=15, fontweight="bold", color="purple")
    ax.text(330, 130, "D", fontsize=15, fontweight="bold", color="purple")
    ax.text(30, 320, "E", fontsize=15, fontweight="bold", color="darkred")
    ax.text(330, 20, "E", fontsize=15, fontweight="bold", color="darkred")

    # Scatter plot predictions by zone color
    color_map = {"A": "#2ca02c", "B": "#ff7f0e", "C": "#d62728", "D": "#9467bd", "E": "#8c564b"}

    zone_arr = np.array(zones)
    for z in ["A", "B", "C", "D", "E"]:
        mask = zone_arr == z
        if np.any(mask):
            ax.scatter(
                y_true[mask],
                y_pred[mask],
                c=color_map[z],
                label=f"Zone {z} ({percentages[z]:.1f}%)",
                alpha=0.75,
                edgecolors="k",
                linewidths=0.5,
                s=45,
            )

    # Statistics Text Box
    stats_text = (
        f"Zone A: {percentages['A']:.1f}%\n"
        f"Zone B: {percentages['B']:.1f}%\n"
        f"Zones A+B: {percentages['A_B_Combined']:.1f}%\n"
        f"Zone C: {percentages['C']:.1f}%\n"
        f"Zone D: {percentages['D']:.1f}%\n"
        f"Zone E: {percentages['E']:.1f}%"
    )

    box_props = dict(boxstyle="round,pad=0.5", facecolor="wheat", alpha=0.8)
    ax.text(
        0.05,
        0.95,
        stats_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=box_props,
    )

    ax.legend(loc="lower right", frameon=True, facecolor="white")
    ax.grid(True, linestyle="--", alpha=0.3)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"[ClinicalEvaluator] Clarke Error Grid plot saved to: {save_path}")
    print(
        f"[ClinicalEvaluator] Clinical Safety Check: Zone A+B = {percentages['A_B_Combined']:.2f}% "
        f"({'PASSED (>95%)' if percentages['A_B_Combined'] >= 95.0 else 'FAILED (<95%)'})"
    )

    return save_path
