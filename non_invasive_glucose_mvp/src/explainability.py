"""
SHAP Explainability Module for Optical Blood Glucose Estimation Model.
Provides model interpretability using SHAP values, generates summary plots,
and identifies top physiological and optical features driving predictions.
"""

from pathlib import Path
from typing import List, Tuple, Optional, Any
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import shap

from src.config import ARTIFACTS_DIR


def generate_shap_analysis(
    model: Any,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    feature_names: Optional[List[str]] = None,
    output_dir: Path = ARTIFACTS_DIR,
) -> Tuple[Any, pd.DataFrame]:
    """
    Computes SHAP values using TreeExplainer/Explainer, saves a SHAP summary plot,
    and logs top 5 most impactful features for glucose prediction.

    Args:
        model: Trained tree-based regression model instance.
        X_train: DataFrame or array of training features.
        X_test: DataFrame or array of test/validation features.
        feature_names: Optional list of feature names.
        output_dir: Directory path to save output artifacts.

    Returns:
        shap_values: Calculated SHAP values object or array.
        top_features_df: DataFrame listing features ranked by mean absolute SHAP value.
    """
    if feature_names is None:
        feature_names = list(X_test.columns) if isinstance(X_test, pd.DataFrame) else [f"feature_{i}" for i in range(X_test.shape[1])]

    print("[SHAP] Initializing SHAP Explainer and computing feature contributions...")
    
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer(X_test)
    except Exception:
        # Fallback to general Explainer
        explainer = shap.Explainer(model, X_train)
        shap_values = explainer(X_test)

    # Convert shap_values matrix for feature ranking
    if hasattr(shap_values, "values"):
        vals = np.abs(shap_values.values)
    else:
        vals = np.abs(np.array(shap_values))

    mean_abs_shap = np.mean(vals, axis=0)
    
    # Handle multi-dimensional output if any
    if mean_abs_shap.ndim > 1:
        mean_abs_shap = np.mean(mean_abs_shap, axis=-1)

    top_features_df = pd.DataFrame({
        "Feature": feature_names,
        "Mean_Abs_SHAP": mean_abs_shap
    }).sort_values(by="Mean_Abs_SHAP", ascending=False).reset_index(drop=True)

    print("\n--- TOP 5 MOST IMPACTFUL OPTICAL & PHYSIOLOGICAL FEATURES ---")
    for idx, row in top_features_df.head(5).iterrows():
        print(f"  {idx+1}. {row['Feature']}: {row['Mean_Abs_SHAP']:.4f}")
    print("-------------------------------------------------------------\n")

    # Generate and save SHAP Summary Plot
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_plot_path = output_dir / "shap_summary.png"

    plt.figure(figsize=(10, 6), dpi=300)
    
    if hasattr(shap_values, "values"):
        shap.summary_plot(shap_values, X_test, feature_names=feature_names, show=False)
    else:
        shap.summary_plot(shap_values, X_test, feature_names=feature_names, show=False)

    plt.title("SHAP Feature Importance Summary - Blood Glucose Model", fontsize=12, fontweight="bold", pad=12)
    plt.tight_layout()
    plt.savefig(summary_plot_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"[SHAP] SHAP Summary Plot saved to: {summary_plot_path.resolve()}")

    return shap_values, top_features_df
