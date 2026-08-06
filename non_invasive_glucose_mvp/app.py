"""
Streamlit Web Interface for Non-Invasive Blood Glucose Estimator.
Features:
  - Multi-wavelength telemetry status (660, 730, 850, 940 nm).
  - Subject selector & CSV signal uploader.
  - Interactive raw vs. filtered PPG signal plots (Plotly).
  - Estimated glucose display with clinical risk category badges (Hypo / Normal / Hyper).
  - Interactive Clarke Error Grid scatter plot highlighting selected subject.
  - SHAP feature contribution analysis.
  - 1-Point Personalization Calibration simulation.
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

from src.config import (
    METADATA_PATH,
    DATA_CSV_DIR,
    CLEANED_FEATURES_PATH,
    MODEL_SAVE_PATH,
    ARTIFACTS_DIR,
    CHANNEL_NAMES,
    SAMPLING_FREQ,
    TARGET_COLUMN,
    GROUP_CV_COLUMN,
)
from src.data_ingestion import load_metadata_excel, clean_and_impute_metadata, load_subject_signal
from src.signal_processing import preprocess_subject_signals
from src.feature_extraction import extract_features_for_subject
from src.clinical_evaluator import assign_clarke_zone, evaluate_clarke_error_grid
from src.calibration import apply_one_point_calibration


st.set_page_config(
    page_title="Non-Invasive Continuous Glucose Estimator",
    page_icon="🩸",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for dark/modern glassmorphism UI
CSS_STYLE = """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #1E88E5 0%, #43A047 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0px;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #B0BEC5;
        margin-bottom: 20px;
    }
    .telemetry-card {
        background-color: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 12px 18px;
        margin-bottom: 15px;
    }
    .metric-badge-normal {
        background-color: #2e7d32;
        color: white;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: bold;
        font-size: 1.1rem;
    }
    .metric-badge-hypo {
        background-color: #ed6c02;
        color: white;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: bold;
        font-size: 1.1rem;
    }
    .metric-badge-hyper {
        background-color: #d32f2f;
        color: white;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: bold;
        font-size: 1.1rem;
    }
    </style>
"""
st.markdown(CSS_STYLE, unsafe_allow_html=True)



@st.cache_data
def load_cached_data():
    try:
        raw_meta = load_metadata_excel(METADATA_PATH)
        clean_meta = clean_and_impute_metadata(raw_meta)
    except Exception:
        clean_meta = pd.DataFrame()

    try:
        if CLEANED_FEATURES_PATH.exists():
            features_df = pd.read_csv(CLEANED_FEATURES_PATH)
        else:
            features_df = pd.DataFrame()
    except Exception:
        features_df = pd.DataFrame()

    return clean_meta, features_df


@st.cache_resource
def load_cached_model():
    if MODEL_SAVE_PATH.exists():
        try:
            return joblib.load(MODEL_SAVE_PATH)
        except Exception:
            return None
    return None


clean_meta, features_df = load_cached_data()
model = load_cached_model()

# ==============================================================================
# HEADER & OPTICAL TELEMETRY PANEL
# ==============================================================================
st.markdown('<div class="main-header">🩸 Non-Invasive Continuous Blood Glucose Estimator</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Multi-Wavelength Optical Photoplethysmography (PPG) Sub-Surface Hemodynamic Engine</div>', unsafe_allow_html=True)

# Telemetry Bar
col_t1, col_t2, col_t3, col_t4 = st.columns(4)
with col_t1:
    st.markdown('<div class="telemetry-card">🔴 <b>660 nm (Red)</b><br><small>Superficial Capillary Bed</small></div>', unsafe_allow_html=True)
with col_t2:
    st.markdown('<div class="telemetry-card">🟡 <b>730 nm (Near-IR 1)</b><br><small>Deoxy-Hemoglobin Transition</small></div>', unsafe_allow_html=True)
with col_t3:
    st.markdown('<div class="telemetry-card">🟢 <b>850 nm (Near-IR 2)</b><br><small>Oxy-Hemoglobin Absorption</small></div>', unsafe_allow_html=True)
with col_t4:
    st.markdown('<div class="telemetry-card">🔵 <b>940 nm (Infrared)</b><br><small>Deep Dermal Water / Glucose</small></div>', unsafe_allow_html=True)

# ==============================================================================
# SIDEBAR CONTROLS
# ==============================================================================
st.sidebar.title("🎛️ Telemetry & Model Controls")

mode = st.sidebar.radio("Data Input Mode", ["Select Dataset Subject", "Upload PPG CSV File"])

subject_id = None
custom_signal_df = None

if mode == "Select Dataset Subject":
    if not clean_meta.empty:
        available_ids = clean_meta["ID"].unique()
        subject_id = st.sidebar.selectbox("Subject ID Selector", available_ids, index=0)
    else:
        st.sidebar.error("Dataset metadata unavailable.")
else:
    uploaded_file = st.sidebar.file_uploader("Upload 4-Channel PPG CSV (660, 730, 850, 940 nm)", type=["csv"])
    if uploaded_file is not None:
        custom_signal_df = pd.read_csv(uploaded_file)
        if custom_signal_df.shape[1] >= 4:
            custom_signal_df = custom_signal_df.iloc[:, :4]
            custom_signal_df.columns = CHANNEL_NAMES

# Personalization Calibration Toggle
st.sidebar.markdown("---")
st.sidebar.subheader("🎯 1-Point Personalization Calibration")
use_calibration = st.sidebar.checkbox("Enable 1-Point Offset Calibration", value=False)
custom_ref_glucose = None
if use_calibration:
    custom_ref_glucose = st.sidebar.number_input("Invasive Reference Glucose (mg/dL)", min_value=40.0, max_value=400.0, value=110.0, step=1.0)


# ==============================================================================
# DATA INGESTION & FEATURE EXTRACTION FOR SELECTED SUBJECT
# ==============================================================================
sig_df = None
selected_meta = None
estimated_glucose = None
raw_pred = None
ref_glucose = None

if mode == "Select Dataset Subject" and subject_id is not None:
    sig_df = load_subject_signal(subject_id, DATA_CSV_DIR)
    selected_meta_rows = clean_meta[clean_meta["ID"] == subject_id]
    if not selected_meta_rows.empty:
        selected_meta = selected_meta_rows.iloc[0]
        ref_glucose = float(selected_meta[TARGET_COLUMN]) if TARGET_COLUMN in selected_meta else None

elif mode == "Upload PPG CSV File" and custom_signal_df is not None:
    sig_df = custom_signal_df
    # Mock default metadata if custom file
    selected_meta = pd.Series({
        "ID": 999,
        "Age": 30.0,
        "Gender": "male",
        "Height": 170.0,
        "Weight": 70.0,
        "Hemoglobin": 140.0,
        "BMI": 24.2,
        TARGET_COLUMN: 100.0,
    })

if sig_df is not None and model is not None and selected_meta is not None:
    feat_dict = extract_features_for_subject(int(selected_meta["ID"]), sig_df, selected_meta)
    feat_df = pd.DataFrame([feat_dict])
    
    drop_cols = [c for c in [TARGET_COLUMN, GROUP_CV_COLUMN] if c in feat_df.columns]
    X_single = feat_df.drop(columns=drop_cols)
    
    raw_pred = float(model.predict(X_single)[0])
    
    if use_calibration and custom_ref_glucose is not None:
        estimated_glucose = apply_one_point_calibration(np.array([raw_pred]), custom_ref_glucose, index=0)[0]
    else:
        estimated_glucose = raw_pred

# ==============================================================================
# MAIN DISPLAY: ESTIMATED GLUCOSE METRICS BADGE
# ==============================================================================
col_m1, col_m2, col_m3, col_m4 = st.columns(4)

if estimated_glucose is not None:
    # Determine risk category
    if estimated_glucose < 70.0:
        badge_html = f'<span class="metric-badge-hypo">⚠️ Hypoglycemia ({estimated_glucose:.1f} mg/dL)</span>'
        risk_text = "LOW BLOOD GLUCOSE (<70 mg/dL)"
    elif estimated_glucose > 140.0:
        badge_html = f'<span class="metric-badge-hyper">🚨 Hyperglycemia ({estimated_glucose:.1f} mg/dL)</span>'
        risk_text = "HIGH BLOOD GLUCOSE (>140 mg/dL)"
    else:
        badge_html = f'<span class="metric-badge-normal">✅ Normal ({estimated_glucose:.1f} mg/dL)</span>'
        risk_text = "EUGLYCEMIC (70–140 mg/dL)"

    with col_m1:
        st.metric("Live Estimated Glucose", f"{estimated_glucose:.1f} mg/dL", delta=f"{estimated_glucose - ref_glucose:.1f} vs Ref" if ref_glucose else None)
    with col_m2:
        if ref_glucose is not None:
            st.metric("Invasive Reference Glucose", f"{ref_glucose:.1f} mg/dL")
        else:
            st.metric("Calibration Mode", "1-Point Enabled" if use_calibration else "Uncalibrated")
    with col_m3:
        st.markdown(f"**Clinical Status**<br>{badge_html}", unsafe_allow_html=True)
    with col_m4:
        if selected_meta is not None:
            st.markdown(f"**Subject Specs**<br>ID: #{selected_meta['ID']} | Age: {selected_meta.get('Age', 0):.0f} | BMI: {selected_meta.get('BMI', 0):.1f}", unsafe_allow_html=True)

st.markdown("---")

# ==============================================================================
# TAB NAVIGATION
# ==============================================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Interactive PPG Waveforms",
    "🎯 Clarke Error Grid Clinical Safety",
    "🔍 SHAP Feature Interpretability",
    "⚙️ Personalization Calibration"
])

# ------------------------------------------------------------------------------
# TAB 1: PPG WAVEFORMS
# ------------------------------------------------------------------------------
with tab1:
    st.subheader("Multi-Wavelength PPG Signal Telemetry")
    if sig_df is not None:
        proc = preprocess_subject_signals(sig_df, fs=SAMPLING_FREQ)
        
        show_channel = st.selectbox("Select Optical Wavelength Channel", CHANNEL_NAMES, index=0)
        time_sec = np.arange(len(sig_df)) / SAMPLING_FREQ

        fig_wave = go.Figure()
        fig_wave.add_trace(go.Scatter(
            x=time_sec[:1000],
            y=sig_df[show_channel].iloc[:1000],
            mode="lines",
            name=f"Raw {show_channel} Intensity",
            line=dict(color="#90A4AE", width=1.5, dash="dot"),
        ))
        fig_wave.add_trace(go.Scatter(
            x=time_sec[:1000],
            y=proc["filtered"][show_channel].iloc[:1000],
            mode="lines",
            name=f"Bandpass Filtered (0.5–8.0 Hz)",
            line=dict(color="#1E88E5", width=2.5),
        ))
        fig_wave.update_layout(
            title=f"Raw Intensity vs. Zero-Phase Filtered PPG Signal ({show_channel})",
            xaxis_title="Time (seconds)",
            yaxis_title="Optical Signal Intensity",
            template="plotly_dark",
            height=420,
        )
        st.plotly_chart(fig_wave, use_container_width=True)
    else:
        st.info("Please select a subject or upload a CSV file from the sidebar to view signals.")

# ------------------------------------------------------------------------------
# TAB 2: CLARKE ERROR GRID
# ------------------------------------------------------------------------------
with tab2:
    st.subheader("Clarke Error Grid Clinical Accuracy Analysis")
    if not features_df.empty and model is not None:
        drop_cols = [c for c in [TARGET_COLUMN, GROUP_CV_COLUMN] if c in features_df.columns]
        X_all = features_df.drop(columns=drop_cols)
        y_all_true = features_df[TARGET_COLUMN].to_numpy()
        y_all_pred = model.predict(X_all)

        # Build Clarke Scatter Plot
        ceg_zones = [assign_clarke_zone(r, p) for r, p in zip(y_all_true, y_all_pred)]
        ceg_df = pd.DataFrame({
            "Subject_ID": features_df[GROUP_CV_COLUMN],
            "Reference": y_all_true,
            "Predicted": y_all_pred,
            "Zone": ceg_zones,
        })

        fig_ceg = px.scatter(
            ceg_df,
            x="Reference",
            y="Predicted",
            color="Zone",
            hover_data=["Subject_ID"],
            title="Clarke Error Grid - Cohort Predictions",
            color_discrete_map={"A": "#2ca02c", "B": "#ff7f0e", "C": "#d62728", "D": "#9467bd", "E": "#8c564b"},
            template="plotly_dark",
            height=550,
        )
        
        # Add Reference Lines
        max_val = max(float(np.max(y_all_true)), float(np.max(y_all_pred)), 400.0)
        fig_ceg.add_trace(go.Scatter(x=[0, max_val], y=[0, max_val], mode="lines", name="y = x", line=dict(color="gray", dash="dash")))

        # Highlight current subject if selected
        if selected_meta is not None and estimated_glucose is not None and ref_glucose is not None:
            fig_ceg.add_trace(go.Scatter(
                x=[ref_glucose],
                y=[estimated_glucose],
                mode="markers",
                name=f"Current Subject #{selected_meta['ID']}",
                marker=dict(size=16, color="#00E676", symbol="star", line=dict(width=2, color="white")),
            ))

        st.plotly_chart(fig_ceg, use_container_width=True)

        zone_percentages, _ = evaluate_clarke_error_grid(np.asarray(y_all_true, dtype=float), np.asarray(y_all_pred, dtype=float))
        col_z1, col_z2, col_z3 = st.columns(3)
        col_z1.metric("Zone A (Accurate)", f"{zone_percentages['A']:.1f}%")
        col_z2.metric("Zone B (Benign)", f"{zone_percentages['B']:.1f}%")
        col_z3.metric("Zones A+B Combined", f"{zone_percentages['A_B_Combined']:.1f}%")
    else:
        st.info("Run `python run_pipeline.py` to generate feature artifacts and model binary for cohort visualization.")

# ------------------------------------------------------------------------------
# TAB 3: SHAP INTERPRETABILITY
# ------------------------------------------------------------------------------
with tab3:
    st.subheader("Model Interpretability & Feature Contributions")
    shap_summary_path = ARTIFACTS_DIR / "shap_summary.png"
    if shap_summary_path.exists():
        st.image(str(shap_summary_path), caption="SHAP Summary Plot - Feature Importances", use_container_width=True)
    else:
        st.info("SHAP Summary Plot will appear here after running `python run_pipeline.py`.")

# ------------------------------------------------------------------------------
# TAB 4: 1-POINT CALIBRATION DEMO
# ------------------------------------------------------------------------------
with tab4:
    st.subheader("1-Point Personalization Offset Calibration Simulation")
    st.write(
        "Individual optical properties (skin tone, epidermal thickness, vascular tone) introduce static baseline offsets. "
        "Providing a single reference blood glucose measurement anchors the baseline optical absorption curve."
    )

    if estimated_glucose is not None and ref_glucose is not None and raw_pred is not None:
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            st.metric("Raw Uncalibrated Estimate", f"{raw_pred:.1f} mg/dL", delta=f"{raw_pred - ref_glucose:.1f} Error")
        with col_c2:
            st.metric("1-Point Calibrated Estimate", f"{estimated_glucose:.1f} mg/dL", delta=f"{estimated_glucose - ref_glucose:.1f} Error")
