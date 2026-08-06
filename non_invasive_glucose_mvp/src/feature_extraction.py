"""
Comprehensive Physiological & Morphological Feature Extraction Engine.
Extracts:
  1. Per-channel time-domain & morphological pulse features (Crest time, Decay time, PPI, PW50, PW75, Area ratios).
  2. Derivative features (VPG max/min velocity, APG a,b,c,d,e wave amplitudes and vascular compliance ratios).
  3. Multi-wavelength optical AC/DC density ratios (R-values across 660, 730, 850, and 940 nm).
  4. Merges signal features with subject demographics and exports to artifacts/cleaned_features.csv.
"""

from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis

from src.config import (
    SAMPLING_FREQ,
    WAVELENGTHS,
    CHANNEL_NAMES,
    CLEANED_FEATURES_PATH,
    TARGET_COLUMN,
)
from src.signal_processing import (
    preprocess_subject_signals,
    detect_pulse_peaks_and_troughs,
)


def extract_apg_waves(apg_cycle: np.ndarray) -> Dict[str, float]:
    """
    Extracts a, b, c, d, e wave amplitudes from a single APG pulse cycle.
    a: initial positive peak
    b: early negative trough
    c: late positive peak
    d: late negative trough
    e: early diastolic positive wave
    """
    n = len(apg_cycle)
    if n < 10:
        return {"a": 1.0, "b": 0.0, "c": 0.0, "d": 0.0, "e": 0.0}

    # Region 1: a-wave (0 to 30% of cycle)
    a_idx = np.argmax(apg_cycle[: max(3, int(n * 0.35))])
    a_val = apg_cycle[a_idx]
    if a_val == 0:
        a_val = 1e-8

    # Region 2: b-wave (after a_idx to 45% of cycle)
    b_start = a_idx + 1
    b_end = max(b_start + 1, int(n * 0.45))
    b_val = np.min(apg_cycle[b_start:b_end]) if b_end > b_start else apg_cycle[a_idx]

    # Region 3: c-wave (40% to 65%)
    c_start = min(b_end, int(n * 0.40))
    c_end = max(c_start + 1, int(n * 0.65))
    c_val = np.max(apg_cycle[c_start:c_end]) if c_end > c_start else apg_cycle[a_idx]

    # Region 4: d-wave (55% to 80%)
    d_start = min(c_end, int(n * 0.55))
    d_end = max(d_start + 1, int(n * 0.80))
    d_val = np.min(apg_cycle[d_start:d_end]) if d_end > d_start else apg_cycle[a_idx]

    # Region 5: e-wave (70% to 100%)
    e_start = min(d_end, int(n * 0.70))
    e_val = np.max(apg_cycle[e_start:]) if e_start < n else apg_cycle[a_idx]

    return {"a": float(a_val), "b": float(b_val), "c": float(c_val), "d": float(d_val), "e": float(e_val)}


def extract_channel_morphology(
    signal_norm: np.ndarray,
    vpg: np.ndarray,
    apg: np.ndarray,
    fs: float = SAMPLING_FREQ,
) -> Dict[str, float]:
    """
    Extracts pulse cycle metrics, derivative metrics, and morphological ratios for a single channel.
    """
    peaks, troughs = detect_pulse_peaks_and_troughs(signal_norm, fs=fs)

    features = {}

    # Basic signal statistics
    features["mean"] = float(np.mean(signal_norm))
    features["std"] = float(np.std(signal_norm))
    features["skew"] = float(skew(signal_norm))
    features["kurtosis"] = float(kurtosis(signal_norm))

    # Peak-to-Peak Intervals (PPI)
    if len(peaks) > 1:
        ppi_samples = np.diff(peaks)
        ppi_sec = ppi_samples / fs
        features["ppi_mean"] = float(np.mean(ppi_sec))
        features["ppi_std"] = float(np.std(ppi_sec))
        features["hr_bpm"] = float(60.0 / (features["ppi_mean"] + 1e-8))
    else:
        features["ppi_mean"] = 0.8
        features["ppi_std"] = 0.0
        features["hr_bpm"] = 75.0

    # Pulse waveform cycle breakdown
    crest_times = []
    decay_times = []
    pw50_list = []
    pw75_list = []
    area_ratios = []

    apg_a_list, apg_b_list, apg_c_list, apg_d_list, apg_e_list = [], [], [], [], []

    # Segment cycles between consecutive troughs
    for i in range(len(troughs) - 1):
        t1, t2 = troughs[i], troughs[i + 1]
        cycle_len = t2 - t1
        if cycle_len < int(0.35 * fs) or cycle_len > int(1.5 * fs):
            continue

        cycle_sig = signal_norm[t1:t2]
        cycle_apg = apg[t1:t2]

        pk_in_cycle = np.where((peaks > t1) & (peaks < t2))[0]
        if len(pk_in_cycle) == 0:
            p_rel = np.argmax(cycle_sig)
        else:
            p_rel = peaks[pk_in_cycle[0]] - t1

        # Crest time (Tr) & Decay time (Td)
        tr = p_rel / fs
        td = (cycle_len - p_rel) / fs
        crest_times.append(tr)
        decay_times.append(td)

        # Area under curve ratios (Systolic A1 vs Diastolic A2)
        a1 = np.sum(cycle_sig[:p_rel]) if p_rel > 0 else 1e-8
        a2 = np.sum(cycle_sig[p_rel:]) if p_rel < cycle_len else 1e-8
        area_ratios.append(a1 / (a2 + 1e-8))

        # Pulse Width at 50% and 75% height
        amp = np.max(cycle_sig) - np.min(cycle_sig)
        h50 = np.min(cycle_sig) + 0.50 * amp
        h75 = np.min(cycle_sig) + 0.75 * amp

        above_50 = np.where(cycle_sig >= h50)[0]
        above_75 = np.where(cycle_sig >= h75)[0]

        pw50 = (above_50[-1] - above_50[0]) / fs if len(above_50) > 1 else 0.1
        pw75 = (above_75[-1] - above_75[0]) / fs if len(above_75) > 1 else 0.05
        pw50_list.append(pw50)
        pw75_list.append(pw75)

        # APG wave amplitudes
        apg_waves = extract_apg_waves(cycle_apg)
        apg_a_list.append(apg_waves["a"])
        apg_b_list.append(apg_waves["b"])
        apg_c_list.append(apg_waves["c"])
        apg_d_list.append(apg_waves["d"])
        apg_e_list.append(apg_waves["e"])

    # Aggregate cycle features
    features["crest_time_mean"] = float(np.mean(crest_times)) if crest_times else 0.15
    features["decay_time_mean"] = float(np.mean(decay_times)) if decay_times else 0.65
    features["area_ratio_mean"] = float(np.mean(area_ratios)) if area_ratios else 1.0
    features["pw50_mean"] = float(np.mean(pw50_list)) if pw50_list else 0.2
    features["pw75_mean"] = float(np.mean(pw75_list)) if pw75_list else 0.1

    # VPG velocity metrics
    features["vpg_max_velocity"] = float(np.max(vpg))
    features["vpg_min_velocity"] = float(np.min(vpg))
    features["vpg_std_velocity"] = float(np.std(vpg))

    # APG vascular elasticity ratios
    a_m = np.mean(apg_a_list) if apg_a_list else 1.0
    b_m = np.mean(apg_b_list) if apg_b_list else 0.0
    c_m = np.mean(apg_c_list) if apg_c_list else 0.0
    d_m = np.mean(apg_d_list) if apg_d_list else 0.0
    e_m = np.mean(apg_e_list) if apg_e_list else 0.0

    denom_a = a_m if abs(a_m) > 1e-8 else 1.0
    features["apg_b_over_a"] = float(b_m / denom_a)
    features["apg_c_over_a"] = float(c_m / denom_a)
    features["apg_d_over_a"] = float(d_m / denom_a)
    features["apg_e_over_a"] = float(e_m / denom_a)
    features["apg_aging_index"] = float((b_m - c_m - d_m - e_m) / denom_a)

    return features


def compute_optical_ratios(
    raw_signal_df: pd.DataFrame,
    filtered_signal_df: pd.DataFrame
) -> Dict[str, float]:
    """
    Computes AC (peak-to-trough amplitude) and DC (mean baseline intensity)
    per channel, and extracts cross-wavelength optical absorption R-values.
    """
    ac_dc_per_channel = {}
    ratios = {}

    for ch in CHANNEL_NAMES:
        raw_sig = raw_signal_df[ch].to_numpy()
        filt_sig = filtered_signal_df[ch].to_numpy()

        dc = np.mean(raw_sig)
        if dc == 0:
            dc = 1e-8

        # AC amplitude: 95th percentile minus 5th percentile of filtered pulse
        ac = np.percentile(filt_sig, 95) - np.percentile(filt_sig, 5)

        ac_dc = ac / dc
        ac_dc_per_channel[ch] = ac_dc
        ratios[f"AC_DC_{ch}"] = float(ac_dc)
        ratios[f"DC_{ch}"] = float(dc)
        ratios[f"AC_{ch}"] = float(ac)

    # Compute all pairwise optical density ratios R_{lambda1/lambda2} = (AC/DC)_1 / (AC/DC)_2
    pairs = [
        ("660nm", "940nm"),
        ("730nm", "850nm"),
        ("660nm", "730nm"),
        ("660nm", "850nm"),
        ("730nm", "940nm"),
        ("850nm", "940nm"),
    ]

    for ch1, ch2 in pairs:
        val1 = ac_dc_per_channel[ch1]
        val2 = ac_dc_per_channel[ch2]
        denom = val2 if abs(val2) > 1e-8 else 1e-8
        ratios[f"R_{ch1}_{ch2}"] = float(val1 / denom)

    return ratios


def extract_features_for_subject(
    subject_id: int,
    raw_signal_df: pd.DataFrame,
    meta_row: pd.Series,
    fs: float = SAMPLING_FREQ
) -> Dict[str, float]:
    """
    Pipeline to extract all morphological, derivative, optical, and demographic features for one subject.
    """
    proc = preprocess_subject_signals(raw_signal_df, fs=fs)

    feature_dict = {"ID": subject_id}

    # Demographics & Blood measurements
    demo_fields = ["Age", "Gender", "Height", "Weight", "Hemoglobin", "BMI"]
    for field in demo_fields:
        if field in meta_row:
            val = meta_row[field]
            if field == "Gender":
                feature_dict["Gender_male"] = 1.0 if str(val).lower() == "male" else 0.0
            else:
                feature_dict[field] = float(val)

    # Target Blood Glucose
    feature_dict[TARGET_COLUMN] = float(meta_row[TARGET_COLUMN])

    # Multi-wavelength optical density ratios
    optical_ratios = compute_optical_ratios(proc["raw"], proc["filtered"])
    feature_dict.update(optical_ratios)

    # Per-channel morphological & derivative features
    for ch in CHANNEL_NAMES:
        ch_features = extract_channel_morphology(
            proc["normalized"][ch].to_numpy(),
            proc["vpg"][ch].to_numpy(),
            proc["apg"][ch].to_numpy(),
            fs=fs,
        )
        for fname, fval in ch_features.items():
            feature_dict[f"{ch}_{fname}"] = fval

    return feature_dict


def build_feature_dataset(
    metadata_df: pd.DataFrame,
    signals_dict: Dict[int, pd.DataFrame],
    save_csv: bool = True,
    output_path: Path = CLEANED_FEATURES_PATH,
) -> pd.DataFrame:
    """
    Iterates across all valid subjects, runs feature extraction, constructs feature DataFrame,
    and exports artifacts/cleaned_features.csv.
    """
    extracted_rows = []

    for _, meta_row in metadata_df.iterrows():
        sub_id = int(meta_row["ID"])
        if sub_id in signals_dict:
            raw_sig = signals_dict[sub_id]
            subj_features = extract_features_for_subject(sub_id, raw_sig, meta_row)
            extracted_rows.append(subj_features)

    features_df = pd.DataFrame(extracted_rows)

    if save_csv:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        features_df.to_csv(output_path, index=False)
        print(f"[FeatureExtraction] Features matrix saved to: {output_path} (Shape: {features_df.shape})")

    return features_df
