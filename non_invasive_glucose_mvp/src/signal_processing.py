"""
Digital Signal Processing Engine for Multi-Wavelength PPG Signals.
Includes Butterworth bandpass filtering, zero-phase distortion removal,
z-score normalization, derivative computation (VPG/APG), and peak/trough detection.
"""

from typing import Tuple, Dict
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks

from src.config import (
    SAMPLING_FREQ,
    BANDPASS_LOW,
    BANDPASS_HIGH,
    FILTER_ORDER,
    CHANNEL_NAMES,
)


def butter_bandpass_filter(
    data: np.ndarray,
    lowcut: float = BANDPASS_LOW,
    highcut: float = BANDPASS_HIGH,
    fs: float = SAMPLING_FREQ,
    order: int = FILTER_ORDER,
) -> np.ndarray:
    """
    Applies a zero-phase 4th-order Butterworth Bandpass filter (0.5 Hz - 8.0 Hz).
    Removes low-frequency baseline wander (<0.5 Hz) and high-frequency noise (>8.0 Hz).
    """
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype="band")
    
    padlen = 3 * max(len(a), len(b))
    n_samples = len(data) if data.ndim == 1 else data.shape[0]

    if n_samples <= padlen:
        return data.copy()

    # Apply zero-phase filtering across 1D or 2D arrays
    if data.ndim == 1:
        filtered = filtfilt(b, a, data)
    else:
        filtered = filtfilt(b, a, data, axis=0)
    
    return filtered


def normalize_signal(data: np.ndarray, method: str = "zscore") -> np.ndarray:
    """
    Normalizes optical signal intensities per channel.
    Methods: 'zscore' (zero mean, unit variance) or 'minmax' (0 to 1).
    """
    if method == "zscore":
        mean = np.mean(data, axis=0, keepdims=True)
        std = np.std(data, axis=0, keepdims=True)
        std = np.where(std == 0, 1e-8, std)
        return (data - mean) / std
    elif method == "minmax":
        min_val = np.min(data, axis=0, keepdims=True)
        max_val = np.max(data, axis=0, keepdims=True)
        range_val = max_val - min_val
        range_val = np.where(range_val == 0, 1e-8, range_val)
        return (data - min_val) / range_val
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def compute_derivatives(signal_1d: np.ndarray, fs: float = SAMPLING_FREQ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes 1st derivative (VPG = Velocity Photoplethysmogram)
    and 2nd derivative (APG = Acceleration Photoplethysmogram).
    """
    dt = 1.0 / fs
    vpg = np.gradient(signal_1d, dt)
    apg = np.gradient(vpg, dt)
    return vpg, apg


def detect_pulse_peaks_and_troughs(
    signal_1d: np.ndarray,
    fs: float = SAMPLING_FREQ
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Detects systolic pulse peaks and onset troughs in a 1D PPG signal.
    Ensures minimum peak distance corresponds to max heart rate (~180 bpm -> 0.33s -> fs * 0.33 samples).
    """
    min_distance = int(0.35 * fs)  # At 200Hz, 70 samples (~50-170 bpm)
    
    # Detect peaks
    peaks, _ = find_peaks(
        signal_1d,
        distance=min_distance,
        prominence=0.1 * (np.max(signal_1d) - np.min(signal_1d))
    )
    
    # Detect troughs by finding peaks in inverted signal
    troughs, _ = find_peaks(
        -signal_1d,
        distance=min_distance,
        prominence=0.1 * (np.max(signal_1d) - np.min(signal_1d))
    )

    return peaks, troughs


def preprocess_subject_signals(
    signal_df: pd.DataFrame,
    fs: float = SAMPLING_FREQ
) -> Dict[str, pd.DataFrame]:
    """
    Preprocesses raw 4-channel signal DataFrame:
      1. Butterworth bandpass filtering (0.5 - 8.0 Hz).
      2. Z-score normalization.
      3. Calculation of VPG and APG for each channel.

    Returns dictionary containing DataFrames for 'filtered', 'normalized', 'vpg', and 'apg'.
    """
    raw_array = signal_df[CHANNEL_NAMES].to_numpy()

    # Step 1: Filter
    filtered_array = butter_bandpass_filter(raw_array, fs=fs)
    filtered_df = pd.DataFrame(filtered_array, columns=CHANNEL_NAMES)

    # Step 2: Normalize
    norm_array = normalize_signal(filtered_array, method="zscore")
    norm_df = pd.DataFrame(norm_array, columns=CHANNEL_NAMES)

    # Step 3: Derivatives
    vpg_dict = {}
    apg_dict = {}
    for col in CHANNEL_NAMES:
        v, a = compute_derivatives(norm_df[col].to_numpy(), fs=fs)
        vpg_dict[col] = v
        apg_dict[col] = a

    vpg_df = pd.DataFrame(vpg_dict)
    apg_df = pd.DataFrame(apg_dict)

    return {
        "filtered": filtered_df,
        "normalized": norm_df,
        "vpg": vpg_df,
        "apg": apg_df,
        "raw": signal_df,
    }
