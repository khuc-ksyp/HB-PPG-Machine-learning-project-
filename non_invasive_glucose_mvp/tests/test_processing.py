"""
Unit tests for signal processing module: Butterworth filtering, normalization, derivatives, and peak detection.
"""

import numpy as np
import pytest

from src.signal_processing import (
    butter_bandpass_filter,
    normalize_signal,
    compute_derivatives,
    detect_pulse_peaks_and_troughs,
)


def test_butter_bandpass_filter():
    fs = 200.0
    t = np.linspace(0, 10, int(10 * fs), endpoint=False)

    # Signal = 1 Hz (valid PPG band) + 0.1 Hz (DC baseline wander) + 20 Hz (high freq noise)
    sig_valid = np.sin(2 * np.pi * 1.0 * t)
    sig_low = 2.0 * np.sin(2 * np.pi * 0.1 * t)
    sig_high = 0.5 * np.sin(2 * np.pi * 20.0 * t)
    raw_signal = sig_valid + sig_low + sig_high

    filtered = butter_bandpass_filter(raw_signal, lowcut=0.5, highcut=8.0, fs=fs, order=4)

    assert len(filtered) == len(raw_signal)
    # Check that low frequency and high frequency components are substantially attenuated
    assert np.std(filtered) < np.std(raw_signal)


def test_normalize_signal():
    data = np.array([[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]])
    norm_z = normalize_signal(data, method="zscore")

    assert norm_z.shape == data.shape
    assert np.isclose(np.mean(norm_z, axis=0), [0.0, 0.0]).all()
    assert np.isclose(np.std(norm_z, axis=0), [1.0, 1.0]).all()


def test_compute_derivatives():
    fs = 200.0
    t = np.linspace(0, 1, int(fs), endpoint=False)
    sig = np.sin(2 * np.pi * 2.0 * t)

    vpg, apg = compute_derivatives(sig, fs=fs)

    assert len(vpg) == len(sig)
    assert len(apg) == len(sig)
    assert np.max(np.abs(vpg)) > 0
    assert np.max(np.abs(apg)) > 0


def test_detect_pulse_peaks_and_troughs():
    fs = 200.0
    t = np.linspace(0, 5, int(5 * fs), endpoint=False)
    # Simulated 1 Hz heartbeat pulse (5 pulses)
    sig = np.sin(2 * np.pi * 1.0 * t)

    peaks, troughs = detect_pulse_peaks_and_troughs(sig, fs=fs)

    assert len(peaks) >= 4
    assert len(troughs) >= 4
