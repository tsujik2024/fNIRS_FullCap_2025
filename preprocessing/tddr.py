"""
tddr.py

Implements the Temporal Derivative Distribution Repair (TDDR) algorithm
for motion artifact correction in fNIRS data.
"""

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt

def tddr(data: pd.DataFrame, sample_rate: float) -> pd.DataFrame:
    """
    Apply Temporal Derivative Distribution Repair (TDDR) to correct motion artifacts
    in fNIRS data.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing fNIRS data. Each float-type column is treated as a
        channel (e.g., 'CH1 HbO', 'CH1 HbR'), while non-float columns (e.g.,
        'Sample number', 'Event') are skipped.
    sample_rate : float
        Sampling rate of the data in Hz.

    Returns
    -------
    corrected_df : pd.DataFrame
        DataFrame with TDDR-corrected data for each float-type channel.
    """
    corrected_df = data.copy()

    # Apply TDDR to each float column (e.g., O2Hb and HHb channels)
    for col in corrected_df.columns:
        # Only process float64 columns
        if corrected_df[col].dtype == np.float64:
            corrected_df[col] = _tddr_on_signal(
                np.array(corrected_df[col], dtype='float64'),
                sample_rate
            )

    return corrected_df

def _tddr_on_signal(signal: np.ndarray, sample_rate: float) -> np.ndarray:
    """
    Internal function that implements the TDDR algorithm on a single 1D signal.

    Parameters
    ----------
    signal : np.ndarray
        One-dimensional fNIRS signal (e.g., O2Hb or HHb).
    sample_rate : float
        Sampling rate in Hz.

    Returns
    -------
    corrected_signal : np.ndarray
        The motion-corrected signal.
    """
    # Remove mean so we can focus on fluctuations
    signal_mean = np.mean(signal)
    signal_centered = signal - signal_mean

    # Low-pass filter at 0.5 Hz (3rd-order Butterworth)
    # - Wn=0.5 means a 0.5 Hz cutoff frequency (absolute, not fraction of Nyquist, thanks to fs=sample_rate)
    sos = butter(N=3, Wn=0.5, output='sos', fs=sample_rate)
    signal_low = sosfiltfilt(sos, signal_centered)
    signal_high = signal_centered - signal_low

    # Compute derivative of the low-frequency component
    deriv = np.diff(signal_low)

    # Initialize weights
    w = np.ones_like(deriv)

    # Iteratively estimate robust weights
    for _ in range(50):
        mu = np.sum(w * deriv) / np.sum(w)
        dev = np.abs(deriv - mu)
        sigma = 1.4826 * np.median(dev)
        r = dev / (sigma * 4.685)  # 4.685 is a tuning constant
        w = ((1 - r**2) * (r < 1)) ** 2

    # Repair derivative
    new_deriv = w * (deriv - mu)
    signal_low_corrected = np.cumsum(np.insert(new_deriv, 0, 0.0))

    corrected_signal = signal_low_corrected + signal_high + signal_mean
    return corrected_signal
