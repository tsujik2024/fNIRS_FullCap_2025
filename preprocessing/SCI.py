import numpy as np
import pandas as pd
from .fir_filter import fir_filter  # Adjust import path as needed


def calc_sci(signal1: np.ndarray, signal2: np.ndarray, fs=50.0, order=1000, Wn=[0.01, 0.1], apply_filter=True) -> float:
    """
    Calculate the Signal Coupling Index (SCI) between two fNIRS signals by computing the correlation coefficient.
    If apply_filter is True (default), signals are filtered (and z-scored) before computing the correlation.
    If False, the signals are assumed already filtered.

    Parameters
    ----------
    signal1 : np.ndarray
        1D array representing one fNIRS signal (e.g., O2Hb).
    signal2 : np.ndarray
        1D array representing the other fNIRS signal (e.g., HHb).
    fs : float, optional
        Sampling rate in Hz. Default is 50 Hz.
    order : int, optional
        Order of the FIR filter. Default is 1000.
    Wn : list, optional
        Cutoff frequencies for the FIR filter. Default is [0.01, 0.1].
    apply_filter : bool, optional
        If True, filter (and z-score) the signals; if False, use them as is.

    Returns
    -------
    sci_value : float
        Correlation coefficient between the (optionally filtered) signals.
    """
    if apply_filter:
        df1 = pd.DataFrame({'data': signal1})
        df2 = pd.DataFrame({'data': signal2})
        filt1_df = fir_filter(df1, order=order, Wn=Wn, fs=int(fs))
        filt2_df = fir_filter(df2, order=order, Wn=Wn, fs=int(fs))
        filt1 = filt1_df['data'].to_numpy()
        filt2 = filt2_df['data'].to_numpy()
    else:
        filt1 = signal1
        filt2 = signal2

    corr_matrix = np.corrcoef(filt1, filt2)
    sci_value = corr_matrix[0, 1]
    return sci_value
