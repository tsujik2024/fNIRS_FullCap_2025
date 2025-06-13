"""
Implements baseline subtraction for fNIRS data, supporting:
- A custom baseline DataFrame (user-provided), or
- The baseline period marked by specific events, or
- A specified sample range.

For StopSignal task: uses a 20-second baseline period from W1/S1 marker.
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


def baseline_subtraction(
        df: pd.DataFrame,
        events_df: pd.DataFrame,
        baseline_df: pd.DataFrame = None
) -> pd.DataFrame:
    """
    Applies baseline subtraction to the given DataFrame of fNIRS signals.

    For StopSignal task:
    - Uses the 20-second period after the first W1/S1 marker as baseline
    - Falls back to first 20 seconds if markers not found

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing fNIRS data (columns for channels), plus any metadata
        columns like 'Sample number', 'Event', 'Time (s)' that should be ignored.
    events_df : pd.DataFrame
        DataFrame specifying events. For StopSignal task, should contain 'BaselineStart'
        and 'BaselineEnd' events marking the baseline period.
        Must have columns:
          - 'Sample number'
          - 'Event'
    baseline_df : pd.DataFrame, optional
        If provided, each channel's baseline mean is computed from this DataFrame
        instead of from events in `df`. Must have the same column names as `df`.
        Default is None.

    Returns
    -------
    corrected_df : pd.DataFrame
        A new DataFrame with the baseline-subtracted signals.

    Raises
    ------
    ValueError
        If the required baseline events are not found or their sample indices are out of range.
    """
    # Normalise event labels once at entry
    events_df["Event"] = (
        events_df["Event"]
        .astype(str)
        .str.strip()  # leading / trailing spaces
        .str.upper()  # case-insensitive
        .str.replace(r"\s+", "", regex=True)  # internal spaces (“S 1”→“S1”)
    )

    corrected_df = df.copy()

    # Identify which columns are channels vs. metadata
    ignore_cols = ['Sample number', 'Event', 'Time (s)', 'Condition', 'Subject']
    data_cols = [col for col in corrected_df.columns if col not in ignore_cols]

    if baseline_df is not None:
        # ------------------------------
        # Use the provided baseline_df
        # ------------------------------
        logger.info("Using provided baseline DataFrame for baseline subtraction")
        for ch in data_cols:
            baseline_mean = baseline_df[ch].mean()
            corrected_df[ch] = corrected_df[ch] - baseline_mean

    else:
            # ---------------------------------------------------
            # Compute baseline from events_df markers
            # ---------------------------------------------------
        logger.info("Computing baseline from events DataFrame")

            # ⬇️ FIX: Force 'Event' to string before searching
        events_df['Event'] = events_df['Event'].astype(str)
        events_df['Event'] = events_df['Event'].str.strip()
        events_df = events_df[events_df['Event'].str.contains(r'[a-zA-Z]', regex=True)]

            # Check for StopSignal baseline markers (BaselineStart and BaselineEnd)
        if 'BaselineStart' in events_df['Event'].values and 'BaselineEnd' in events_df['Event'].values:
            start_sample = events_df.loc[events_df['Event'] == 'BaselineStart', 'Sample number'].values[0]
            end_sample = events_df.loc[events_df['Event'] == 'BaselineEnd', 'Sample number'].values[0]

            # Fall back to traditional S1/S2 markers if available
        elif 'S1' in events_df['Event'].values and 'S2' in events_df['Event'].values:
            logger.info("Using S1/S2 markers to define baseline period")
            start_sample = events_df.loc[events_df['Event'] == 'S1', 'Sample number'].values[0]
            end_sample = events_df.loc[events_df['Event'] == 'S2', 'Sample number'].values[0]

            # Another fallback - if W1/S1 are found, use 20 seconds from that point
        elif any(marker in events_df['Event'].values for marker in ['S1', 'W1']):
            logger.info("Using W1/S1 marker to define baseline period (20s duration)")
                # Find the first W1 or S1 marker
            for marker in ['S1', 'W1']:
                if marker in events_df['Event'].values:
                    start_sample = events_df.loc[events_df['Event'] == marker, 'Sample number'].values[0]
                    break

            # Estimate sampling rate if not explicitly available
            if hasattr(df, 'fs'):
                fs = df.fs
            else:
                # Estimate from time column if available
                if 'Time (s)' in df.columns and len(df['Time (s)']) > 1:
                    time_diff = df['Time (s)'].iloc[1] - df['Time (s)'].iloc[0]
                    if time_diff > 0:
                        fs = 1 / time_diff
                    else:
                        fs = 50.0  # Default to 50 Hz if can't determine
                else:
                    fs = 50.0  # Default to 50 Hz

            # Use 20 seconds from start_sample
            end_sample = start_sample + int(20 * fs)

        else:
            # Last resort - use first 20 seconds if no markers found
            logger.warning("No baseline markers found, using first 20 seconds")
            start_sample = 4

            # Estimate sampling rate if not explicitly available (as above)
            if hasattr(df, 'fs'):
                fs = df.fs
            else:
                if 'Time (s)' in df.columns and len(df['Time (s)']) > 1:
                    time_diff = df['Time (s)'].iloc[1] - df['Time (s)'].iloc[0]
                    if time_diff > 0:
                        fs = 1 / time_diff
                    else:
                        fs = 50.0
                else:
                    fs = 50.0

            end_sample = int(20 * fs)

        start = int(start_sample)
        end = int(end_sample)

        # Check if 'start' and 'end' are within bounds
        if not (0 <= start < len(corrected_df)) or not (0 < end <= len(corrected_df)):
            logger.warning(
                f"Event indices out of bounds: start={start}, end={end}, "
                f"data length={len(corrected_df)}. Adjusting to valid range."
            )
            start = max(0, min(start, len(corrected_df) - 1))
            end = max(1, min(end, len(corrected_df)))

        if start >= end:
            logger.warning(
                f"The baseline interval is invalid: start={start} >= end={end}. "
                "Using first 20 seconds instead."
            )
            start = 0
            end = min(int(20 * 50), len(corrected_df))  # Assume 50Hz if all else fails

        # Subtract the mean during the baseline period
        logger.info(f"Applying baseline correction using samples {start} to {end}")
        for ch in data_cols:
            baseline_segment = corrected_df.loc[start:end, ch]
            baseline_mean = baseline_segment.mean()
            corrected_df[ch] = corrected_df[ch] - baseline_mean

        # Add baseline info as attributes for debugging/verification
        corrected_df.attrs['baseline_start'] = start
        corrected_df.attrs['baseline_end'] = end

    return corrected_df