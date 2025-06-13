# loaders.py -------------------------------------------------------------
"""Full‑cap fNIRS TXT reader for the **fnirs_FullCap_2025** pipeline.

This single loader is the *only* point where we touch raw OxySoft TXT
exports.  It guarantees that every caller receives:

* numeric hemoglobin columns (float64)
* numeric ``Sample number`` (int/float – depends on input)
* string ``Event`` with blanks for missing values
* a tidy ``events`` DataFrame that includes onset sample and duration

Down‑stream functions (FIR filter, SCR, TDDR, plotting, etc.) should **not**
perform any additional type coercion or column renaming.
"""

from __future__ import annotations
import re
import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# -------------------------------------------------------------------------
# Public API
# -------------------------------------------------------------------------

def read_txt_file(file_path: str | Path) -> Dict[str, pd.DataFrame | dict]:
    """Parse a single OxySoft ``.txt`` export.

    Parameters
    ----------
    file_path
        Absolute or relative path to a ``.txt`` file.

    Returns
    -------
    dict
        ``{"metadata": dict, "data": DataFrame, "events": DataFrame}``

    Notes
    -----
    * Column names are standardised to the pattern ``CH# HbO`` / ``CH# HHb``.
    * All hemoglobin channels are float64; events remain strings.
    * The helper will raise *early* if the file is empty or malformed so the
      calling script can skip/record bad files.
    """

    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(file_path)

    # ── Read all non‑blank lines ──────────────────────────────────────────
    try:
        with file_path.open("r", encoding="utf-8", errors="ignore") as fh:
            lines: List[str] = [ln.strip() for ln in fh if ln.strip()]
    except Exception as err:  # pragma: no cover
        logger.error("Could not open %s – %s", file_path, err)
        raise

    if not lines:
        raise IOError(f"File '{file_path}' is empty or unreadable.")

    rows: List[List[str]] = [ln.split("\t") for ln in lines]

    # ── Parse sections ────────────────────────────────────────────────────
    metadata: dict = _read_metadata(rows, file_path)
    df: pd.DataFrame = _read_data(rows, file_path)

    # Channel rename ► type coercion ► event extraction
    df = _reassign_channels(df, file_path)
    _coerce_column_types(df)
    events_df: pd.DataFrame = _extract_events(df)

    return {"metadata": metadata, "data": df, "events": events_df}

# -------------------------------------------------------------------------
# Helper functions (kept private – leading underscore)
# -------------------------------------------------------------------------

def _read_metadata(rows: List[List[str]], file_path: Path) -> Dict[str, str]:
    """Grab simple key:value pairs from the header (first ~7 lines)."""
    meta: Dict[str, str] = {"Export file": str(file_path)}
    for row in rows[:7]:
        if not row:
            continue
        if "OxySoft export of:" in row[0]:
            meta["Original file"] = row[1] if len(row) > 1 else "Unknown"
            continue
        if len(row) >= 2 and ":" in row[0]:
            key = row[0].split(":")[0].strip()
            meta[key] = row[1].strip()
    return meta

# -------------------------------------------------------------------------
#  _read_data – parse OxySoft “three-rows-per-sample” section
# -------------------------------------------------------------------------



NUM_RE = re.compile(r"^[+-]?\d+(?:\.\d+)?$")  # numeric value detector


def _read_data(rows: list, file_path: str) -> pd.DataFrame:
    """
    Parse the DATA part of an OxySoft full-cap TXT export.

    The file has interleaved O2Hb/HHb values (O2Hb-0, HHb-0, O2Hb-1, HHb-1...).
    Each sample spans multiple rows with the sample number in column 1 of first row.
    """
    # ── locate header section ────────────────────────────────────────────
    start = end = sample_rate = None
    for idx, row in enumerate(rows):
        if "Datafile sample rate:" in row:
            try:
                sample_rate = int(float(row[1]))
            except Exception:
                sample_rate = None
        elif "(Sample number)" in row:
            start = idx
        elif "(Event)" in row:
            end = idx
            break

    if None in (start, end, sample_rate):
        raise ValueError(f"Malformed header in {file_path}")

    # ── grab data rows (everything after the blank line) ────────────────
    data_rows = rows[end + 4:]
    if not data_rows:
        raise ValueError(f"No data rows found in {file_path}")
    # drop trailing blank line
    if data_rows and len(data_rows[-1]) == 1 and data_rows[-1][0] == "":
        data_rows = data_rows[:-1]

    # ── process samples ──────────────────────────────────────────────────
    logical = []
    i = 0

    while i < len(data_rows):
        row = data_rows[i]
        # Skip empty rows
        if not row:
            i += 1
            continue

        # Check if this row starts a new sample (has numeric value in column 1)
        if not NUM_RE.match(row[0]):
            i += 1
            continue

        # This is a row with a sample number
        sample_num = int(float(row[0]))

        # Collect all values until we find the next sample
        values = []
        event = ""
        j = i

        while j < len(data_rows):
            current_row = data_rows[j]

            # If we're on the first row, start after sample number
            start_col = 1 if j == i else 0

            # Process each cell in this row
            for col in range(start_col, len(current_row)):
                cell = current_row[col].strip()

                # Check if this is a numeric value (part of the 52 channels)
                if NUM_RE.match(cell) and len(values) < 52:
                    values.append(cell)
                # Check if this is an event marker (non-numeric after all channels)
                elif cell and len(values) >= 52 and not event:
                    event = cell

            j += 1

            # Stop if we hit the next sample or have all values plus event
            if j < len(data_rows) and NUM_RE.match(data_rows[j][0]):
                break

        # If we got all 52 channels, process this sample
        if len(values) >= 52:
            # Split interleaved values into O2Hb and HHb
            hbo = values[0::2]  # even indices (0, 2, 4...)
            hhb = values[1::2]  # odd indices (1, 3, 5...)

            # Ensure we have exactly 26 of each (truncate or pad if needed)
            hbo = hbo[:26]
            hhb = hhb[:26]

            # Pad if somehow we didn't get enough values
            if len(hbo) < 26:
                hbo.extend(['nan'] * (26 - len(hbo)))
            if len(hhb) < 26:
                hhb.extend(['nan'] * (26 - len(hhb)))

            logical.append([sample_num, *hbo, *hhb, event])

        # Move to the row where we left off
        i = j

    # ── build DataFrame ────────────────────────────────────────────────
    cols = (
            ["Sample number"] +
            [f"CH{c} HbO" for c in range(26)] +
            [f"CH{c} HHb" for c in range(26)] +
            ["Event"]
    )
    df = pd.DataFrame(logical, columns=cols)

    # numeric coercion
    num_cols = [c for c in df.columns if c not in ("Sample number", "Event")]
    df[num_cols] = df[num_cols].apply(pd.to_numeric, errors="coerce")
    df["Sample number"] = pd.to_numeric(df["Sample number"], errors="coerce")

    # Enhanced numeric coercion
    num_cols = [c for c in df.columns if c not in ("Sample number", "Event")]
    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').astype('float64')

    # Ensure Sample number is properly converted
    df["Sample number"] = pd.to_numeric(df["Sample number"], errors='coerce').astype('int64')

    # tidy Event column
    df["Event"] = (
        df["Event"]
        .astype(str)
        .str.strip()
        .replace({"": np.nan, "nan": np.nan})
    )

    # remove startup artifact (samples ≤ 3)
    df = df[df["Sample number"] > 3].reset_index(drop=True)

    # Debug info to help troubleshoot "shorter than 1 second" issue
    num_samples = len(df)
    duration_sec = num_samples / sample_rate if sample_rate else 0
    print(f"Processed {num_samples} samples ({duration_sec:.2f} seconds at {sample_rate} Hz)")

    return df

def _coerce_column_types(df: pd.DataFrame) -> None:
    """Convert columns in‑place to their final dtypes."""
    for col in df.columns:
        if col == "Event":
            df[col] = (
                df[col]
                .astype(str)
                .str.strip()
                .replace({"nan": "", "NaN": "", "None": ""})
            )
        elif col == "Sample number":
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce")


def _extract_events(df: pd.DataFrame) -> pd.DataFrame:
    """Translate the *Event* column into an onset/duration table."""
    if not {"Sample number", "Event"}.issubset(df.columns):
        return pd.DataFrame(columns=["Sample number", "Event", "Duration"])

    events: list[dict] = []
    current = None
    onset = None

    for samp, ev in zip(df["Sample number"], df["Event"]):
        if not ev:
            continue
        if ev != current:
            # close previous
            if current is not None:
                events.append({
                    "Sample number": onset,
                    "Event": current,
                    "Duration": samp - onset,
                })
            current = ev
            onset = samp
    # close final
    if current is not None:
        events.append({
            "Sample number": onset,
            "Event": current,
            "Duration": df["Sample number"].iloc[-1] - onset + 1,
        })

    return pd.DataFrame(events)


def _reassign_channels(df: pd.DataFrame, file_path: Path) -> pd.DataFrame:
    """Robustly convert *any* O2Hb/HHb column pair into ``CH# HbO`` / ``CH# HHb``.

    Works even when the file contains extra columns (SCI values, timestamps, etc.)
    or when the channel order is not strictly [O2Hb, HHb, O2Hb, HHb, …].
    """
    cols = list(df.columns)
    meta_cols = [c for c in ("Sample number", "Event") if c in cols]

    # --- Identify oxy / deoxy channels -----------------------------------
    oxy_cols  = [c for c in cols if re.search(r"(HbO|O2Hb)", c, re.IGNORECASE)]
    deoxy_cols = [c for c in cols if re.search(r"(HHb|HbR)",  c, re.IGNORECASE)]

    if not oxy_cols or not deoxy_cols or len(oxy_cols) != len(deoxy_cols):
        logger.warning("Unequal or missing HbO/HHb columns in %s – keeping original names", file_path)
        return df

    # Preserve original order as it appears in the txt file
    oxy_cols_sorted   = [c for c in cols if c in oxy_cols]
    deoxy_cols_sorted = [c for c in cols if c in deoxy_cols]

    # Map each oxy channel to its paired deoxy **by appearance order**
    paired = list(zip(oxy_cols_sorted, deoxy_cols_sorted))

    new_names = meta_cols.copy()
    for idx, (oxy, deoxy) in enumerate(paired):
        new_names.extend([f"CH{idx} HbO", f"CH{idx} HHb"])

    # Append any non‑Hb columns that slipped through (e.g., SCI)
    other_cols = [c for c in cols if c not in meta_cols + oxy_cols + deoxy_cols]
    new_names.extend(other_cols)

    # Re‑order DataFrame to match new_names list
    ordered_cols = meta_cols + oxy_cols_sorted + deoxy_cols_sorted + other_cols
    df = df[ordered_cols]
    df.columns = new_names
    return df