# process_file.py
import os
import numpy as np
import pandas as pd
import logging
from typing import Optional, Tuple, List, Dict
from fnirs_FullCap_2025.read.loaders import read_txt_file
from fnirs_FullCap_2025.read.channel_utils import get_short_map, adjust_regions_to_naming
from fnirs_FullCap_2025.preprocessing.fir_filter import fir_filter
from fnirs_FullCap_2025.preprocessing.SCI import calc_sci
from fnirs_FullCap_2025.preprocessing.short_channel_regression import scr_regression
from fnirs_FullCap_2025.preprocessing.tddr import tddr
from fnirs_FullCap_2025.preprocessing.baseline_correction import baseline_subtraction
from fnirs_FullCap_2025.preprocessing.average_channels import FullCapChannelAverager
from fnirs_FullCap_2025.viz.visualizer import FNIRSVisualizer

logger = logging.getLogger(__name__)

class FullCapProcessor:
    def __init__(self, fs: float = 50.0, sci_threshold: float = 0.6):
        self.fs = fs
        self.sci_threshold = sci_threshold
        self.visualizer = FNIRSVisualizer(fs=fs)
        self.channel_averager = FullCapChannelAverager()
        self.warning_files = []

    def process_file(
            self,
            file_path: str,
            output_base_dir: str,
            input_base_dir: str,
            y_limits: Optional[Tuple[float, float]] = None
    ) -> Optional[pd.DataFrame]:
        """
        Run full pipeline on one OxySoft *.txt* export:
          • raw plots  (all-channels, per-region, grand-mean)
          • processed plots (grand-mean, per-region)
          • processed CSV
        Returns the processed DataFrame or *None* on failure.
        """
        try:
            # ── directories & names ─────────────────────────────────────────
            out_dir = self._create_output_dir(output_base_dir, input_base_dir, file_path)
            basename = os.path.splitext(os.path.basename(file_path))[0]

            # ── load txt ────────────────────────────────────────────────────
            raw_df, events = self._load_and_prep_data(file_path)
            if raw_df is None:
                return None

            # ── RAW PLOTS ───────────────────────────────────────────────────
            # 1️⃣ all channels in one PDF
            self.visualizer.plot_raw_all_channels(
                data=raw_df,
                output_path=os.path.join(out_dir, f"{basename}_raw_all_channels.pdf"),
                y_limits=y_limits,
            )

            # -------------------------------------------------------------
            # 2️⃣  regional averages  →  DataFrame
            # -------------------------------------------------------------
            # (a) average individual channels into regions
            raw_region_df = self.channel_averager.average_regions(raw_df)  # ← removed region_map arg

            # (b) add the combined L/R hemisphere columns
            raw_region_df = self.channel_averager.average_hemispheres(raw_region_df)  # ← NEW

            #region_map = adjust_regions_to_naming(raw_df.columns)
            #raw_region_df = self.channel_averager.average_regions(raw_df, region_map)

            # ──────────────────────────────────────────────────────────────
            # 2️⃣ build dict {region: DataFrame} for raw-region plot
            # ──────────────────────────────────────────────────────────────
            region_dict: Dict[str, pd.DataFrame] = {}

            # Accept either “… oxy” or “…_oxy” just in case
            oxy_suffixes = ("_oxy", " oxy")

            for col in raw_region_df.columns:
                if col.endswith(oxy_suffixes):
                    # figure out which suffix we matched
                    matched_suffix = next(s for s in oxy_suffixes if col.endswith(s))
                    region = col[:-len(matched_suffix)].rstrip("_ ").strip()  # ⇒ "PFC_L"  or "PFC_combined"

                    # build the twin HHb column name with the same separator
                    deoxy_col = f"{region}{matched_suffix.replace('oxy', 'deoxy')}"

                    if deoxy_col in raw_region_df.columns:
                        tmp = raw_region_df[[col, deoxy_col]].copy()
                        safe_region = region.replace(" ", "_")  # final safety pass
                        tmp.columns = [f"{safe_region}_oxy", f"{safe_region}_deoxy"]
                        region_dict[safe_region] = tmp

            # If we found at least one valid region, plot it
            if region_dict:
                self.visualizer.plot_raw_regions(
                    regional_data=region_dict,
                    output_path=os.path.join(out_dir, f"{basename}_raw_regions.pdf"),
                    y_limits=y_limits
                )
            else:
                logger.warning(
                    f"No valid region pairs found for {file_path}; skipping raw-region plot."
                )

            # 3️⃣ raw grand mean
            hbo_cols = [c for c in raw_df if 'HbO' in c or 'O2Hb' in c]
            hhb_cols = [c for c in raw_df if 'HHb' in c or 'HbR' in c]
            raw_overall = pd.DataFrame({
                'grand_oxy': raw_df[hbo_cols].mean(axis=1),
                'grand_deoxy': raw_df[hhb_cols].mean(axis=1),
            })
            self.visualizer.plot_raw_overall(
                data=raw_overall,
                output_path=os.path.join(out_dir, f"{basename}_raw_overall.pdf"),
                y_limits=y_limits,
            )

            # ── PROCESSING PIPELINE ─────────────────────────────────────────
            proc_df = self._process_pipeline_stages(raw_df, events)
            if proc_df is None:
                return None

            # ── PROCESSING PIPELINE ─────────────────────────────────────────
            proc_df = self._process_pipeline_stages(raw_df, events)
            if proc_df is None:
                return None

            # ── NEW ► average processed channels into regions ─────────────
            proc_region_df = self.channel_averager.average_regions(proc_df)

            # add combined hemispheres
            proc_region_df = self.channel_averager.average_hemispheres(proc_region_df)  # ← NEW

            # rename “ … oxy / … deoxy”  → “…_oxy / …_deoxy”
            proc_region_df.columns = [
                c.replace(" oxy", "_oxy").replace(" deoxy", "_deoxy")
                for c in proc_region_df.columns
            ]

            # add to the main processed DataFrame (keep channel cols too if you like)
            proc_df = pd.concat([proc_df, proc_region_df], axis=1)

            # ── PROCESSED PLOTS ─────────────────────────────────────────────
            proc_df['grand_oxy'] = proc_df[[c for c in proc_df if c.endswith('_oxy')]].mean(axis=1)
            proc_df['grand_deoxy'] = proc_df[[c for c in proc_df if c.endswith('_deoxy')]].mean(axis=1)

            self.visualizer.plot_processed_overall(
                data=proc_df,
                output_path=os.path.join(out_dir, f"{basename}_processed_overall.pdf"),
                y_limits=y_limits,
            )
            self.visualizer.plot_processed_regions(
                data=proc_df,
                output_path=os.path.join(out_dir, f"{basename}_processed_regions.pdf"),
                y_limits=y_limits,
            )

            # ── save CSV ────────────────────────────────────────────────────
            proc_df.to_csv(os.path.join(out_dir, f"{basename}_processed.csv"), index=False)
            return proc_df

        except Exception as exc:
            logger.error(f"Error processing {file_path}: {exc}", exc_info=True)
            self.warning_files.append((file_path, str(exc)))
            return None

    def _create_output_dir(self, output_base: str, input_base: str, file_path: str) -> str:
        rel = os.path.relpath(os.path.dirname(file_path), start=input_base)
        out = os.path.join(output_base, rel)
        os.makedirs(out, exist_ok=True)
        return out

    def _load_and_prep_data(self, file_path: str) -> Tuple[Optional[pd.DataFrame], pd.DataFrame]:
        try:
            raw_dict = read_txt_file(file_path)
            df = raw_dict['data']
            events = raw_dict.get('events', pd.DataFrame())
            events = self._clean_events(events)
            if 'Sample number' not in df:
                df.insert(0, 'Sample number', np.arange(len(df)))
            # Drop first second
            drop = int(self.fs)
            if len(df) > drop:
                df = df.iloc[drop:].reset_index(drop=True)
                df['Sample number'] = np.arange(len(df))
                if not events.empty:
                    events = events[events['Sample number'] >= drop].copy()
                    events['Sample number'] -= drop
            return df, events
        except Exception as e:
            logger.error(f"Load error {file_path}: {e}")
            return None, pd.DataFrame()

    def _clean_events(self, events: pd.DataFrame) -> pd.DataFrame:
        if events.empty:
            return pd.DataFrame(columns=['Sample number', 'Event'])
        ev = events.copy()
        ev['Event'] = (
            ev['Event'].astype(str)
            .str.upper()
            .str.replace(r"\s+", "", regex=True)
        )
        return ev.dropna(subset=['Event'])

    def _validate_time_values(self, data: pd.DataFrame) -> float:
        try:
            arr = np.arange(len(data)) / float(self.fs)
            dur = float(arr[-1])
            if not isinstance(dur, (int, float)):
                raise TypeError
            return dur
        except Exception:
            raise ValueError("Invalid time calculation")

    def _process_pipeline_stages(
            self,
            data: pd.DataFrame,
            events: pd.DataFrame
    ) -> Optional[pd.DataFrame]:
        """
        FIR → SCR → TDDR → baseline – returns DataFrame with
        columns ending '_oxy' / '_deoxy'.
        """
        try:
            hb_cols = [c for c in data
                       if any(tag in c for tag in ('HbO', 'O2Hb', 'HHb', 'HbR'))]
            num_only = data[hb_cols].apply(pd.to_numeric, errors='coerce')

            # 1️⃣ FIR band-pass (array → DataFrame)
            filt = fir_filter(num_only,
                              order=1000, Wn=[0.01, 0.1], fs=int(self.fs))

            self._calculate_sci(filt)

            # 2️⃣ Short-channel regression
            scr_df = self._apply_scr(filt)  # already DataFrame

            # 3️⃣ TDDR (array → DataFrame)
            tddr_df = tddr(scr_df, sample_rate=self.fs)  # pass the DataFrame itself
            if tddr_df is None:
                raise ValueError("TDDR failed")

            # 4️⃣ Baseline correction (returns DataFrame)
            proc = self._apply_baseline_correction(tddr_df, events)

            # 5️⃣ Rename “… oxy/deoxy” → “…_oxy/deoxy”
            rename_map = {}
            for c in proc.columns:
                if c.endswith(" oxy"):
                    rename_map[c] = c.replace(" oxy", "_oxy")
                elif c.endswith(" deoxy"):
                    rename_map[c] = c.replace(" deoxy", "_deoxy")
            proc = proc.rename(columns=rename_map)

            return proc

        except Exception as err:
            logger.error(f"Processing pipeline failed: {err}", exc_info=True)
            return None

    def _calculate_sci(self, data: pd.DataFrame) -> None:
        hbo = [c for c in data if 'HbO' in c or 'O2Hb' in c]
        hhb = [c for c in data if 'HHb' in c or 'HbR' in c]
        for ox in hbo:
            matches = [c for c in hhb if ox.split()[0] in c]
            if not matches: continue
            try:
                sci = calc_sci(data[ox].values, data[matches[0]].values, fs=self.fs, apply_filter=False)
                if sci < self.sci_threshold:
                    logger.warning(f"Low SCI {sci:.2f} for {ox}")
            except Exception as e:
                logger.warning(f"SCI error {ox}: {e}")

    def _apply_scr(self, data: pd.DataFrame) -> pd.DataFrame:
        """Short-channel regression that always returns a DataFrame."""
        try:
            long_oxy = [c for c in data if 'O2Hb' in c or 'HbO' in c]
            mapping = get_short_map(long_oxy)

            if not mapping:  # no SC channels present
                return data.copy()

            long_cols = list(mapping.keys())
            short_cols = list(mapping.values())
            long_hhb = [c.replace('HbO', 'HHb').replace('O2Hb', 'HHb') for c in long_cols]
            short_hhb = [c.replace('HbO', 'HHb').replace('O2Hb', 'HHb') for c in short_cols]

            # SCR returns a NumPy array – capture the order we pass in
            ordered = long_cols + long_hhb
            scr_arr = scr_regression(
                data[ordered],
                data[short_cols + short_hhb]
            )

            scr_df = pd.DataFrame(scr_arr, columns=ordered, index=data.index)
            other = data.drop(columns=ordered + short_cols + short_hhb, errors='ignore')

            return pd.concat([scr_df, other], axis=1)

        except Exception as e:
            logger.warning(f"SCR failed: {e}")
            return data.copy()

    def _apply_baseline_correction(self, data: pd.DataFrame, events: pd.DataFrame, baseline_duration: float = 20.0) -> pd.DataFrame:
        try:
            if events.empty:
                start = 0
            else:
                start = events.iloc[0]['Sample number']
            end = start + int(baseline_duration * self.fs)
            bdf = pd.DataFrame({'Sample number':[start,end], 'Event':['BaselineStart','BaselineEnd']})
            return baseline_subtraction(data, bdf)
        except Exception as e:
            logger.warning(f"Baseline error: {e}")
            return data.copy()