import re
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np


class FullCapChannelAverager:
    """
    Handles channel averaging for full-head fNIRS caps with complex channel arrangements.
    Manages region definitions, hemisphere separation, and channel naming conventions.
    """

    def __init__(self):
        """
        Initialize with full-cap specific channel configuration.
        """
        # Define regions and their channels (using your region definitions)
        self.regions = {
            "PFC_L": {"channels": [4, 6, 7], "hemisphere": "left"},
            "PFC_R": {"channels": [0, 1, 2], "hemisphere": "right"},
            "SMA_L": {"channels": [11, 12], "hemisphere": "left"},
            "SMA_R": {"channels": [15, 16], "hemisphere": "right"},
            "M1_L": {"channels": [9, 13], "hemisphere": "left"},
            "M1_R": {"channels": [17, 19], "hemisphere": "right"},
            "S1_L": {"channels": [10, 14], "hemisphere": "left"},
            "S1_R": {"channels": [18, 20], "hemisphere": "right"},
            "V1_L": {"channels": [8], "hemisphere": "left"},
            "V1_R": {"channels": [21], "hemisphere": "right"}
        }

        # Short channel mapping (from your get_short_map function)
        self.short_channel_mapping = {
            0: 14, 1: 14, 2: 14, 3: 16, 4: 16, 5: 16,
            6: 14, 7: 14, 8: 14, 9: 16, 10: 16, 11: 44,
            12: 44, 13: 44, 15: 52, 17: 52, 18: 52, 19: 52
        }

        self.current_naming_convention = None  # Will detect automatically

    def detect_naming_convention(self, column_names: List[str]) -> Optional[str]:
        """
        Detect the naming convention used in the data columns.

        Args:
            column_names: List of column names from the DataFrame

        Returns:
            Either 'HbO' or 'O2Hb' convention, or None if undetermined
        """
        if any('O2Hb' in col for col in column_names):
            return 'O2Hb'
        elif any('HbO' in col for col in column_names):
            return 'HbO'
        return None

    def get_region_channels(self, region: str, hb_type: str) -> List[str]:
        """
        Get the proper column names for a region and hemoglobin type.

        Args:
            region: Region name (e.g., "PFC_L")
            hb_type: Either "HbO" or "HHb"

        Returns:
            List of column names matching the current naming convention
        """
        if region not in self.regions:
            raise ValueError(f"Unknown region: {region}")

        base_names = [f"CH{ch}" for ch in self.regions[region]["channels"]]

        if self.current_naming_convention == 'O2Hb' and hb_type == 'HbO':
            return [f"{name} O2Hb" for name in base_names]
        return [f"{name} {hb_type}" for name in base_names]

    def average_regions(self, df: pd.DataFrame,
                        channels_to_exclude: Optional[List[int]] = None) -> pd.DataFrame:
        """
        Average channels by region, handling naming conventions and exclusions.

        Args:
            df: Input DataFrame with fNIRS data
            channels_to_exclude: List of channel numbers to exclude

        Returns:
            DataFrame with region averages
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame")

        self.current_naming_convention = self.detect_naming_convention(df.columns)
        channels_to_exclude = channels_to_exclude or []
        df_copy = df.copy()

        result_data = {}

        # Preserve metadata columns if present
        for col in ['Sample number', 'Event']:
            if col in df_copy.columns:
                result_data[col] = df_copy[col]

        # Calculate region averages
        for region in self.regions:
            # Get valid columns for this region
            oxy_cols = [col for col in self.get_region_channels(region, 'HbO')
                        if col in df_copy.columns and
                        int(re.search(r'CH(\d+)', col).group(1)) not in channels_to_exclude]
            deoxy_cols = [col for col in self.get_region_channels(region, 'HHb')
                          if col in df_copy.columns and
                          int(re.search(r'CH(\d+)', col).group(1)) not in channels_to_exclude]

            # Calculate averages
            result_data[f"{region}_oxy"] = self._safe_mean(df_copy, oxy_cols)
            result_data[f"{region}_deoxy"] = self._safe_mean(df_copy, deoxy_cols)

        return pd.DataFrame(result_data, index=df_copy.index)

    def average_hemispheres(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Combine left and right hemisphere regions into combined averages.

        Args:
            df: DataFrame containing region-level data

        Returns:
            DataFrame with additional combined hemisphere averages
        """
        result_df = df.copy()

        # Get unique region prefixes (e.g., "PFC" from "PFC_L" and "PFC_R")
        region_types = set([name.split('_')[0] for name in self.regions.keys()])

        for region_type in region_types:
            for hb_type in ['oxy', 'deoxy']:
                left_col = f"{region_type}_L_{hb_type}"
                right_col = f"{region_type}_R_{hb_type}"
                combined_col = f"{region_type}_combined_{hb_type}"

                if left_col in df.columns and right_col in df.columns:
                    result_df[combined_col] = (df[left_col] + df[right_col]) / 2
                elif left_col in df.columns:
                    result_df[combined_col] = df[left_col]
                elif right_col in df.columns:
                    result_df[combined_col] = df[right_col]

        return result_df

    def get_short_map(self, column_names: List[str]) -> Dict[str, str]:
        """
        Map long channels to their corresponding short channels.
        """
        # First try exact matches with current naming convention
        hb_suffix = "O2Hb" if self.current_naming_convention == 'O2Hb' else "HbO"

        manual_mapping = {}
        for long_num, short_num in self.short_channel_mapping.items():
            long_col = f"CH{long_num} {hb_suffix}"
            short_col = f"CH{short_num} {hb_suffix}"
            if long_col in column_names and short_col in column_names:
                manual_mapping[long_col] = short_col

        if manual_mapping:
            return manual_mapping

        # Fallback to pattern matching if needed
        pattern = re.compile(r'CH(\d+) (O2Hb|HbO)')
        channel_map = {}

        for col in column_names:
            match = pattern.match(col)
            if match:
                ch_num = int(match.group(1))
                if ch_num in self.short_channel_mapping:
                    short_num = self.short_channel_mapping[ch_num]
                    short_col = f"CH{short_num} {match.group(2)}"
                    if short_col in column_names:
                        channel_map[col] = short_col

        return channel_map

    def _safe_mean(self, df: pd.DataFrame, columns: List[str]) -> pd.Series:
        """Calculate mean while handling empty column lists."""
        if not columns:
            return pd.Series(np.nan, index=df.index)
        return df[columns].mean(axis=1)

    def adjust_regions_to_naming(self, column_names: List[str]) -> Dict[str, List[str]]:
        """
        Adjust region channel names to match the current naming convention.
        """
        adjusted_regions = {}
        for region_name, region_data in self.regions.items():
            orig_channels = (
                    [f"CH{ch} HbO" for ch in region_data["channels"]] +
                    [f"CH{ch} HHb" for ch in region_data["channels"]]
            )

            adjusted_channels = []
            for ch in orig_channels:
                if 'HbO' in ch and self.current_naming_convention == 'O2Hb':
                    adjusted_ch = ch.replace('HbO', 'O2Hb')
                else:
                    adjusted_ch = ch

                if adjusted_ch in column_names:
                    adjusted_channels.append(adjusted_ch)

            adjusted_regions[region_name] = adjusted_channels

        return adjusted_regions