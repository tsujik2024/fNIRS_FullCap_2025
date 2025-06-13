import os
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class StatisticsCalculator:
    """
    Handles calculation and organization of fNIRS statistics including:
    - Grand averages
    - Regional averages
    - Time-segmented statistics
    - Summary sheet generation
    """

    def __init__(self, input_base_dir: str = ""):
        """
        Initialize the statistics calculator.

        Args:
            input_base_dir: Base directory for input files (used for path resolution)
        """
        self.input_base_dir = input_base_dir
        self.required_regions = ['pfc', 'sma', 'm1', 's1', 'v1']

    def calculate_subject_y_limits(self, subject_data: pd.DataFrame) -> Tuple[float, float]:
        """Calculate y-axis limits for plotting"""
        try:
            # Select only hemoglobin columns
            signal_cols = [col for col in subject_data.columns
                           if 'HbO' in col or 'HHb' in col]

            if not signal_cols:
                return (-1, 1)  # Default range if no columns found

            # Convert to numeric and drop NA
            numeric_data = subject_data[signal_cols].apply(pd.to_numeric, errors='coerce')
            numeric_data = numeric_data.dropna()

            if numeric_data.empty:
                return (-1, 1)  # Default range if all NA

            # Calculate max absolute value with buffer
            max_abs = max(abs(numeric_data.max().max()),
                          abs(numeric_data.min().min()))
            buffer = max_abs * 0.2  # 20% buffer
            return (-max_abs - buffer, max_abs + buffer)

        except Exception as e:
            logger.error(f"Error calculating y-limits: {str(e)}")
            return (-1, 1)  # Fallback range
    def collect_statistics(self, processed_files: List[str], output_base_dir: str) -> pd.DataFrame:
        """
        Generate comprehensive statistics from processed files.

        Args:
            processed_files: List of paths to processed files
            output_base_dir: Base output directory

        Returns:
            DataFrame containing all compiled statistics
        """
        all_stats = []

        for file_path in processed_files:
            try:
                stats = self._process_single_file(file_path, output_base_dir)
                if stats:
                    all_stats.append(stats)
            except Exception as e:
                logger.error(f"Error processing {file_path}: {str(e)}")

        return pd.DataFrame(all_stats) if all_stats else self._create_empty_stats_df()

    def _process_single_file(self, file_path: str, output_base_dir: str) -> Optional[Dict]:
        """Process statistics for a single file."""
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        processed_path = self._get_processed_path(file_path, output_base_dir)

        if not os.path.exists(processed_path):
            logger.warning(f"Processed file not found: {processed_path}")
            return None

        df = pd.read_csv(processed_path)

        if 'grand oxy' not in df.columns:
            logger.warning(f"Missing grand oxy in {processed_path}")
            return None

        # Extract metadata
        metadata = self._extract_metadata(file_path, file_name)

        # Calculate statistics
        stats = {
            **metadata,
            **self._calculate_grand_stats(df),
            **self._calculate_regional_stats(df)
        }

        logger.info(f"Processed stats for: {file_name}")
        return stats

    def _get_processed_path(self, file_path: str, output_base_dir: str) -> str:
        """Get path to processed CSV for a given input file."""
        relative_path = os.path.relpath(os.path.dirname(file_path), start=self.input_base_dir)
        output_dir = os.path.join(output_base_dir, relative_path)
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        return os.path.join(output_dir, f"{file_name}_processed.csv")

    def _extract_metadata(self, file_path: str, file_name: str) -> Dict:
        """Extract subject, timepoint, and condition from file path."""
        path_parts = file_path.split(os.sep)

        # Extract subject
        subject = next((p for p in path_parts
                        if "ohsu_turn" in p.lower() or
                        any(x in p.lower() for x in ["subject", "subj", "sub-"])), "Unknown")

        # Extract timepoint
        timepoint = next((p for p in path_parts
                          if p.lower() in ["baseline", "pre", "post"]), "Unknown")

        # Determine condition
        condition = ("LongWalk_ST" if "ST" in file_name or "SingleTask" in file_name
                     else "LongWalk_DT" if "DT" in file_name or "DualTask" in file_name
        else "Unknown")

        return {
            'Subject': subject,
            'Timepoint': timepoint,
            'Condition': condition
        }

    def _calculate_grand_stats(self, df: pd.DataFrame) -> Dict:
        """Calculate statistics for grand averages."""
        total_samples = len(df)
        half = total_samples // 2

        return {
            'Overall grand oxy Mean': df['grand oxy'].mean(),
            'First Half grand oxy Mean': df['grand oxy'].iloc[:half].mean(),
            'Second Half grand oxy Mean': df['grand oxy'].iloc[half:].mean()
        }

    def _calculate_regional_stats(self, df: pd.DataFrame) -> Dict:
        """Calculate statistics for all regions."""
        stats = {}

        for region in self.required_regions:
            oxy_col = f"{region}_combined oxy"

            if oxy_col in df.columns:
                half = len(df) // 2
                stats.update({
                    f'{region.upper()} Combined Overall Mean': df[oxy_col].mean(),
                    f'{region.upper()} Combined First Half Mean': df[oxy_col].iloc[:half].mean(),
                    f'{region.upper()} Combined Second Half Mean': df[oxy_col].iloc[half:].mean()
                })

        return stats

    def _create_empty_stats_df(self) -> pd.DataFrame:
        """Create empty DataFrame with correct columns."""
        columns = [
            'Subject', 'Timepoint', 'Condition',
            'Overall grand oxy Mean', 'First Half grand oxy Mean', 'Second Half grand oxy Mean'
        ]

        # Add columns for each region
        for region in self.required_regions:
            columns.extend([
                f'{region.upper()} Combined Overall Mean',
                f'{region.upper()} Combined First Half Mean',
                f'{region.upper()} Combined Second Half Mean'
            ])

        return pd.DataFrame(columns=columns)

    def create_summary_sheets(self, stats_df: pd.DataFrame, output_folder: str) -> None:
        """
        Generate summary CSV files for different conditions and regions.

        Args:
            stats_df: DataFrame containing all statistics
            output_folder: Directory to save summary files
        """
        if stats_df.empty:
            logger.warning("No statistics to summarize")
            return

        # Define output columns
        basic_cols = ['Subject', 'Timepoint']
        grand_cols = [
            'Overall grand oxy Mean',
            'First Half grand oxy Mean',
            'Second Half grand oxy Mean'
        ]
        region_cols = [
                          f'{r.upper()} Combined Overall Mean' for r in self.required_regions
                      ] + [
                          f'{r.upper()} Combined First Half Mean' for r in self.required_regions
                      ] + [
                          f'{r.upper()} Combined Second Half Mean' for r in self.required_regions
                      ]

        # Save full statistics
        stats_df.to_csv(os.path.join(output_folder, 'all_subjects_statistics.csv'), index=False)

        # Save condition-specific summaries
        for condition in ['LongWalk_ST', 'LongWalk_DT']:
            if condition in stats_df['Condition'].values:
                # Full summary
                cond_df = stats_df[stats_df['Condition'] == condition]
                cond_df[basic_cols + grand_cols + region_cols].to_csv(
                    os.path.join(output_folder, f'summary_{condition.split("_")[-1]}.csv'),
                    index=False
                )

                # Regional summary
                cond_df[basic_cols + region_cols].to_csv(
                    os.path.join(output_folder, f'regions_summary_{condition.split("_")[-1]}.csv'),
                    index=False
                )