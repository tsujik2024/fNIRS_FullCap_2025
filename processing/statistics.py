import os
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class StatisticsCalculator:
    """
    Handles calculation and organization of fNIRS statistics including:
    - Grand averages (both HbO and HHb)
    - Regional averages (both HbO and HHb)
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
                           if 'HbO' in col or 'HHb' in col or 'oxy' in col or 'deoxy' in col]

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

    def collect_statistics_from_csvs(self, processed_files: List[str], output_base_dir: str) -> pd.DataFrame:
        """
        Generate comprehensive statistics directly from processed CSV files.
        This method bypasses the original file path requirements and works directly with CSVs.

        Args:
            processed_files: List of paths to processed CSV files
            output_base_dir: Base output directory

        Returns:
            DataFrame containing all compiled statistics
        """
        logger.info(f"Processing {len(processed_files)} CSV files for statistics...")

        if not processed_files:
            logger.warning("No processed files provided")
            return self._create_empty_stats_df()

        # Load and combine all CSV files
        all_dataframes = []
        for csv_file in processed_files:
            try:
                df = self._load_and_enrich_csv(csv_file)
                all_dataframes.append(df)
                logger.info(f"✅ Loaded: {os.path.basename(csv_file)}")
            except Exception as e:
                logger.error(f"❌ Error loading {csv_file}: {e}")

        if not all_dataframes:
            logger.error("No CSV files could be loaded!")
            return self._create_empty_stats_df()

        # Combine all dataframes
        combined_df = pd.concat(all_dataframes, ignore_index=True)
        logger.info(f"📈 Combined dataset shape: {combined_df.shape}")

        # Rename columns for compatibility
        combined_df = combined_df.rename(columns={
            'grand_oxy': 'grand oxy',
            'grand_deoxy': 'grand deoxy'
        })

        # Calculate grouped statistics
        return self._calculate_grouped_statistics(combined_df)

    def _load_and_enrich_csv(self, csv_file: str) -> pd.DataFrame:
        """Load CSV and add metadata columns if missing."""
        df = pd.read_csv(csv_file)

        # Add subject_id if missing
        if 'subject_id' not in df.columns:
            subject_id = os.path.basename(csv_file).split('_')[0] + '_' + os.path.basename(csv_file).split('_')[1]
            df['subject_id'] = subject_id

        # Add visit if missing
        if 'visit' not in df.columns:
            visit = next((part for part in csv_file.split('/') if part.startswith('Visit')), 'Unknown')
            df['visit'] = visit

        # Add condition if missing
        if 'condition' not in df.columns and 'Condition' not in df.columns:
            condition = self._extract_condition_from_filename(os.path.basename(csv_file))
            df['condition'] = condition
            df['Condition'] = condition

        # Add file path
        if 'file_path' not in df.columns:
            df['file_path'] = csv_file

        return df

    def _extract_condition_from_filename(self, filename: str) -> str:
        """Extract condition from filename with priority handling."""
        # Prioritize Cue_Walking before Walking
        if 'Cue_Walking' in filename:
            if 'DT1' in filename:
                return 'Cue_Walking_DT1'
            elif 'DT3' in filename:
                return 'Cue_Walking_DT3'
            elif 'ST' in filename:
                return 'Cue_Walking_ST'
            else:
                return 'Cue_Walking'
        elif 'Walking_ST' in filename or 'WalkingST' in filename:
            return 'Walking_ST'
        elif 'Walking_DT1' in filename or 'WalkingDT1' in filename:
            return 'Walking_DT1'
        elif 'Walking_DT2' in filename or 'WalkingDT2' in filename:
            return 'Walking_DT2'
        elif 'Walking_DT3' in filename or 'WalkingDT3' in filename:
            return 'Walking_DT3'
        elif 'Sitting' in filename:
            return 'Sitting'
        elif 'Standing' in filename:
            return 'Standing'
        else:
            # Fallback parsing
            parts = filename.replace('_processed.csv', '').split('_')
            for i in range(len(parts) - 1):
                if 'Walking' in parts[i] or 'Cue' in parts[i]:
                    return f"{parts[i]}_{parts[i + 1]}"
            return 'Unknown'

    def _calculate_grouped_statistics(self, combined_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate comprehensive statistics for both HbO and HHb channels."""
        logger.info("🧮 Calculating grouped statistics (grand + regional HbO + HHb)...")

        grouped_stats = []
        group_cols = ['subject_id', 'visit', 'condition']

        # Define all HbO and HHb columns to analyze
        hbo_columns = [col for col in combined_df.columns if col.endswith('_oxy') or col == 'grand oxy']
        hhb_columns = [col for col in combined_df.columns if col.endswith('_deoxy') or col == 'grand deoxy']

        # Also include individual channel columns
        hbo_channels = [col for col in combined_df.columns if 'HbO' in col]
        hhb_channels = [col for col in combined_df.columns if 'HHb' in col]

        # Combine all hemoglobin columns
        all_hbo_columns = list(set(hbo_columns + hbo_channels))
        all_hhb_columns = list(set(hhb_columns + hhb_channels))

        logger.info(f"Found {len(all_hbo_columns)} HbO columns and {len(all_hhb_columns)} HHb columns")

        for keys, group in combined_df.groupby(group_cols):
            subject_id, visit, condition = keys
            total_samples = len(group)
            half = total_samples // 2

            row = {
                'subject_id': subject_id,
                'visit': visit,
                'condition': condition,
                'Condition': condition
            }

            # Process HbO columns
            for col in all_hbo_columns:
                if col in group.columns:
                    try:
                        row[f'{col} - Overall Mean'] = group[col].mean()
                        row[f'{col} - First Half Mean'] = group[col].iloc[:half].mean()
                        row[f'{col} - Second Half Mean'] = group[col].iloc[half:].mean()
                    except Exception as e:
                        logger.warning(f"⚠️ Skipping HbO column {col} due to error: {e}")

            # Process HHb columns
            for col in all_hhb_columns:
                if col in group.columns:
                    try:
                        row[f'{col} - Overall Mean'] = group[col].mean()
                        row[f'{col} - First Half Mean'] = group[col].iloc[:half].mean()
                        row[f'{col} - Second Half Mean'] = group[col].iloc[half:].mean()
                    except Exception as e:
                        logger.warning(f"⚠️ Skipping HHb column {col} due to error: {e}")

            grouped_stats.append(row)

        return pd.DataFrame(grouped_stats)

    def collect_statistics(self, processed_files: List[str], output_base_dir: str) -> pd.DataFrame:
        """
        Original method - kept for backward compatibility but enhanced with error handling.
        """
        try:
            return self.collect_statistics_from_csvs(processed_files, output_base_dir)
        except Exception as e:
            logger.error(f"Error in collect_statistics: {str(e)}")
            logger.info("Falling back to direct CSV processing...")
            return self.collect_statistics_from_csvs(processed_files, output_base_dir)

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

        # Calculate statistics for both HbO and HHb
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
        condition = self._extract_condition_from_filename(file_name)

        return {
            'Subject': subject,
            'Timepoint': timepoint,
            'Condition': condition
        }

    def _calculate_grand_stats(self, df: pd.DataFrame) -> Dict:
        """Calculate statistics for grand averages - both HbO and HHb."""
        total_samples = len(df)
        half = total_samples // 2

        stats = {}

        # HbO (oxy) statistics
        if 'grand oxy' in df.columns:
            stats.update({
                'Overall grand oxy Mean': df['grand oxy'].mean(),
                'First Half grand oxy Mean': df['grand oxy'].iloc[:half].mean(),
                'Second Half grand oxy Mean': df['grand oxy'].iloc[half:].mean()
            })

        # HHb (deoxy) statistics
        if 'grand deoxy' in df.columns:
            stats.update({
                'Overall grand deoxy Mean': df['grand deoxy'].mean(),
                'First Half grand deoxy Mean': df['grand deoxy'].iloc[:half].mean(),
                'Second Half grand deoxy Mean': df['grand deoxy'].iloc[half:].mean()
            })

        return stats

    def _calculate_regional_stats(self, df: pd.DataFrame) -> Dict:
        """Calculate statistics for all regions - both HbO and HHb."""
        stats = {}

        for region in self.required_regions:
            # HbO statistics
            oxy_col = f"{region}_combined oxy"
            if oxy_col in df.columns:
                half = len(df) // 2
                stats.update({
                    f'{region.upper()} Combined Overall HbO Mean': df[oxy_col].mean(),
                    f'{region.upper()} Combined First Half HbO Mean': df[oxy_col].iloc[:half].mean(),
                    f'{region.upper()} Combined Second Half HbO Mean': df[oxy_col].iloc[half:].mean()
                })

            # HHb statistics
            deoxy_col = f"{region}_combined deoxy"
            if deoxy_col in df.columns:
                half = len(df) // 2
                stats.update({
                    f'{region.upper()} Combined Overall HHb Mean': df[deoxy_col].mean(),
                    f'{region.upper()} Combined First Half HHb Mean': df[deoxy_col].iloc[:half].mean(),
                    f'{region.upper()} Combined Second Half HHb Mean': df[deoxy_col].iloc[half:].mean()
                })

        return stats

    def _create_empty_stats_df(self) -> pd.DataFrame:
        """Create empty DataFrame with correct columns."""
        columns = [
            'subject_id', 'visit', 'condition', 'Condition',
            'Overall grand oxy Mean', 'First Half grand oxy Mean', 'Second Half grand oxy Mean',
            'Overall grand deoxy Mean', 'First Half grand deoxy Mean', 'Second Half grand deoxy Mean'
        ]

        # Add columns for each region (both HbO and HHb)
        for region in self.required_regions:
            columns.extend([
                f'{region.upper()} Combined Overall HbO Mean',
                f'{region.upper()} Combined First Half HbO Mean',
                f'{region.upper()} Combined Second Half HbO Mean',
                f'{region.upper()} Combined Overall HHb Mean',
                f'{region.upper()} Combined First Half HHb Mean',
                f'{region.upper()} Combined Second Half HHb Mean'
            ])

        return pd.DataFrame(columns=columns)

    def create_summary_sheets(self, stats_df: pd.DataFrame, output_folder: str) -> None:
        """
        Generate summary CSV files for different conditions and regions.
        Enhanced to handle both HbO and HHb data.

        Args:
            stats_df: DataFrame containing all statistics
            output_folder: Directory to save summary files
        """
        if stats_df.empty:
            logger.warning("No statistics to summarize")
            return

        # Define output columns
        basic_cols = ['subject_id', 'visit', 'condition']

        # Grand columns for both HbO and HHb
        grand_hbo_cols = [
            'Overall grand oxy Mean',
            'First Half grand oxy Mean',
            'Second Half grand oxy Mean'
        ]
        grand_hhb_cols = [
            'Overall grand deoxy Mean',
            'First Half grand deoxy Mean',
            'Second Half grand deoxy Mean'
        ]

        # Regional columns for both HbO and HHb
        region_hbo_cols = [
                              f'{r.upper()} Combined Overall HbO Mean' for r in self.required_regions
                          ] + [
                              f'{r.upper()} Combined First Half HbO Mean' for r in self.required_regions
                          ] + [
                              f'{r.upper()} Combined Second Half HbO Mean' for r in self.required_regions
                          ]

        region_hhb_cols = [
                              f'{r.upper()} Combined Overall HHb Mean' for r in self.required_regions
                          ] + [
                              f'{r.upper()} Combined First Half HHb Mean' for r in self.required_regions
                          ] + [
                              f'{r.upper()} Combined Second Half HHb Mean' for r in self.required_regions
                          ]

        # Save full statistics
        stats_df.to_csv(os.path.join(output_folder, 'all_subjects_statistics.csv'), index=False)
        logger.info("💾 Saved: all_subjects_statistics.csv")

        # Save condition-specific summaries
        unique_conditions = stats_df['condition'].unique()
        for condition in unique_conditions:
            if pd.notna(condition) and condition != 'Unknown':
                cond_df = stats_df[stats_df['condition'] == condition]

                # Determine appropriate columns based on what's available
                available_cols = [col for col in grand_hbo_cols + grand_hhb_cols + region_hbo_cols + region_hhb_cols
                                  if col in cond_df.columns]

                if available_cols:
                    # Full summary
                    cond_df[basic_cols + available_cols].to_csv(
                        os.path.join(output_folder, f'summary_{condition}.csv'),
                        index=False
                    )
                    logger.info(f"💾 Saved: summary_{condition}.csv")

        # Generate analysis summary
        self._generate_analysis_summary(stats_df, output_folder)

    def _generate_analysis_summary(self, combined_df: pd.DataFrame, output_folder: str) -> None:
        """Generate analysis summary with channel counts."""
        try:
            hbo_channels = [col for col in combined_df.columns if 'HbO' in col or '_oxy' in col]
            hhb_channels = [col for col in combined_df.columns if 'HHb' in col or '_deoxy' in col]
            left_regions = [col for col in combined_df.columns if '_L_' in col]
            right_regions = [col for col in combined_df.columns if '_R_' in col]
            combined_regions = [col for col in combined_df.columns if '_combined_' in col]
            grand_measures = [col for col in combined_df.columns if col.startswith('grand')]

            summary_info = {
                'total_subjects': len(combined_df['subject_id'].unique()) if 'subject_id' in combined_df.columns else 0,
                'total_visits': len(combined_df['visit'].unique()) if 'visit' in combined_df.columns else 0,
                'total_conditions': len(combined_df['condition'].unique()) if 'condition' in combined_df.columns else 0,
                'total_rows': len(combined_df),
                'individual_hbo_channels': len(hbo_channels),
                'individual_hhb_channels': len(hhb_channels),
                'left_regions': len(left_regions),
                'right_regions': len(right_regions),
                'combined_regions': len(combined_regions),
                'grand_measures': len(grand_measures)
            }

            summary_df = pd.DataFrame([summary_info])
            summary_file = os.path.join(output_folder, "analysis_summary.csv")
            summary_df.to_csv(summary_file, index=False)
            logger.info(f"📋 Saved: analysis_summary.csv")

        except Exception as e:
            logger.error(f"Error generating analysis summary: {e}")


# === STANDALONE EXECUTION SCRIPT ===
if __name__ == "__main__":
    # This allows the script to be run independently

    # === Set your path ===
    output_base_dir = "/Users/tsujik/Documents/Automaticityprocessedjune2025"
    stats_collector = StatisticsCalculator()

    # === Find all _processed.csv files ===
    processed_csv_files = []
    for root, dirs, files in os.walk(output_base_dir):
        for file in files:
            if file.endswith("_processed.csv") and "bad_SCI" not in file:
                full_path = os.path.join(root, file)
                processed_csv_files.append(full_path)

    print(f"🔍 Found {len(processed_csv_files)} processed CSVs")

    if not processed_csv_files:
        print("❌ No processed CSV files found!")
        exit()

    # === Run statistics collection ===
    print("📊 Running statistics collection...")
    try:
        stats_df = stats_collector.collect_statistics_from_csvs(processed_csv_files, output_base_dir)

        if stats_df.empty:
            print("❌ No statistics were calculated!")
        else:
            print(f"✅ Calculated statistics for {len(stats_df)} entries")

            # === Save results ===
            output_stats_file = os.path.join(output_base_dir, "all_subjects_statistics.csv")
            stats_df.to_csv(output_stats_file, index=False)
            print(f"💾 Saved: {output_stats_file}")

            # === Generate summary sheets ===
            try:
                stats_collector.create_summary_sheets(stats_df, output_base_dir)
                print("📊 Summary sheets created successfully!")
            except Exception as e:
                print(f"⚠️ Warning: Could not create summary sheets: {e}")

            # === Display preview ===
            print(f"\n📋 Statistics Preview:")
            preview_cols = ['subject_id', 'visit', 'condition']
            available_preview_cols = [col for col in preview_cols if col in stats_df.columns]

            # Add some data columns for preview
            data_cols = [col for col in stats_df.columns if 'Overall' in col and 'Mean' in col][:4]
            available_preview_cols.extend(data_cols)

            if available_preview_cols:
                print(stats_df[available_preview_cols].head())

        print("✅ Stats-only summary complete.")

    except Exception as e:
        print(f"❌ Error during statistics collection: {e}")
        import traceback

        traceback.print_exc()
