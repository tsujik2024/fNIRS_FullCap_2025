import os
from typing import List, Dict, Tuple, Optional
import logging
from fnirs_FullCap_2025.processing.process_file import FullCapProcessor

logger = logging.getLogger(__name__)


class BatchProcessor:
    """Handles batch processing of multiple fNIRS files."""

    def __init__(self, fs: float = 50.0, sci_threshold: float = 0.6):
        """
        Initialize batch processor with parameters.

        Args:
            fs: Sampling frequency in Hz
            sci_threshold: Threshold for SCI calculation
        """
        self.fs = fs
        self.sci_threshold = sci_threshold
        self.warning_files = []
        logger.info(f"Initialized BatchProcessor (fs={fs}, SCI threshold={sci_threshold})")

    def process_batch(self, input_base_dir: str, output_base_dir: str) -> Dict[str, List[str]]:
        """
        Process all files in the input directory.

        Args:
            input_base_dir: Root directory containing input files
            output_base_dir: Root directory for output files

        Returns:
            Dictionary mapping subjects to their processed files
        """
        try:
            # Ensure output directory exists
            os.makedirs(output_base_dir, exist_ok=True)

            # Find all text files
            txt_files = self._find_input_files(input_base_dir)
            if not txt_files:
                logger.warning(f"No .txt files found in {input_base_dir}")
                return {}

            # Organize files by subject
            subject_files = self._organize_files_by_subject(txt_files)

            # Process each subject's files
            processed_files = self._process_subject_files(subject_files, input_base_dir, output_base_dir)

            # Save warnings if any
            self._save_warnings(output_base_dir)

            return processed_files

        except Exception as e:
            logger.error(f"Batch processing failed: {str(e)}", exc_info=True)
            raise

    # Private helper methods ---------------------------------------------------

    def _find_input_files(self, input_dir: str) -> List[str]:
        """Find all .txt files in the input directory tree."""
        txt_files = []
        for root, _, files in os.walk(input_dir):
            for file in files:
                if file.endswith('.txt'):
                    txt_files.append(os.path.join(root, file))
        txt_files.sort()
        logger.info(f"Found {len(txt_files)} .txt files in {input_dir}")
        return txt_files

    def _organize_files_by_subject(self, file_paths: List[str]) -> Dict[str, List[str]]:
        """Organize files by subject based on directory structure."""
        subject_files = {}
        for file_path in file_paths:
            subject = self._extract_subject_id(file_path)
            subject_files.setdefault(subject, []).append(file_path)
        return subject_files

    @staticmethod
    def _extract_subject_id(file_path: str) -> str:
        """Extract subject ID from file path."""
        parts = file_path.split(os.sep)
        for part in parts:
            if "OHSU_Turn" in part or "sub-" in part:
                return part
        return "Unknown"

    def _process_subject_files(self, subject_files: Dict[str, List[str]],
                               input_base_dir: str, output_base_dir: str) -> Dict[str, List[str]]:
        """Process all files for each subject."""
        processed_files = {}

        for subject, files in subject_files.items():
            logger.info(f"\nProcessing subject: {subject}")

            # Calculate y-limits for this subject
            y_limits = self._calculate_subject_y_limits(files)

            # Process each file
            subject_processed = []
            for file_path in files:
                processor = FullCapProcessor(fs=self.fs, sci_threshold=self.sci_threshold)
                result = processor.process_file(
                    file_path=file_path,
                    output_base_dir=output_base_dir,
                    input_base_dir=input_base_dir,
                    y_limits=y_limits
                )

                if result is not None:
                    subject_processed.append(file_path)
                    self.warning_files.extend(processor.warning_files)

            processed_files[subject] = subject_processed
            logger.info(f"Processed {len(subject_processed)}/{len(files)} files for {subject}")

        return processed_files

    def _calculate_subject_y_limits(self, file_paths: List[str]) -> Optional[Tuple[float, float]]:
        """Calculate consistent y-axis limits for all files from one subject."""
        try:
            from fnirs_FullCap_2025.read.loaders import read_txt_file
            data_frames = []

            for file_path in file_paths:
                try:
                    data_frames.append(read_txt_file(file_path)["data"])
                except Exception as e:
                    self.warning_files.append((file_path, f"Y-limit read error: {str(e)}"))
                    logger.warning(f"Couldn't read {file_path} for y-limit calculation: {str(e)}")

            if not data_frames:
                return None

            all_max = 0
            for df in data_frames:
                cols = [col for col in df.columns if any(k in col for k in ['HbO', 'O2Hb', 'HHb', 'HbR'])]
                if cols:
                    cur_max = max(abs(df[cols].max().max()), abs(df[cols].min().min()))
                    all_max = max(all_max, cur_max)

            all_max *= 1.2  # Add padding
            return -all_max, all_max

        except Exception as e:
            logger.error(f"Y-limit calculation failed: {str(e)}")
            self.warning_files.append((None, f"Y-limit calculation error: {str(e)}"))
            return None

    def _save_warnings(self, output_base_dir: str) -> None:
        """Save all warning messages to a file."""
        if not self.warning_files:
            return

        warn_file = os.path.join(output_base_dir, "processing_warnings.txt")
        try:
            with open(warn_file, "w") as f:
                for path, msg in self.warning_files:
                    f.write(f"{path}: {msg}\n")
            logger.info(f"Saved {len(self.warning_files)} warnings to {warn_file}")
        except Exception as e:
            logger.error(f"Failed to save warnings: {str(e)}")


def main():
    """Example command-line entry point."""
    import argparse
    parser = argparse.ArgumentParser(description='Process fNIRS batch data')
    parser.add_argument('input_dir', help='Input directory containing .txt files')
    parser.add_argument('output_dir', help='Output directory for processed files')
    parser.add_argument('--fs', type=float, default=50.0, help='Sampling frequency')
    parser.add_argument('--sci_threshold', type=float, default=0.6,
                        help='Signal quality index threshold')
    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Process batch
    processor = BatchProcessor(fs=args.fs, sci_threshold=args.sci_threshold)
    results = processor.process_batch(args.input_dir, args.output_dir)

    print(f"\nProcessing complete. Results for {len(results)} subjects saved to {args.output_dir}")


if __name__ == "__main__":
    main()