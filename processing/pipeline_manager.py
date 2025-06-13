# pipeline_manager.py
import os
import logging
from typing import List, Dict
from fnirs_FullCap_2025.processing.process_file import FullCapProcessor
from fnirs_FullCap_2025.processing.statistics import StatisticsCalculator

logger = logging.getLogger(__name__)

class PipelineManager:
    """Orchestrates batch processing of fNIRS files."""

    def __init__(self):
        self.input_base_dir = ""
        self.processor = None
        self.stats_calc = None

    def run_pipeline(
        self,
        input_dir: str,
        output_dir: str,
        fs: float = 50.0,
        sci_threshold: float = 0.6
    ) -> None:
        """
        Discover .txt files under input_dir, process each through the FullCapProcessor,
        generate summaries, and save any warnings.
        """
        self.input_base_dir = input_dir
        os.makedirs(output_dir, exist_ok=True)
        self.processor = FullCapProcessor(fs=fs, sci_threshold=sci_threshold)
        self.stats_calc = StatisticsCalculator(input_base_dir=input_dir)

        # Discover files
        txt_files = self._discover_txt_files(input_dir)
        # Group by subject folder name
        subject_map = self._group_by_subject(txt_files)

        processed_files: List[str] = []

        for subject, files in subject_map.items():
            subj_out = os.path.join(output_dir, subject)
            os.makedirs(subj_out, exist_ok=True)
            for file_path in files:
                result = self.processor.process_file(
                    file_path=file_path,
                    output_base_dir=subj_out,
                    input_base_dir=input_dir
                )
                if result is not None:
                    processed_files.append(file_path)

        # Create stats if any files processed
        if processed_files:
            stats_df = self.stats_calc.collect_statistics(processed_files, output_dir)
            self.stats_calc.create_summary_sheets(stats_df, output_dir)

        # Save warnings
        if hasattr(self.processor, 'warning_files') and self.processor.warning_files:
            warn_path = os.path.join(output_dir, 'warnings.txt')
            with open(warn_path, 'w') as wf:
                for p, msg in self.processor.warning_files:
                    wf.write(f"{p}: {msg}\n")
            logger.info(f"Warnings saved to {warn_path}")

    def _discover_txt_files(self, root_dir: str) -> List[str]:
        txts: List[str] = []
        for root, _, files in os.walk(root_dir):
            for f in files:
                if f.endswith('.txt'):
                    txts.append(os.path.join(root, f))
        logger.info(f"Found {len(txts)} .txt files")
        return sorted(txts)

    def _group_by_subject(self, paths: List[str]) -> Dict[str, List[str]]:
        groups: Dict[str, List[str]] = {}
        for p in paths:
            subject = os.path.basename(os.path.dirname(p))
            groups.setdefault(subject, []).append(p)
        return groups
