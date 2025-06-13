#!/usr/bin/env python3
"""Minimal CLI entry point for fNIRS processing pipeline"""

import argparse
import logging
from fnirs_FullCap_2025.processing.pipeline_manager import PipelineManager


def configure_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="fNIRS FullCap Data Processing Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("input_dir", help="Directory containing raw .txt files")
    parser.add_argument("output_dir", help="Output directory for processed data")
    parser.add_argument("--fs", type=float, default=50.0,
                        help="Sampling frequency in Hz")
    parser.add_argument("--sci_threshold", type=float, default=0.6,
                        help="Signal quality index threshold (0-1)")
    return parser.parse_args()


def main():
    args = parse_args()
    configure_logging()

    pipeline = PipelineManager()
    pipeline.run_pipeline(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        fs=args.fs,
        sci_threshold=args.sci_threshold
    )


if __name__ == "__main__":
    main()